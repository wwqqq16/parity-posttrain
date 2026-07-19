"""Policy objectives for trajectory post-training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)

PolicyNormalization = Literal[
    "token",
    "sequence",
    "trajectory",
]


@dataclass(frozen=True)
class ClippedPolicyLossResult:
    """Clipped policy loss and detached diagnostics."""

    loss: torch.Tensor
    mean_ratio: float
    approximate_kl: float
    clip_fraction: float
    active_clip_fraction: float
    trainable_token_count: int


def centered_reward_advantages(
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Center sequence rewards to create advantages."""

    if rewards.ndim != 1:
        raise ValueError(
            "rewards must have shape [batch_size]"
        )

    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty")

    if not rewards.is_floating_point():
        raise ValueError("rewards must have floating dtype")

    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("rewards must be finite")

    float_rewards = rewards.float()

    return float_rewards - float_rewards.mean()


def _trajectory_row_advantages(
    batch: TrajectoryTrainingBatch,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build advantages and group IDs by task trajectory."""

    group_by_task: dict[str, int] = {}
    row_group_ids: list[int] = []
    first_rows: list[int] = []

    for row, task_id in enumerate(batch.task_ids):
        group_id = group_by_task.get(task_id)

        if group_id is None:
            group_id = len(group_by_task)
            group_by_task[task_id] = group_id
            first_rows.append(row)

        row_group_ids.append(group_id)

    row_group_indices = torch.tensor(
        row_group_ids,
        dtype=torch.long,
        device=batch.rewards.device,
    )
    first_row_indices = torch.tensor(
        first_rows,
        dtype=torch.long,
        device=batch.rewards.device,
    )
    group_rewards = batch.rewards[first_row_indices]
    expected_row_rewards = group_rewards[
        row_group_indices
    ]

    if not bool(
        torch.equal(
            batch.rewards,
            expected_row_rewards,
        )
    ):
        raise ValueError(
            "all examples in one trajectory must "
            "share the same reward"
        )

    group_advantages = centered_reward_advantages(
        group_rewards
    )
    row_advantages = group_advantages[
        row_group_indices
    ]

    return (
        row_advantages,
        row_group_indices,
        len(first_rows),
    )


def _mean_grouped_objective(
    *,
    values: torch.Tensor,
    group_indices: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Average values inside groups, then across groups."""

    group_sums = torch.zeros(
        group_count,
        dtype=values.dtype,
        device=values.device,
    ).scatter_add(
        0,
        group_indices,
        values,
    )
    group_token_counts = torch.zeros(
        group_count,
        dtype=values.dtype,
        device=values.device,
    ).scatter_add(
        0,
        group_indices,
        torch.ones_like(values),
    )

    if bool(
        torch.any(group_token_counts == 0).item()
    ):
        raise ValueError(
            "every normalization group must contain "
            "at least one trainable token"
        )

    return (
        group_sums / group_token_counts
    ).mean()


def clipped_policy_loss(
    *,
    trainer_logprobs: torch.Tensor,
    batch: TrajectoryTrainingBatch,
    clip_epsilon: float = 0.2,
    normalization: PolicyNormalization = "token",
) -> ClippedPolicyLossResult:
    """Compute a generated-token clipped policy objective."""

    batch.validate()

    if trainer_logprobs.shape != batch.input_ids.shape:
        raise ValueError(
            "trainer_logprobs must match input_ids shape"
        )

    if not trainer_logprobs.is_floating_point():
        raise ValueError(
            "trainer_logprobs must have floating dtype"
        )

    if trainer_logprobs.device != batch.loss_mask.device:
        raise ValueError(
            "trainer_logprobs and batch must share device"
        )

    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError(
            "clip_epsilon must be between zero and one"
        )

    if normalization not in {
        "token",
        "sequence",
        "trajectory",
    }:
        raise ValueError(
            "normalization must be 'token', "
            "'sequence', or 'trajectory'"
        )

    mask = batch.loss_mask
    trainable_token_count = int(mask.sum().item())

    if trainable_token_count <= 0:
        raise ValueError(
            "batch must contain trainable tokens"
        )

    current_logprobs = trainer_logprobs[mask]
    old_logprobs = batch.rollout_logprobs[
        mask
    ].to(dtype=current_logprobs.dtype)

    if not bool(
        torch.isfinite(current_logprobs).all().item()
    ):
        raise ValueError(
            "trainer_logprobs must be finite on "
            "trainable positions"
        )

    if not bool(
        torch.isfinite(old_logprobs).all().item()
    ):
        raise ValueError(
            "rollout_logprobs must be finite on "
            "trainable positions"
        )

    if normalization == "trajectory":
        (
            row_advantages,
            row_group_indices,
            group_count,
        ) = _trajectory_row_advantages(batch)
    else:
        row_advantages = centered_reward_advantages(
            batch.rewards
        )
        row_group_indices = torch.arange(
            mask.shape[0],
            dtype=torch.long,
            device=mask.device,
        )
        group_count = mask.shape[0]

    token_advantages = row_advantages.unsqueeze(
        1
    ).expand_as(trainer_logprobs)[mask].detach()

    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    unclipped_objective = ratio * token_advantages
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )
    clipped_objective = (
        clipped_ratio * token_advantages
    )

    surrogate_objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    if normalization == "token":
        loss = -surrogate_objective.mean()
    else:
        token_group_indices = (
            row_group_indices.unsqueeze(1)
            .expand_as(mask)[mask]
        )
        loss = -_mean_grouped_objective(
            values=surrogate_objective,
            group_indices=token_group_indices,
            group_count=group_count,
        )

    approximate_kl = (
        (ratio - 1.0) - log_ratio
    ).detach().mean()

    out_of_range_tokens = (
        (ratio < 1.0 - clip_epsilon)
        | (ratio > 1.0 + clip_epsilon)
    )
    active_clipped_tokens = (
        (
            (token_advantages > 0)
            & (ratio > 1.0 + clip_epsilon)
        )
        | (
            (token_advantages < 0)
            & (ratio < 1.0 - clip_epsilon)
        )
    )

    return ClippedPolicyLossResult(
        loss=loss,
        mean_ratio=float(
            ratio.detach().mean().item()
        ),
        approximate_kl=float(
            approximate_kl.item()
        ),
        clip_fraction=float(
            out_of_range_tokens.float().mean().item()
        ),
        active_clip_fraction=float(
            active_clipped_tokens.float().mean().item()
        ),
        trainable_token_count=trainable_token_count,
    )
