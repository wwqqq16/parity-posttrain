"""Policy objectives for trajectory post-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)


@dataclass(frozen=True)
class ClippedPolicyLossResult:
    """Clipped policy loss and detached diagnostics."""

    loss: torch.Tensor
    mean_ratio: float
    approximate_kl: float
    clip_fraction: float
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


def clipped_policy_loss(
    *,
    trainer_logprobs: torch.Tensor,
    batch: TrajectoryTrainingBatch,
    clip_epsilon: float = 0.2,
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

    sequence_advantages = centered_reward_advantages(
        batch.rewards
    )
    token_advantages = sequence_advantages.unsqueeze(
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
    loss = -surrogate_objective.mean()

    approximate_kl = (
        (ratio - 1.0) - log_ratio
    ).detach().mean()

    clipped_tokens = (
        (ratio < 1.0 - clip_epsilon)
        | (ratio > 1.0 + clip_epsilon)
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
            clipped_tokens.float().mean().item()
        ),
        trainable_token_count=trainable_token_count,
    )
