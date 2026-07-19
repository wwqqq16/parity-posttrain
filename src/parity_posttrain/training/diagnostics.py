"""Token-level diagnostics for clipped policy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
    _trajectory_row_advantages,
    centered_reward_advantages,
)

ClipDirection = Literal[
    "none",
    "positive_high",
    "negative_low",
]


@dataclass(frozen=True)
class TokenClippingDiagnostic:
    """Clipping state for one generated token."""

    flat_token_index: int
    batch_row: int
    task_id: str
    turn_index: int
    generated_token_index: int
    sequence_position: int
    token_id: int
    advantage: float
    old_logprob: float
    current_logprob: float
    ratio: float
    out_of_range: bool
    active_clipped: bool
    clip_direction: ClipDirection


def build_token_clipping_diagnostics(
    *,
    trainer_logprobs: torch.Tensor,
    batch: TrajectoryTrainingBatch,
    clip_epsilon: float = 0.2,
    normalization: PolicyNormalization = "token",
) -> tuple[TokenClippingDiagnostic, ...]:
    """Build diagnostics for every generated token."""

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
    positions = mask.nonzero(as_tuple=False)

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
        row_advantages, _, _ = (
            _trajectory_row_advantages(batch)
        )
    else:
        row_advantages = centered_reward_advantages(
            batch.rewards
        )

    token_advantages = (
        row_advantages.unsqueeze(1)
        .expand_as(trainer_logprobs)[mask]
        .detach()
    )

    ratios = torch.exp(
        current_logprobs - old_logprobs
    ).detach()

    if not bool(torch.isfinite(ratios).all().item()):
        raise ValueError(
            "policy ratios must be finite"
        )

    out_of_range = (
        (ratios < 1.0 - clip_epsilon)
        | (ratios > 1.0 + clip_epsilon)
    )

    active_clipped = (
        (
            (token_advantages > 0)
            & (ratios > 1.0 + clip_epsilon)
        )
        | (
            (token_advantages < 0)
            & (ratios < 1.0 - clip_epsilon)
        )
    )

    diagnostics: list[TokenClippingDiagnostic] = []

    for flat_index, position in enumerate(positions):
        row = int(position[0].item())
        sequence_position = int(position[1].item())

        generated_token_index = (
            int(
                mask[
                    row,
                    : sequence_position + 1,
                ]
                .sum()
                .item()
            )
            - 1
        )

        advantage = float(
            token_advantages[flat_index].item()
        )
        ratio = float(ratios[flat_index].item())
        is_active = bool(
            active_clipped[flat_index].item()
        )

        clip_direction: ClipDirection = "none"

        if is_active and advantage > 0:
            clip_direction = "positive_high"
        elif is_active and advantage < 0:
            clip_direction = "negative_low"

        diagnostics.append(
            TokenClippingDiagnostic(
                flat_token_index=flat_index,
                batch_row=row,
                task_id=batch.task_ids[row],
                turn_index=batch.turn_indices[row],
                generated_token_index=(
                    generated_token_index
                ),
                sequence_position=sequence_position,
                token_id=int(
                    batch.input_ids[
                        row,
                        sequence_position,
                    ].item()
                ),
                advantage=advantage,
                old_logprob=float(
                    old_logprobs[flat_index].item()
                ),
                current_logprob=float(
                    current_logprobs[
                        flat_index
                    ].detach().item()
                ),
                ratio=ratio,
                out_of_range=bool(
                    out_of_range[flat_index].item()
                ),
                active_clipped=is_active,
                clip_direction=clip_direction,
            )
        )

    return tuple(diagnostics)
