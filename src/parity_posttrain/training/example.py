"""Training examples derived from agent rollout generations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

IGNORE_INDEX = -100


def _normalize_token_ids(
    token_ids: Sequence[int],
    *,
    field: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    """Validate and freeze a token-ID sequence."""

    normalized = tuple(token_ids)

    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty")

    for token_id in normalized:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise ValueError(
                f"{field} must contain non-negative integers"
            )

    return normalized


def _normalize_logprobs(
    logprobs: Sequence[float],
) -> tuple[float, ...]:
    """Validate and freeze rollout token log-probabilities."""

    normalized: list[float] = []

    for value in logprobs:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                "rollout_logprobs must contain numeric values"
            )

        float_value = float(value)

        if not math.isfinite(float_value):
            raise ValueError(
                "rollout_logprobs must contain finite values"
            )

        normalized.append(float_value)

    return tuple(normalized)


def build_generated_token_loss_mask(
    prompt_token_count: int,
    generated_token_count: int,
) -> tuple[int, ...]:
    """Mask prompt tokens and select generated tokens."""

    if prompt_token_count < 0:
        raise ValueError(
            "prompt_token_count must be non-negative"
        )

    if generated_token_count <= 0:
        raise ValueError(
            "generated_token_count must be positive"
        )

    return (
        (0,) * prompt_token_count
        + (1,) * generated_token_count
    )


def build_generated_token_labels(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[int, ...]:
    """Build causal-LM labels with prompt positions ignored."""

    prompt = _normalize_token_ids(
        prompt_token_ids,
        field="prompt_token_ids",
        allow_empty=True,
    )
    generated = _normalize_token_ids(
        generated_token_ids,
        field="generated_token_ids",
        allow_empty=False,
    )

    return (
        (ignore_index,) * len(prompt)
        + generated
    )


@dataclass(frozen=True)
class TrajectoryTrainingExample:
    """One trainable agent generation and its rollout metadata."""

    task_id: str
    turn_index: int
    status: str
    model_name: str
    reward: float
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        """Return the complete prompt-plus-generation sequence."""

        return (
            self.prompt_token_ids
            + self.generated_token_ids
        )

    @property
    def labels(self) -> tuple[int, ...]:
        """Return generated-only causal-LM labels."""

        return build_generated_token_labels(
            self.prompt_token_ids,
            self.generated_token_ids,
        )

    @property
    def loss_mask(self) -> tuple[int, ...]:
        """Return a generated-token-only binary loss mask."""

        return build_generated_token_loss_mask(
            len(self.prompt_token_ids),
            len(self.generated_token_ids),
        )

    def validate(self) -> None:
        """Validate training-example invariants."""

        if not self.task_id:
            raise ValueError("task_id must not be empty")

        if self.turn_index < 0:
            raise ValueError(
                "turn_index must be non-negative"
            )

        if not self.status:
            raise ValueError("status must not be empty")

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")

        if not self.prompt_token_ids:
            raise ValueError(
                "prompt_token_ids must not be empty"
            )

        if not self.generated_token_ids:
            raise ValueError(
                "generated_token_ids must not be empty"
            )

        if (
            len(self.generated_token_ids)
            != len(self.rollout_logprobs)
        ):
            raise ValueError(
                "generated_token_ids and rollout_logprobs "
                "must have equal length"
            )

        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                "input_ids and labels must have equal length"
            )

        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError(
                "input_ids and loss_mask must have equal length"
            )

        if sum(self.loss_mask) != len(
            self.generated_token_ids
        ):
            raise ValueError(
                "loss_mask must select every generated token"
            )


def build_trajectory_training_example(
    *,
    task_id: str,
    turn_index: int,
    status: str,
    model_name: str,
    reward: float,
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    rollout_logprobs: Sequence[float],
) -> TrajectoryTrainingExample:
    """Build and validate one immutable training example."""

    prompt = _normalize_token_ids(
        prompt_token_ids,
        field="prompt_token_ids",
        allow_empty=False,
    )
    generated = _normalize_token_ids(
        generated_token_ids,
        field="generated_token_ids",
        allow_empty=False,
    )
    normalized_logprobs = _normalize_logprobs(
        rollout_logprobs
    )

    example = TrajectoryTrainingExample(
        task_id=task_id,
        turn_index=turn_index,
        status=status,
        model_name=model_name,
        reward=float(reward),
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        rollout_logprobs=normalized_logprobs,
    )
    example.validate()

    return example
