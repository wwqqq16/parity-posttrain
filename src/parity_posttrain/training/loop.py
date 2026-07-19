"""Multi-step trajectory policy optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)
from parity_posttrain.training.step import (
    TrainingStepResult,
    run_clipped_policy_step,
)


@dataclass(frozen=True)
class TrainingLoopResult:
    """Metrics from a fixed number of policy-update steps."""

    requested_steps: int
    normalization: PolicyNormalization
    steps: tuple[TrainingStepResult, ...]

    @property
    def completed_steps(self) -> int:
        """Return the number of completed optimizer steps."""

        return len(self.steps)

    @property
    def final_step(self) -> TrainingStepResult:
        """Return metrics from the final optimizer step."""

        if not self.steps:
            raise ValueError(
                "training loop contains no completed steps"
            )

        return self.steps[-1]

    def validate(self) -> None:
        """Validate loop-level metrics."""

        if (
            isinstance(self.requested_steps, bool)
            or not isinstance(self.requested_steps, int)
            or self.requested_steps <= 0
        ):
            raise ValueError(
                "requested_steps must be a positive integer"
            )

        if len(self.steps) != self.requested_steps:
            raise ValueError(
                "completed steps must match requested_steps"
            )

        if not self.steps:
            raise ValueError(
                "steps must not be empty"
            )

        trainable_token_count = (
            self.steps[0].trainable_token_count
        )

        for step in self.steps:
            if step.normalization != self.normalization:
                raise ValueError(
                    "all steps must use the loop "
                    "normalization"
                )

            if (
                step.trainable_token_count
                != trainable_token_count
            ):
                raise ValueError(
                    "all steps must use the same "
                    "trainable token count"
                )


def run_clipped_policy_training(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrajectoryTrainingBatch,
    steps: int,
    clip_epsilon: float = 0.2,
    max_gradient_norm: float = 1.0,
    normalization: PolicyNormalization = "token",
) -> TrainingLoopResult:
    """Run repeated updates against fixed rollout logprobs."""

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise ValueError(
            "steps must be a positive integer"
        )

    batch.validate()
    results: list[TrainingStepResult] = []

    for _ in range(steps):
        result = run_clipped_policy_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_epsilon=clip_epsilon,
            max_gradient_norm=max_gradient_norm,
            normalization=normalization,
        )
        results.append(result)

    loop_result = TrainingLoopResult(
        requested_steps=steps,
        normalization=normalization,
        steps=tuple(results),
    )
    loop_result.validate()

    return loop_result
