"""Multi-step trajectory policy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)
from parity_posttrain.training.step import (
    TrainingStepObserver,
    TrainingStepResult,
    run_clipped_policy_step,
)


@dataclass(frozen=True)
class TrainingLoopResult:
    """Metrics from a fixed number of policy-update steps."""

    requested_steps: int
    normalization: PolicyNormalization
    steps: tuple[TrainingStepResult, ...]
    parameter_deltas: tuple[float, ...]

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

    @property
    def final_parameter_delta(self) -> float:
        """Return the cumulative parameter delta after the final step."""

        if not self.parameter_deltas:
            raise ValueError(
                "training loop contains no parameter deltas"
            )

        return self.parameter_deltas[-1]

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

        if len(self.parameter_deltas) != self.requested_steps:
            raise ValueError(
                "parameter deltas must match requested_steps"
            )

        if not self.steps:
            raise ValueError(
                "steps must not be empty"
            )

        if not all(
            math.isfinite(delta) and delta >= 0
            for delta in self.parameter_deltas
        ):
            raise ValueError(
                "parameter deltas must be finite and non-negative"
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


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, ...]:
    """Return unique tensors managed by an optimizer."""

    parameters: list[torch.Tensor] = []
    seen_ids: set[int] = set()

    for group in optimizer.param_groups:
        for candidate in group["params"]:
            if not isinstance(candidate, torch.Tensor):
                raise TypeError(
                    "optimizer parameters must be tensors"
                )

            candidate_id = id(candidate)
            if candidate_id in seen_ids:
                continue

            seen_ids.add(candidate_id)
            parameters.append(candidate)

    if not parameters:
        raise ValueError(
            "optimizer must contain at least one parameter"
        )

    return tuple(parameters)


def _maximum_parameter_delta(
    parameters: tuple[torch.Tensor, ...],
    snapshots: tuple[torch.Tensor, ...],
) -> float:
    """Return maximum absolute change from initial parameters."""

    return max(
        float(
            (
                parameter.detach()
                - snapshot
            )
            .abs()
            .max()
            .item()
        )
        for parameter, snapshot in zip(
            parameters,
            snapshots,
            strict=True,
        )
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
    observer: TrainingStepObserver | None = None,
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

    parameters = _optimizer_parameters(optimizer)
    snapshots = tuple(
        parameter.detach().clone()
        for parameter in parameters
    )

    results: list[TrainingStepResult] = []
    parameter_deltas: list[float] = []

    for step_index in range(1, steps + 1):
        result = run_clipped_policy_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_epsilon=clip_epsilon,
            max_gradient_norm=max_gradient_norm,
            normalization=normalization,
            step_index=step_index,
            observer=observer,
        )
        results.append(result)
        parameter_deltas.append(
            _maximum_parameter_delta(
                parameters,
                snapshots,
            )
        )

    loop_result = TrainingLoopResult(
        requested_steps=steps,
        normalization=normalization,
        steps=tuple(results),
        parameter_deltas=tuple(parameter_deltas),
    )
    loop_result.validate()

    return loop_result
