"""Execute one closed-loop policy-training experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.closed_loop import (
    ClosedLoopSummary,
    ClosedLoopTaskComparison,
    ClosedLoopTaskSnapshot,
    build_closed_loop_aggregate,
)
from parity_posttrain.training.comparison import (
    TrainingComparisonStep,
)
from parity_posttrain.training.loop import (
    run_clipped_policy_training,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)

RerolloutCallback = Callable[
    [torch.nn.Module],
    Sequence[ClosedLoopTaskSnapshot],
]


def _validate_snapshots(
    snapshots: Sequence[ClosedLoopTaskSnapshot],
    *,
    name: str,
) -> tuple[ClosedLoopTaskSnapshot, ...]:
    """Validate and materialize one snapshot sequence."""

    resolved = tuple(snapshots)

    if not resolved:
        raise ValueError(
            f"{name} snapshots must not be empty"
        )

    task_ids: set[str] = set()

    for snapshot in resolved:
        snapshot.validate()

        if snapshot.task_id in task_ids:
            raise ValueError(
                f"{name} task IDs must be unique"
            )

        task_ids.add(snapshot.task_id)

    return resolved


def _build_training_steps(
    *,
    step_results: Sequence[object],
    parameter_deltas: Sequence[float],
) -> tuple[TrainingComparisonStep, ...]:
    """Convert loop metrics into closed-loop step records."""

    from parity_posttrain.training.step import (
        TrainingStepResult,
    )

    resolved_results = tuple(step_results)
    resolved_deltas = tuple(parameter_deltas)

    if len(resolved_results) != len(resolved_deltas):
        raise ValueError(
            "step results and parameter deltas must match"
        )

    comparison_steps: list[
        TrainingComparisonStep
    ] = []

    for step_index, (
        step_result,
        parameter_delta,
    ) in enumerate(
        zip(
            resolved_results,
            resolved_deltas,
            strict=True,
        ),
        start=1,
    ):
        if not isinstance(
            step_result,
            TrainingStepResult,
        ):
            raise TypeError(
                "step results must be "
                "TrainingStepResult instances"
            )

        comparison_steps.append(
            TrainingComparisonStep(
                step_index=step_index,
                loss=step_result.loss,
                gradient_norm=(
                    step_result.gradient_norm
                ),
                mean_ratio=step_result.mean_ratio,
                approximate_kl=(
                    step_result.approximate_kl
                ),
                clip_fraction=(
                    step_result.clip_fraction
                ),
                trainable_token_count=(
                    step_result.trainable_token_count
                ),
                parameter_delta=parameter_delta,
            )
        )

    return tuple(comparison_steps)


def run_closed_loop_experiment(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrajectoryTrainingBatch,
    before_snapshots: Sequence[
        ClosedLoopTaskSnapshot
    ],
    rerollout: RerolloutCallback,
    source_artifact: str,
    model_name: str,
    steps: int,
    normalization: PolicyNormalization = "trajectory",
    clip_epsilon: float = 0.2,
    max_gradient_norm: float = 1.0,
) -> ClosedLoopSummary:
    """Train a model and evaluate it through a rerollout callback.

    The callback receives the updated model in evaluation mode.
    Model parameters remain updated after this function returns,
    while the original train/eval mode is restored.
    """

    if not source_artifact:
        raise ValueError(
            "source_artifact must not be empty"
        )

    if not model_name:
        raise ValueError(
            "model_name must not be empty"
        )

    before = _validate_snapshots(
        before_snapshots,
        name="before",
    )
    batch.validate()

    original_training_mode = model.training

    loop_result = run_clipped_policy_training(
        model=model,
        optimizer=optimizer,
        batch=batch,
        steps=steps,
        clip_epsilon=clip_epsilon,
        max_gradient_norm=max_gradient_norm,
        normalization=normalization,
    )

    try:
        model.eval()
        after = _validate_snapshots(
            rerollout(model),
            name="after",
        )
    finally:
        model.train(original_training_mode)

    before_ids = tuple(
        snapshot.task_id
        for snapshot in before
    )
    after_ids = tuple(
        snapshot.task_id
        for snapshot in after
    )

    if after_ids != before_ids:
        raise ValueError(
            "after task IDs must match before task IDs "
            "and order"
        )

    tasks = tuple(
        ClosedLoopTaskComparison(
            before=before_snapshot,
            after=after_snapshot,
        )
        for before_snapshot, after_snapshot in zip(
            before,
            after,
            strict=True,
        )
    )

    summary = ClosedLoopSummary(
        source_artifact=source_artifact,
        model_name=model_name,
        normalization=normalization,
        optimizer_steps=steps,
        training_steps=_build_training_steps(
            step_results=loop_result.steps,
            parameter_deltas=(
                loop_result.parameter_deltas
            ),
        ),
        before=build_closed_loop_aggregate(before),
        after=build_closed_loop_aggregate(after),
        tasks=tasks,
    )
    summary.validate()

    return summary
