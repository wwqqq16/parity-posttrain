"""Run controlled policy-normalization comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.comparison import (
    TaskLogprobShift,
    TrainingComparisonRow,
    TrainingComparisonStep,
    TrainingComparisonSummary,
    TrainingComparisonTask,
)
from parity_posttrain.training.example import (
    TrajectoryTrainingExample,
)
from parity_posttrain.training.logprobs import (
    rescore_training_batch,
)
from parity_posttrain.training.loop import (
    run_clipped_policy_training,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)


def select_training_examples(
    examples: Sequence[TrajectoryTrainingExample],
    task_ids: Sequence[str],
) -> tuple[TrajectoryTrainingExample, ...]:
    """Select every turn for requested tasks in requested order."""

    requested = tuple(task_ids)

    if not requested:
        raise ValueError("task_ids must not be empty")

    if any(not task_id.strip() for task_id in requested):
        raise ValueError(
            "task_ids must contain non-empty strings"
        )

    if len(set(requested)) != len(requested):
        raise ValueError("task_ids must be unique")

    grouped: dict[
        str,
        list[TrajectoryTrainingExample],
    ] = {}

    for example in examples:
        example.validate()

        if example.task_id in requested:
            grouped.setdefault(
                example.task_id,
                [],
            ).append(example)

    missing = [
        task_id
        for task_id in requested
        if task_id not in grouped
    ]

    if missing:
        raise ValueError(
            "requested tasks were not found: "
            + ", ".join(missing)
        )

    selected: list[TrajectoryTrainingExample] = []

    for task_id in requested:
        task_examples = sorted(
            grouped[task_id],
            key=lambda example: example.turn_index,
        )
        turn_indices = [
            example.turn_index
            for example in task_examples
        ]

        if (
            len(set(turn_indices))
            != len(turn_indices)
        ):
            raise ValueError(
                f"task {task_id} contains duplicate "
                "turn indices"
            )

        rewards = {
            example.reward
            for example in task_examples
        }

        if len(rewards) != 1:
            raise ValueError(
                f"task {task_id} contains inconsistent "
                "rewards"
            )

        selected.extend(task_examples)

    model_names = {
        example.model_name
        for example in selected
    }

    if len(model_names) != 1:
        raise ValueError(
            "selected examples must use one model"
        )

    return tuple(selected)


def _build_task_metadata(
    batch: TrajectoryTrainingBatch,
) -> tuple[TrainingComparisonTask, ...]:
    """Aggregate batch rows into complete trajectories."""

    rows_by_task: dict[str, list[int]] = {}

    for row, task_id in enumerate(batch.task_ids):
        rows_by_task.setdefault(task_id, []).append(row)

    tasks: list[TrainingComparisonTask] = []

    for task_id, rows in rows_by_task.items():
        reward = float(
            batch.rewards[rows[0]].item()
        )
        row_rewards = [
            float(batch.rewards[row].item())
            for row in rows
        ]

        if any(
            row_reward != reward
            for row_reward in row_rewards
        ):
            raise ValueError(
                f"task {task_id} contains inconsistent "
                "batch rewards"
            )

        turn_indices = [
            batch.turn_indices[row]
            for row in rows
        ]

        if (
            len(set(turn_indices))
            != len(turn_indices)
        ):
            raise ValueError(
                f"task {task_id} contains duplicate "
                "turn indices"
            )

        generated_token_count = sum(
            int(batch.loss_mask[row].sum().item())
            for row in rows
        )

        task = TrainingComparisonTask(
            task_id=task_id,
            reward=reward,
            turn_count=len(rows),
            generated_token_count=(
                generated_token_count
            ),
        )
        task.validate()
        tasks.append(task)

    return tuple(tasks)


def _resolve_trainable_parameters(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[torch.nn.Parameter, ...],
]:
    """Resolve an exact set of trainable model parameters."""

    requested = tuple(parameter_names)

    if not requested:
        raise ValueError(
            "trainable_parameter_names must not be empty"
        )

    if any(not name.strip() for name in requested):
        raise ValueError(
            "trainable parameter names must not be empty"
        )

    if len(set(requested)) != len(requested):
        raise ValueError(
            "trainable parameter names must be unique"
        )

    available = dict(model.named_parameters())
    missing = [
        name
        for name in requested
        if name not in available
    ]

    if missing:
        raise ValueError(
            "unknown trainable parameters: "
            + ", ".join(missing)
        )

    parameters = tuple(
        available[name]
        for name in requested
    )

    return requested, parameters


def _restore_parameter_values(
    parameters: Sequence[torch.nn.Parameter],
    snapshots: Sequence[torch.Tensor],
) -> None:
    """Restore trainable parameters to their baseline values."""

    with torch.no_grad():
        for parameter, snapshot in zip(
            parameters,
            snapshots,
            strict=True,
        ):
            parameter.copy_(snapshot)


def _build_task_shifts(
    *,
    shifts: torch.Tensor,
    batch: TrajectoryTrainingBatch,
    tasks: Sequence[TrainingComparisonTask],
) -> tuple[TaskLogprobShift, ...]:
    """Aggregate token logprob shifts by complete task."""

    results: list[TaskLogprobShift] = []

    for task in tasks:
        selected_rows = torch.tensor(
            [
                task_id == task.task_id
                for task_id in batch.task_ids
            ],
            dtype=torch.bool,
            device=batch.loss_mask.device,
        )
        task_mask = (
            batch.loss_mask
            & selected_rows.unsqueeze(1)
        )
        values = shifts[task_mask]

        if values.numel() == 0:
            raise ValueError(
                f"task {task.task_id} has no selected tokens"
            )

        absolute_values = values.abs()
        result = TaskLogprobShift(
            task_id=task.task_id,
            mean_logprob_shift=float(
                values.mean().item()
            ),
            mean_absolute_logprob_shift=float(
                absolute_values.mean().item()
            ),
            max_absolute_logprob_shift=float(
                absolute_values.max().item()
            ),
        )
        result.validate()
        results.append(result)

    return tuple(results)


def run_training_comparison(
    *,
    model: torch.nn.Module,
    batch: TrajectoryTrainingBatch,
    source_artifact: str,
    model_name: str,
    trainable_parameter_names: Sequence[str],
    normalizations: Sequence[
        PolicyNormalization
    ] = (
        "token",
        "sequence",
        "trajectory",
    ),
    steps: int = 1,
    learning_rate: float = 0.05,
    clip_epsilon: float = 0.2,
    max_gradient_norm: float = 1.0,
) -> TrainingComparisonSummary:
    """Compare controlled updates from identical model weights."""

    batch.validate()

    if not source_artifact:
        raise ValueError(
            "source_artifact must not be empty"
        )

    if not model_name:
        raise ValueError("model_name must not be empty")

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise ValueError(
            "steps must be a positive integer"
        )

    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be finite and positive"
        )

    if (
        not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValueError(
            "clip_epsilon must be between zero and one"
        )

    if (
        not math.isfinite(max_gradient_norm)
        or max_gradient_norm <= 0
    ):
        raise ValueError(
            "max_gradient_norm must be finite and positive"
        )

    normalization_conditions = tuple(
        normalizations
    )

    if not normalization_conditions:
        raise ValueError(
            "normalizations must not be empty"
        )

    if (
        len(set(normalization_conditions))
        != len(normalization_conditions)
    ):
        raise ValueError(
            "normalizations must be unique"
        )

    (
        resolved_names,
        trainable_parameters,
    ) = _resolve_trainable_parameters(
        model,
        trainable_parameter_names,
    )

    all_parameters = tuple(model.parameters())
    original_requires_grad = tuple(
        parameter.requires_grad
        for parameter in all_parameters
    )
    original_training_mode = model.training
    parameter_snapshots = tuple(
        parameter.detach().clone()
        for parameter in trainable_parameters
    )
    tasks = _build_task_metadata(batch)

    try:
        for parameter in all_parameters:
            parameter.requires_grad_(False)

        for parameter in trainable_parameters:
            parameter.requires_grad_(True)

        model.eval()

        with torch.no_grad():
            baseline_logprobs = cast(
                torch.FloatTensor,
                rescore_training_batch(
                    model=model,
                    batch=batch,
                )
                .detach()
                .clone(),
            )

        controlled_batch = replace(
            batch,
            rollout_logprobs=baseline_logprobs,
        )
        controlled_batch.validate()

        rows: list[TrainingComparisonRow] = []

        for normalization in normalization_conditions:
            _restore_parameter_values(
                trainable_parameters,
                parameter_snapshots,
            )
            model.zero_grad(set_to_none=True)

            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=learning_rate,
            )
            loop_result = run_clipped_policy_training(
                model=model,
                optimizer=optimizer,
                batch=controlled_batch,
                steps=steps,
                clip_epsilon=clip_epsilon,
                max_gradient_norm=max_gradient_norm,
                normalization=normalization,
            )
            comparison_steps = tuple(
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
                for step_index, (
                    step_result,
                    parameter_delta,
                ) in enumerate(
                    zip(
                        loop_result.steps,
                        loop_result.parameter_deltas,
                        strict=True,
                    ),
                    start=1,
                )
            )
            final_step = loop_result.final_step

            model.eval()

            with torch.no_grad():
                after_logprobs = (
                    rescore_training_batch(
                        model=model,
                        batch=controlled_batch,
                    )
                    .detach()
                    .clone()
                )

            shifts = (
                after_logprobs
                - baseline_logprobs
            )
            selected_shifts = shifts[
                controlled_batch.loss_mask
            ]
            absolute_shifts = selected_shifts.abs()

            row = TrainingComparisonRow(
                normalization=normalization,
                loss=final_step.loss,
                gradient_norm=final_step.gradient_norm,
                mean_ratio=final_step.mean_ratio,
                approximate_kl=final_step.approximate_kl,
                clip_fraction=final_step.clip_fraction,
                trainable_token_count=(
                    final_step.trainable_token_count
                ),
                parameter_delta=(
                    loop_result.final_parameter_delta
                ),
                mean_absolute_logprob_shift=float(
                    absolute_shifts.mean().item()
                ),
                max_absolute_logprob_shift=float(
                    absolute_shifts.max().item()
                ),
                task_shifts=_build_task_shifts(
                    shifts=shifts,
                    batch=controlled_batch,
                    tasks=tasks,
                ),
                steps=comparison_steps,
            )
            row.validate()
            rows.append(row)

        summary = TrainingComparisonSummary(
            source_artifact=source_artifact,
            model_name=model_name,
            device=str(
                trainable_parameters[0].device
            ),
            dtype=str(
                trainable_parameters[0].dtype
            ),
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            max_gradient_norm=max_gradient_norm,
            trainable_parameter_names=resolved_names,
            trainable_parameter_count=sum(
                parameter.numel()
                for parameter in trainable_parameters
            ),
            tasks=tasks,
            rows=tuple(rows),
        )
        summary.validate()

        return summary
    finally:
        model.zero_grad(set_to_none=True)
        _restore_parameter_values(
            trainable_parameters,
            parameter_snapshots,
        )

        for parameter, requires_grad in zip(
            all_parameters,
            original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

        model.train(original_training_mode)
