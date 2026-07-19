"""Result models for policy-normalization comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass

from parity_posttrain.training.objective import (
    PolicyNormalization,
)


@dataclass(frozen=True)
class TrainingComparisonTask:
    """Metadata for one selected agent trajectory."""

    task_id: str
    reward: float
    turn_count: int
    generated_token_count: int

    def validate(self) -> None:
        """Validate selected trajectory metadata."""

        if not self.task_id:
            raise ValueError("task_id must not be empty")

        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")

        if self.turn_count <= 0:
            raise ValueError("turn_count must be positive")

        if self.generated_token_count <= 0:
            raise ValueError(
                "generated_token_count must be positive"
            )


@dataclass(frozen=True)
class TaskLogprobShift:
    """Post-update logprob movement for one trajectory."""

    task_id: str
    mean_logprob_shift: float
    mean_absolute_logprob_shift: float
    max_absolute_logprob_shift: float

    def validate(self) -> None:
        """Validate per-trajectory logprob metrics."""

        if not self.task_id:
            raise ValueError("task_id must not be empty")

        values = (
            self.mean_logprob_shift,
            self.mean_absolute_logprob_shift,
            self.max_absolute_logprob_shift,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "task logprob metrics must be finite"
            )

        if self.mean_absolute_logprob_shift < 0:
            raise ValueError(
                "mean_absolute_logprob_shift must be "
                "non-negative"
            )

        if self.max_absolute_logprob_shift < 0:
            raise ValueError(
                "max_absolute_logprob_shift must be "
                "non-negative"
            )

        if (
            self.max_absolute_logprob_shift
            < self.mean_absolute_logprob_shift
        ):
            raise ValueError(
                "max_absolute_logprob_shift must not be "
                "smaller than mean_absolute_logprob_shift"
            )


@dataclass(frozen=True)
class TrainingComparisonStep:
    """Metrics from one optimizer step."""

    step_index: int
    loss: float
    gradient_norm: float
    mean_ratio: float
    approximate_kl: float
    clip_fraction: float
    trainable_token_count: int
    parameter_delta: float

    active_clip_fraction: float = 0.0

    def validate(self) -> None:
        """Validate one optimizer-step record."""

        if (
            not math.isfinite(
                self.active_clip_fraction
            )
            or not 0.0
            <= self.active_clip_fraction
            <= 1.0
        ):
            raise ValueError(
                "active_clip_fraction must be "
                "between zero and one"
            )

        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index <= 0
        ):
            raise ValueError(
                "step_index must be a positive integer"
            )

        finite_values = (
            self.loss,
            self.gradient_norm,
            self.mean_ratio,
            self.approximate_kl,
            self.clip_fraction,
            self.parameter_delta,
        )

        if not all(
            math.isfinite(value)
            for value in finite_values
        ):
            raise ValueError(
                "training-step metrics must be finite"
            )

        if self.gradient_norm < 0:
            raise ValueError(
                "gradient_norm must be non-negative"
            )

        if self.mean_ratio < 0:
            raise ValueError(
                "mean_ratio must be non-negative"
            )

        if not 0 <= self.clip_fraction <= 1:
            raise ValueError(
                "clip_fraction must be between zero and one"
            )

        if self.trainable_token_count <= 0:
            raise ValueError(
                "trainable_token_count must be positive"
            )

        if self.parameter_delta < 0:
            raise ValueError(
                "parameter_delta must be non-negative"
            )


@dataclass(frozen=True)
class TrainingComparisonRow:
    """Metrics from one policy-normalization condition."""

    normalization: PolicyNormalization
    loss: float
    gradient_norm: float
    mean_ratio: float
    approximate_kl: float
    clip_fraction: float
    trainable_token_count: int
    parameter_delta: float
    mean_absolute_logprob_shift: float
    max_absolute_logprob_shift: float
    task_shifts: tuple[TaskLogprobShift, ...]
    steps: tuple[TrainingComparisonStep, ...] = ()

    @property
    def resolved_steps(
        self,
    ) -> tuple[TrainingComparisonStep, ...]:
        """Return explicit steps or a legacy one-step record."""

        if self.steps:
            return self.steps

        return (
            TrainingComparisonStep(
                step_index=1,
                loss=self.loss,
                gradient_norm=self.gradient_norm,
                mean_ratio=self.mean_ratio,
                approximate_kl=self.approximate_kl,
                clip_fraction=self.clip_fraction,
                trainable_token_count=(
                    self.trainable_token_count
                ),
                parameter_delta=self.parameter_delta,
            ),
        )

    def validate(self) -> None:
        """Validate one comparison condition."""

        if self.normalization not in {
            "token",
            "sequence",
            "trajectory",
        }:
            raise ValueError(
                "normalization must be token, sequence, "
                "or trajectory"
            )

        finite_values = (
            self.loss,
            self.gradient_norm,
            self.mean_ratio,
            self.approximate_kl,
            self.clip_fraction,
            self.parameter_delta,
            self.mean_absolute_logprob_shift,
            self.max_absolute_logprob_shift,
        )

        if not all(
            math.isfinite(value)
            for value in finite_values
        ):
            raise ValueError(
                "comparison metrics must be finite"
            )

        if self.gradient_norm < 0:
            raise ValueError(
                "gradient_norm must be non-negative"
            )

        if self.mean_ratio < 0:
            raise ValueError(
                "mean_ratio must be non-negative"
            )

        if not 0 <= self.clip_fraction <= 1:
            raise ValueError(
                "clip_fraction must be between zero and one"
            )

        if self.trainable_token_count <= 0:
            raise ValueError(
                "trainable_token_count must be positive"
            )

        if self.parameter_delta < 0:
            raise ValueError(
                "parameter_delta must be non-negative"
            )

        if self.mean_absolute_logprob_shift < 0:
            raise ValueError(
                "mean_absolute_logprob_shift must be "
                "non-negative"
            )

        if (
            self.max_absolute_logprob_shift
            < self.mean_absolute_logprob_shift
        ):
            raise ValueError(
                "max_absolute_logprob_shift must not be "
                "smaller than mean_absolute_logprob_shift"
            )

        step_metrics = self.resolved_steps

        expected_indices = tuple(
            range(1, len(step_metrics) + 1)
        )
        actual_indices = tuple(
            step.step_index
            for step in step_metrics
        )

        if actual_indices != expected_indices:
            raise ValueError(
                "step indices must be consecutive "
                "and start at one"
            )

        for step in step_metrics:
            step.validate()

            if (
                step.trainable_token_count
                != self.trainable_token_count
            ):
                raise ValueError(
                    "step trainable_token_count must "
                    "match row"
                )

        final_step = step_metrics[-1]

        for metric_name, row_value, step_value in (
            ("loss", self.loss, final_step.loss),
            (
                "gradient_norm",
                self.gradient_norm,
                final_step.gradient_norm,
            ),
            (
                "mean_ratio",
                self.mean_ratio,
                final_step.mean_ratio,
            ),
            (
                "approximate_kl",
                self.approximate_kl,
                final_step.approximate_kl,
            ),
            (
                "clip_fraction",
                self.clip_fraction,
                final_step.clip_fraction,
            ),
            (
                "parameter_delta",
                self.parameter_delta,
                final_step.parameter_delta,
            ),
        ):
            if row_value != step_value:
                raise ValueError(
                    f"final step {metric_name} must "
                    "match row"
                )

        if not self.task_shifts:
            raise ValueError(
                "task_shifts must not be empty"
            )

        task_ids: set[str] = set()

        for shift in self.task_shifts:
            shift.validate()

            if shift.task_id in task_ids:
                raise ValueError(
                    "task_shifts must contain unique task IDs"
                )

            task_ids.add(shift.task_id)


@dataclass(frozen=True)
class TrainingComparisonSummary:
    """Reproducible policy-normalization experiment summary."""

    source_artifact: str
    model_name: str
    device: str
    dtype: str
    learning_rate: float
    clip_epsilon: float
    max_gradient_norm: float
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    tasks: tuple[TrainingComparisonTask, ...]
    rows: tuple[TrainingComparisonRow, ...]

    def validate(self) -> None:
        """Validate the complete comparison summary."""

        for text_field, text_value in (
            ("source_artifact", self.source_artifact),
            ("model_name", self.model_name),
            ("device", self.device),
            ("dtype", self.dtype),
        ):
            if not text_value:
                raise ValueError(
                    f"{text_field} must not be empty"
                )

        for numeric_field, numeric_value in (
            ("learning_rate", self.learning_rate),
            ("max_gradient_norm", self.max_gradient_norm),
        ):
            if (
                not math.isfinite(numeric_value)
                or numeric_value <= 0
            ):
                raise ValueError(
                    f"{numeric_field} must be finite "
                    "and positive"
                )

        if (
            not math.isfinite(self.clip_epsilon)
            or not 0 < self.clip_epsilon < 1
        ):
            raise ValueError(
                "clip_epsilon must be between zero and one"
            )

        if not self.trainable_parameter_names:
            raise ValueError(
                "trainable_parameter_names must not be empty"
            )

        if (
            len(set(self.trainable_parameter_names))
            != len(self.trainable_parameter_names)
        ):
            raise ValueError(
                "trainable parameter names must be unique"
            )

        if any(
            not name
            for name in self.trainable_parameter_names
        ):
            raise ValueError(
                "trainable parameter names must not be empty"
            )

        if self.trainable_parameter_count <= 0:
            raise ValueError(
                "trainable_parameter_count must be positive"
            )

        if not self.tasks:
            raise ValueError("tasks must not be empty")

        task_ids: list[str] = []

        for task in self.tasks:
            task.validate()
            task_ids.append(task.task_id)

        if len(set(task_ids)) != len(task_ids):
            raise ValueError(
                "tasks must contain unique task IDs"
            )

        if not self.rows:
            raise ValueError("rows must not be empty")

        expected_token_count = sum(
            task.generated_token_count
            for task in self.tasks
        )
        normalizations: set[PolicyNormalization] = set()
        expected_step_count = len(
            self.rows[0].resolved_steps
        )

        for row in self.rows:
            row.validate()

            if (
                len(row.resolved_steps)
                != expected_step_count
            ):
                raise ValueError(
                    "all rows must contain the same "
                    "number of steps"
                )

            if row.normalization in normalizations:
                raise ValueError(
                    "rows must contain unique normalizations"
                )

            normalizations.add(row.normalization)

            if (
                row.trainable_token_count
                != expected_token_count
            ):
                raise ValueError(
                    "row trainable_token_count does not "
                    "match selected tasks"
                )

            row_task_ids = [
                shift.task_id
                for shift in row.task_shifts
            ]

            if row_task_ids != task_ids:
                raise ValueError(
                    "row task shifts must match selected "
                    "tasks and order"
                )


def training_comparison_to_dict(
    summary: TrainingComparisonSummary,
) -> dict[str, object]:
    """Convert a training comparison to JSON-safe data."""

    summary.validate()

    return {
        "schema_version": 2,
        "source": {
            "artifact": summary.source_artifact,
            "model_name": summary.model_name,
            "device": summary.device,
            "dtype": summary.dtype,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "reward": task.reward,
                    "turn_count": task.turn_count,
                    "generated_token_count": (
                        task.generated_token_count
                    ),
                }
                for task in summary.tasks
            ],
        },
        "training": {
            "learning_rate": summary.learning_rate,
            "clip_epsilon": summary.clip_epsilon,
            "max_gradient_norm": (
                summary.max_gradient_norm
            ),
            "steps": len(
                summary.rows[0].resolved_steps
            ),
            "trainable_parameter_names": list(
                summary.trainable_parameter_names
            ),
            "trainable_parameter_count": (
                summary.trainable_parameter_count
            ),
        },
        "results": [
            {
                "normalization": row.normalization,
                "loss": row.loss,
                "gradient_norm": row.gradient_norm,
                "mean_ratio": row.mean_ratio,
                "approximate_kl": row.approximate_kl,
                "clip_fraction": row.clip_fraction,
                "trainable_token_count": (
                    row.trainable_token_count
                ),
                "parameter_delta": row.parameter_delta,
                "mean_absolute_logprob_shift": (
                    row.mean_absolute_logprob_shift
                ),
                "max_absolute_logprob_shift": (
                    row.max_absolute_logprob_shift
                ),
                "steps": [
                    {
                        "step_index": step.step_index,
                        "loss": step.loss,
                        "gradient_norm": (
                            step.gradient_norm
                        ),
                        "mean_ratio": step.mean_ratio,
                        "approximate_kl": (
                            step.approximate_kl
                        ),
                        "clip_fraction": (
                            step.clip_fraction
                        ),
                        "trainable_token_count": (
                            step.trainable_token_count
                        ),
                        "parameter_delta": (
                            step.parameter_delta
                        ),
                    }
                    for step in row.resolved_steps
                ],
                "task_shifts": [
                    {
                        "task_id": shift.task_id,
                        "mean_logprob_shift": (
                            shift.mean_logprob_shift
                        ),
                        "mean_absolute_logprob_shift": (
                            shift.mean_absolute_logprob_shift
                        ),
                        "max_absolute_logprob_shift": (
                            shift.max_absolute_logprob_shift
                        ),
                    }
                    for shift in row.task_shifts
                ],
            }
            for row in summary.rows
        ],
    }
