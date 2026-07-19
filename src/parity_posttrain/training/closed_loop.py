"""Closed-loop rollout, training, and rerollout result models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from parity_posttrain.training.comparison import (
    TrainingComparisonStep,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)


def _validate_non_negative_integer(
    *,
    name: str,
    value: int,
) -> None:
    """Validate one non-negative integer field."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )


@dataclass(frozen=True)
class ClosedLoopTaskSnapshot:
    """One task outcome before or after policy training."""

    task_id: str
    category: str
    status: str
    reward: float
    answer_correct: bool
    generation_count: int
    generated_token_count: int

    def validate(self) -> None:
        """Validate one task snapshot."""

        if not self.task_id:
            raise ValueError(
                "task_id must not be empty"
            )

        if not self.category:
            raise ValueError(
                "category must not be empty"
            )

        if not self.status:
            raise ValueError(
                "status must not be empty"
            )

        if not math.isfinite(self.reward):
            raise ValueError(
                "reward must be finite"
            )

        if not isinstance(self.answer_correct, bool):
            raise ValueError(
                "answer_correct must be boolean"
            )

        _validate_non_negative_integer(
            name="generation_count",
            value=self.generation_count,
        )
        _validate_non_negative_integer(
            name="generated_token_count",
            value=self.generated_token_count,
        )


@dataclass(frozen=True)
class ClosedLoopAggregate:
    """Aggregate task metrics for one rollout phase."""

    task_count: int
    completed_count: int
    correct_answer_count: int
    total_reward: float
    mean_reward: float
    generation_count: int
    generated_token_count: int

    @property
    def completion_rate(self) -> float:
        """Return the fraction of completed tasks."""

        return self.completed_count / self.task_count

    @property
    def answer_accuracy(self) -> float:
        """Return the fraction of correct answers."""

        return self.correct_answer_count / self.task_count

    def validate(self) -> None:
        """Validate aggregate task metrics."""

        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count <= 0
        ):
            raise ValueError(
                "task_count must be a positive integer"
            )

        for name, value in (
            ("completed_count", self.completed_count),
            (
                "correct_answer_count",
                self.correct_answer_count,
            ),
            ("generation_count", self.generation_count),
            (
                "generated_token_count",
                self.generated_token_count,
            ),
        ):
            _validate_non_negative_integer(
                name=name,
                value=value,
            )

        if self.completed_count > self.task_count:
            raise ValueError(
                "completed_count must not exceed task_count"
            )

        if self.correct_answer_count > self.task_count:
            raise ValueError(
                "correct_answer_count must not exceed "
                "task_count"
            )

        if not math.isfinite(self.total_reward):
            raise ValueError(
                "total_reward must be finite"
            )

        if not math.isfinite(self.mean_reward):
            raise ValueError(
                "mean_reward must be finite"
            )

        expected_mean = (
            self.total_reward / self.task_count
        )

        if not math.isclose(
            self.mean_reward,
            expected_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "mean_reward must match total_reward "
                "divided by task_count"
            )


def build_closed_loop_aggregate(
    snapshots: Sequence[ClosedLoopTaskSnapshot],
) -> ClosedLoopAggregate:
    """Build aggregate metrics from task snapshots."""

    resolved_snapshots = tuple(snapshots)

    if not resolved_snapshots:
        raise ValueError(
            "snapshots must not be empty"
        )

    for snapshot in resolved_snapshots:
        snapshot.validate()

    total_reward = sum(
        snapshot.reward
        for snapshot in resolved_snapshots
    )

    aggregate = ClosedLoopAggregate(
        task_count=len(resolved_snapshots),
        completed_count=sum(
            snapshot.status == "completed"
            for snapshot in resolved_snapshots
        ),
        correct_answer_count=sum(
            snapshot.answer_correct
            for snapshot in resolved_snapshots
        ),
        total_reward=total_reward,
        mean_reward=(
            total_reward / len(resolved_snapshots)
        ),
        generation_count=sum(
            snapshot.generation_count
            for snapshot in resolved_snapshots
        ),
        generated_token_count=sum(
            snapshot.generated_token_count
            for snapshot in resolved_snapshots
        ),
    )
    aggregate.validate()

    return aggregate


@dataclass(frozen=True)
class ClosedLoopTaskComparison:
    """Before-and-after result for one task."""

    before: ClosedLoopTaskSnapshot
    after: ClosedLoopTaskSnapshot

    @property
    def task_id(self) -> str:
        """Return the shared task identifier."""

        return self.before.task_id

    @property
    def category(self) -> str:
        """Return the shared task category."""

        return self.before.category

    @property
    def reward_delta(self) -> float:
        """Return after reward minus before reward."""

        return self.after.reward - self.before.reward

    @property
    def answer_correct_delta(self) -> int:
        """Return the integer change in answer correctness."""

        return (
            int(self.after.answer_correct)
            - int(self.before.answer_correct)
        )

    @property
    def generation_count_delta(self) -> int:
        """Return the generation-count change."""

        return (
            self.after.generation_count
            - self.before.generation_count
        )

    @property
    def generated_token_count_delta(self) -> int:
        """Return the generated-token-count change."""

        return (
            self.after.generated_token_count
            - self.before.generated_token_count
        )

    def validate(self) -> None:
        """Validate a paired task comparison."""

        self.before.validate()
        self.after.validate()

        if self.before.task_id != self.after.task_id:
            raise ValueError(
                "before and after task_id must match"
            )

        if self.before.category != self.after.category:
            raise ValueError(
                "before and after category must match"
            )


def _aggregates_match(
    left: ClosedLoopAggregate,
    right: ClosedLoopAggregate,
) -> bool:
    """Return whether two aggregates contain equal metrics."""

    return (
        left.task_count == right.task_count
        and left.completed_count
        == right.completed_count
        and left.correct_answer_count
        == right.correct_answer_count
        and math.isclose(
            left.total_reward,
            right.total_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            left.mean_reward,
            right.mean_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and left.generation_count
        == right.generation_count
        and left.generated_token_count
        == right.generated_token_count
    )


@dataclass(frozen=True)
class ClosedLoopSummary:
    """Result of rollout, training, and rerollout."""

    source_artifact: str
    model_name: str
    normalization: PolicyNormalization
    optimizer_steps: int
    training_steps: tuple[TrainingComparisonStep, ...]
    before: ClosedLoopAggregate
    after: ClosedLoopAggregate
    tasks: tuple[ClosedLoopTaskComparison, ...]

    @property
    def reward_delta(self) -> float:
        """Return total reward change."""

        return (
            self.after.total_reward
            - self.before.total_reward
        )

    @property
    def mean_reward_delta(self) -> float:
        """Return mean reward change."""

        return (
            self.after.mean_reward
            - self.before.mean_reward
        )

    @property
    def completed_count_delta(self) -> int:
        """Return completed-task-count change."""

        return (
            self.after.completed_count
            - self.before.completed_count
        )

    @property
    def correct_answer_count_delta(self) -> int:
        """Return correct-answer-count change."""

        return (
            self.after.correct_answer_count
            - self.before.correct_answer_count
        )

    def validate(self) -> None:
        """Validate a closed-loop summary."""

        if not self.source_artifact:
            raise ValueError(
                "source_artifact must not be empty"
            )

        if not self.model_name:
            raise ValueError(
                "model_name must not be empty"
            )

        if self.normalization not in {
            "token",
            "sequence",
            "trajectory",
        }:
            raise ValueError(
                "normalization is not supported"
            )

        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps <= 0
        ):
            raise ValueError(
                "optimizer_steps must be a positive integer"
            )

        if (
            len(self.training_steps)
            != self.optimizer_steps
        ):
            raise ValueError(
                "training_steps must match optimizer_steps"
            )

        expected_step_indices = tuple(
            range(1, self.optimizer_steps + 1)
        )
        actual_step_indices = tuple(
            step.step_index
            for step in self.training_steps
        )

        if actual_step_indices != expected_step_indices:
            raise ValueError(
                "training step indices must be consecutive "
                "and start at one"
            )

        for step in self.training_steps:
            step.validate()

        if not self.tasks:
            raise ValueError(
                "tasks must not be empty"
            )

        task_ids: set[str] = set()

        for task in self.tasks:
            task.validate()

            if task.task_id in task_ids:
                raise ValueError(
                    "task IDs must be unique"
                )

            task_ids.add(task.task_id)

        self.before.validate()
        self.after.validate()

        expected_before = build_closed_loop_aggregate(
            tuple(
                task.before
                for task in self.tasks
            )
        )
        expected_after = build_closed_loop_aggregate(
            tuple(
                task.after
                for task in self.tasks
            )
        )

        if not _aggregates_match(
            self.before,
            expected_before,
        ):
            raise ValueError(
                "before aggregate must match task snapshots"
            )

        if not _aggregates_match(
            self.after,
            expected_after,
        ):
            raise ValueError(
                "after aggregate must match task snapshots"
            )


def _snapshot_to_dict(
    snapshot: ClosedLoopTaskSnapshot,
) -> dict[str, Any]:
    """Serialize one task snapshot."""

    return {
        "task_id": snapshot.task_id,
        "category": snapshot.category,
        "status": snapshot.status,
        "reward": snapshot.reward,
        "answer_correct": snapshot.answer_correct,
        "generation_count": snapshot.generation_count,
        "generated_token_count": (
            snapshot.generated_token_count
        ),
    }


def _aggregate_to_dict(
    aggregate: ClosedLoopAggregate,
) -> dict[str, Any]:
    """Serialize one aggregate."""

    return {
        "task_count": aggregate.task_count,
        "completed_count": aggregate.completed_count,
        "correct_answer_count": (
            aggregate.correct_answer_count
        ),
        "total_reward": aggregate.total_reward,
        "mean_reward": aggregate.mean_reward,
        "completion_rate": aggregate.completion_rate,
        "answer_accuracy": aggregate.answer_accuracy,
        "generation_count": aggregate.generation_count,
        "generated_token_count": (
            aggregate.generated_token_count
        ),
    }


def closed_loop_summary_to_dict(
    summary: ClosedLoopSummary,
) -> dict[str, Any]:
    """Serialize a validated closed-loop summary."""

    summary.validate()

    return {
        "schema_version": 1,
        "source_artifact": summary.source_artifact,
        "model_name": summary.model_name,
        "training": {
            "normalization": summary.normalization,
            "optimizer_steps": summary.optimizer_steps,
            "steps": [
                {
                    "step_index": step.step_index,
                    "loss": step.loss,
                    "gradient_norm": step.gradient_norm,
                    "mean_ratio": step.mean_ratio,
                    "approximate_kl": (
                        step.approximate_kl
                    ),
                    "clip_fraction": step.clip_fraction,
                    "trainable_token_count": (
                        step.trainable_token_count
                    ),
                    "parameter_delta": (
                        step.parameter_delta
                    ),
                }
                for step in summary.training_steps
            ],
        },
        "before": _aggregate_to_dict(summary.before),
        "after": _aggregate_to_dict(summary.after),
        "deltas": {
            "total_reward": summary.reward_delta,
            "mean_reward": summary.mean_reward_delta,
            "completed_count": (
                summary.completed_count_delta
            ),
            "correct_answer_count": (
                summary.correct_answer_count_delta
            ),
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "category": task.category,
                "before": _snapshot_to_dict(task.before),
                "after": _snapshot_to_dict(task.after),
                "deltas": {
                    "reward": task.reward_delta,
                    "answer_correct": (
                        task.answer_correct_delta
                    ),
                    "generation_count": (
                        task.generation_count_delta
                    ),
                    "generated_token_count": (
                        task.generated_token_count_delta
                    ),
                },
            }
            for task in summary.tasks
        ],
    }
