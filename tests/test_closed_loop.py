"""Tests for closed-loop experiment result models."""

from __future__ import annotations

from dataclasses import replace

import pytest

from parity_posttrain.training import (
    ClosedLoopSummary,
    ClosedLoopTaskComparison,
    ClosedLoopTaskSnapshot,
    TrainingComparisonStep,
    build_closed_loop_aggregate,
    closed_loop_summary_to_dict,
    fingerprint_generated_token_ids,
)


def make_snapshot(
    *,
    task_id: str,
    category: str,
    status: str,
    reward: float,
    answer_correct: bool,
    generation_count: int = 1,
    generated_token_count: int = 10,
    token_sequences: tuple[
        tuple[int, ...],
        ...,
    ] | None = None,
) -> ClosedLoopTaskSnapshot:
    """Create one test task snapshot."""

    if token_sequences is None:
        token_sequences = (
            tuple(range(generated_token_count)),
        )

    return ClosedLoopTaskSnapshot(
        task_id=task_id,
        category=category,
        status=status,
        reward=reward,
        answer_correct=answer_correct,
        generation_count=generation_count,
        generated_token_count=generated_token_count,
        trajectory_fingerprint=(
            fingerprint_generated_token_ids(
                token_sequences
            )
        ),
    )


def make_training_step() -> TrainingComparisonStep:
    """Create one test optimizer-step record."""

    return TrainingComparisonStep(
        step_index=1,
        loss=0.2,
        gradient_norm=0.01,
        mean_ratio=1.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        trainable_token_count=20,
        parameter_delta=1e-4,
    )


def make_tasks() -> tuple[
    ClosedLoopTaskComparison,
    ...,
]:
    """Create one unchanged and one improved task."""

    return (
        ClosedLoopTaskComparison(
            before=make_snapshot(
                task_id="catalog",
                category="catalog",
                status="completed",
                reward=1.0,
                answer_correct=True,
            ),
            after=make_snapshot(
                task_id="catalog",
                category="catalog",
                status="completed",
                reward=1.0,
                answer_correct=True,
            ),
        ),
        ClosedLoopTaskComparison(
            before=make_snapshot(
                task_id="shopping",
                category="shopping",
                status="protocol_error",
                reward=0.0,
                answer_correct=False,
            ),
            after=make_snapshot(
                task_id="shopping",
                category="shopping",
                status="completed",
                reward=1.0,
                answer_correct=True,
                generation_count=2,
                generated_token_count=16,
                token_sequences=(
                    tuple(range(8)),
                    tuple(range(8, 16)),
                ),
            ),
        ),
    )


def make_summary() -> ClosedLoopSummary:
    """Create a valid closed-loop summary."""

    tasks = make_tasks()

    return ClosedLoopSummary(
        source_artifact="artifact.json",
        model_name="tiny-model",
        normalization="trajectory",
        optimizer_steps=1,
        training_steps=(make_training_step(),),
        before=build_closed_loop_aggregate(
            tuple(
                task.before
                for task in tasks
            )
        ),
        after=build_closed_loop_aggregate(
            tuple(
                task.after
                for task in tasks
            )
        ),
        tasks=tasks,
    )


def test_builds_closed_loop_aggregate() -> None:
    aggregate = make_summary().after

    assert aggregate.task_count == 2
    assert aggregate.completed_count == 2
    assert aggregate.correct_answer_count == 2
    assert aggregate.total_reward == pytest.approx(2.0)
    assert aggregate.mean_reward == pytest.approx(1.0)
    assert aggregate.completion_rate == pytest.approx(1.0)
    assert aggregate.answer_accuracy == pytest.approx(1.0)
    assert aggregate.generation_count == 3
    assert aggregate.generated_token_count == 26


def test_serializes_closed_loop_summary() -> None:
    payload = closed_loop_summary_to_dict(
        make_summary()
    )

    assert payload["schema_version"] == 3
    assert (
        payload["training"]["steps"][0][
            "active_clip_fraction"
        ]
        == 0.0
    )
    assert payload["model_name"] == "tiny-model"
    assert payload["training"]["optimizer_steps"] == 1
    assert payload["deltas"]["total_reward"] == 1.0
    assert payload["deltas"]["completed_count"] == 1
    assert (
        payload["deltas"]["correct_answer_count"]
        == 1
    )

    tasks = payload["tasks"]

    assert len(tasks) == 2
    assert tasks[1]["task_id"] == "shopping"
    assert tasks[1]["deltas"]["reward"] == 1.0
    assert tasks[1]["deltas"]["answer_correct"] == 1
    assert (
        tasks[1]["deltas"]["generated_token_count"]
        == 6
    )
    assert (
        tasks[1]["deltas"]["trajectory_changed"]
        is True
    )
    assert len(
        tasks[1]["before"]["trajectory_fingerprint"]
    ) == 64


def test_rejects_mismatched_task_identifiers() -> None:
    task = make_tasks()[0]
    invalid = replace(
        task,
        after=replace(
            task.after,
            task_id="different",
        ),
    )

    with pytest.raises(
        ValueError,
        match="task_id must match",
    ):
        invalid.validate()


def test_rejects_duplicate_task_ids() -> None:
    task = make_tasks()[0]
    tasks = (task, task)

    invalid = replace(
        make_summary(),
        before=build_closed_loop_aggregate(
            tuple(
                item.before
                for item in tasks
            )
        ),
        after=build_closed_loop_aggregate(
            tuple(
                item.after
                for item in tasks
            )
        ),
        tasks=tasks,
    )

    with pytest.raises(
        ValueError,
        match="task IDs must be unique",
    ):
        invalid.validate()


def test_rejects_training_step_count_mismatch() -> None:
    invalid = replace(
        make_summary(),
        optimizer_steps=2,
    )

    with pytest.raises(
        ValueError,
        match="training_steps must match",
    ):
        invalid.validate()
