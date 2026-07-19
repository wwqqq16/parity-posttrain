"""Tests for training comparison result models."""

from __future__ import annotations

from dataclasses import replace

import pytest

from parity_posttrain.training.comparison import (
    TaskLogprobShift,
    TrainingComparisonRow,
    TrainingComparisonStep,
    TrainingComparisonSummary,
    TrainingComparisonTask,
    training_comparison_to_dict,
)


def make_tasks() -> tuple[TrainingComparisonTask, ...]:
    """Create representative trajectory metadata."""

    return (
        TrainingComparisonTask(
            task_id="catalog_004",
            reward=1.0,
            turn_count=2,
            generated_token_count=42,
        ),
        TrainingComparisonTask(
            task_id="shopping_004",
            reward=0.0,
            turn_count=1,
            generated_token_count=47,
        ),
    )


def make_row(
    normalization: str = "token",
) -> TrainingComparisonRow:
    """Create one representative comparison row."""

    return TrainingComparisonRow(
        normalization=normalization,  # type: ignore[arg-type]
        loss=0.2,
        gradient_norm=0.004,
        mean_ratio=1.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        trainable_token_count=89,
        parameter_delta=5e-5,
        mean_absolute_logprob_shift=2e-6,
        max_absolute_logprob_shift=1e-4,
        task_shifts=(
            TaskLogprobShift(
                task_id="catalog_004",
                mean_logprob_shift=2e-7,
                mean_absolute_logprob_shift=1e-6,
                max_absolute_logprob_shift=5e-5,
            ),
            TaskLogprobShift(
                task_id="shopping_004",
                mean_logprob_shift=-2e-6,
                mean_absolute_logprob_shift=3e-6,
                max_absolute_logprob_shift=1e-4,
            ),
        ),
    )


def make_summary() -> TrainingComparisonSummary:
    """Create a representative comparison summary."""

    return TrainingComparisonSummary(
        source_artifact=(
            "artifacts/agent_benchmark.json"
        ),
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        dtype="torch.float32",
        learning_rate=0.05,
        clip_epsilon=0.2,
        max_gradient_norm=1.0,
        trainable_parameter_names=(
            "model.norm.weight",
        ),
        trainable_parameter_count=896,
        tasks=make_tasks(),
        rows=(
            make_row("token"),
            make_row("sequence"),
            make_row("trajectory"),
        ),
    )


def test_training_comparison_serializes() -> None:
    payload = training_comparison_to_dict(
        make_summary()
    )

    assert payload["schema_version"] == 2

    source = payload["source"]
    training = payload["training"]
    results = payload["results"]

    assert isinstance(source, dict)
    assert isinstance(training, dict)
    assert isinstance(results, list)

    assert source["device"] == "cpu"
    assert source["dtype"] == "torch.float32"
    assert training["trainable_parameter_count"] == 896
    assert training["steps"] == 1
    assert len(results) == 3
    assert results[0]["steps"][0]["step_index"] == 1
    assert results[0]["normalization"] == "token"

    task_shifts = results[0]["task_shifts"]

    assert isinstance(task_shifts, list)
    assert task_shifts[0]["task_id"] == "catalog_004"


def test_summary_rejects_duplicate_normalization() -> None:
    summary = make_summary()
    duplicate = replace(
        summary,
        rows=(
            make_row("token"),
            make_row("token"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="unique normalizations",
    ):
        duplicate.validate()


def test_summary_rejects_mismatched_task_shifts() -> None:
    summary = make_summary()
    row = make_row("token")
    reversed_row = replace(
        row,
        task_shifts=tuple(
            reversed(row.task_shifts)
        ),
    )
    invalid = replace(
        summary,
        rows=(reversed_row,),
    )

    with pytest.raises(
        ValueError,
        match="match selected tasks and order",
    ):
        invalid.validate()


def test_summary_rejects_token_count_mismatch() -> None:
    summary = make_summary()
    invalid_row = replace(
        make_row("token"),
        trainable_token_count=90,
    )
    invalid = replace(
        summary,
        rows=(invalid_row,),
    )

    with pytest.raises(
        ValueError,
        match="does not match selected tasks",
    ):
        invalid.validate()


def test_multi_step_row_serializes() -> None:
    first_step = TrainingComparisonStep(
        step_index=1,
        loss=0.3,
        gradient_norm=0.005,
        mean_ratio=1.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        trainable_token_count=89,
        parameter_delta=2e-5,
    )
    final_step = TrainingComparisonStep(
        step_index=2,
        loss=0.2,
        gradient_norm=0.004,
        mean_ratio=1.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        trainable_token_count=89,
        parameter_delta=5e-5,
    )
    row = replace(
        make_row(),
        steps=(first_step, final_step),
    )
    summary = replace(
        make_summary(),
        rows=(row,),
    )

    payload = training_comparison_to_dict(summary)
    training = payload["training"]
    results = payload["results"]

    assert isinstance(training, dict)
    assert isinstance(results, list)
    assert training["steps"] == 2
    assert len(results[0]["steps"]) == 2
    assert (
        results[0]["steps"][1]["parameter_delta"]
        == 5e-5
    )


def test_row_rejects_mismatched_final_step() -> None:
    mismatched_step = TrainingComparisonStep(
        step_index=1,
        loss=0.3,
        gradient_norm=0.004,
        mean_ratio=1.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        trainable_token_count=89,
        parameter_delta=5e-5,
    )
    invalid = replace(
        make_row(),
        steps=(mismatched_step,),
    )

    with pytest.raises(
        ValueError,
        match="final step loss must match row",
    ):
        invalid.validate()
