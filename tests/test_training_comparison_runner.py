"""Tests for controlled training-comparison execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from parity_posttrain.training import (
    TrajectoryTrainingBatch,
    TrajectoryTrainingExample,
    build_trajectory_training_example,
    collate_training_examples,
    run_training_comparison,
    select_training_examples,
)


class TinyCausalModel(torch.nn.Module):
    """Small differentiable model for comparison tests."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_bias = torch.nn.Parameter(
            torch.tensor(
                [0.0, 0.2, -0.1, 0.1],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Return position-independent causal logits."""

        del attention_mask
        del use_cache
        del return_dict

        logits = self.logit_bias.view(
            1,
            1,
            -1,
        ).expand(
            input_ids.shape[0],
            input_ids.shape[1],
            -1,
        )

        return SimpleNamespace(logits=logits)


def make_examples() -> tuple[
    TrajectoryTrainingExample,
    ...,
]:
    """Create one two-turn and one one-turn trajectory."""

    return (
        build_trajectory_training_example(
            task_id="positive",
            turn_index=1,
            status="completed",
            model_name="tiny-model",
            reward=1.0,
            prompt_token_ids=[1],
            generated_token_ids=[2],
            rollout_logprobs=[0.0],
        ),
        build_trajectory_training_example(
            task_id="negative",
            turn_index=0,
            status="protocol_error",
            model_name="tiny-model",
            reward=0.0,
            prompt_token_ids=[1],
            generated_token_ids=[3, 3, 3],
            rollout_logprobs=[
                0.0,
                0.0,
                0.0,
            ],
        ),
        build_trajectory_training_example(
            task_id="positive",
            turn_index=0,
            status="completed",
            model_name="tiny-model",
            reward=1.0,
            prompt_token_ids=[1],
            generated_token_ids=[2],
            rollout_logprobs=[0.0],
        ),
    )


def make_batch() -> TrajectoryTrainingBatch:
    """Create a selected and collated comparison batch."""

    selected = select_training_examples(
        make_examples(),
        ("positive", "negative"),
    )

    return collate_training_examples(
        selected,
        pad_token_id=0,
    )


def test_selects_all_turns_in_requested_order() -> None:
    selected = select_training_examples(
        make_examples(),
        ("negative", "positive"),
    )

    assert [
        (
            example.task_id,
            example.turn_index,
        )
        for example in selected
    ] == [
        ("negative", 0),
        ("positive", 0),
        ("positive", 1),
    ]


def test_runs_all_normalizations_and_restores_model() -> None:
    model = TinyCausalModel()
    model.eval()
    before = model.logit_bias.detach().clone()

    summary = run_training_comparison(
        model=model,
        batch=make_batch(),
        source_artifact="artifact.json",
        model_name="tiny-model",
        trainable_parameter_names=("logit_bias",),
        learning_rate=0.1,
        max_gradient_norm=10.0,
    )

    assert summary.device == "cpu"
    assert summary.dtype == "torch.float32"
    assert summary.trainable_parameter_names == (
        "logit_bias",
    )
    assert summary.trainable_parameter_count == 4

    assert [
        task.task_id
        for task in summary.tasks
    ] == [
        "positive",
        "negative",
    ]
    assert summary.tasks[0].turn_count == 2
    assert (
        summary.tasks[0].generated_token_count
        == 2
    )
    assert summary.tasks[1].turn_count == 1
    assert (
        summary.tasks[1].generated_token_count
        == 3
    )

    assert [
        row.normalization
        for row in summary.rows
    ] == [
        "token",
        "sequence",
        "trajectory",
    ]

    for row in summary.rows:
        assert len(row.steps) == 1
        assert row.steps[0].step_index == 1
        assert row.trainable_token_count == 5
        assert row.mean_ratio == pytest.approx(1.0)
        assert row.approximate_kl == pytest.approx(
            0.0
        )
        assert row.gradient_norm > 0.0
        assert row.parameter_delta > 0.0
        assert (
            row.max_absolute_logprob_shift
            > 0.0
        )
        assert [
            shift.task_id
            for shift in row.task_shifts
        ] == [
            "positive",
            "negative",
        ]

    assert torch.equal(
        model.logit_bias.detach(),
        before,
    )
    assert model.training is False


def test_runs_multiple_steps_and_records_history() -> None:
    model = TinyCausalModel()
    model.eval()
    before = model.logit_bias.detach().clone()

    summary = run_training_comparison(
        model=model,
        batch=make_batch(),
        source_artifact="artifact.json",
        model_name="tiny-model",
        trainable_parameter_names=("logit_bias",),
        normalizations=("token",),
        steps=3,
        learning_rate=0.1,
        max_gradient_norm=10.0,
    )

    assert len(summary.rows) == 1
    row = summary.rows[0]

    assert [
        step.step_index
        for step in row.steps
    ] == [1, 2, 3]

    assert row.steps[0].mean_ratio == pytest.approx(
        1.0
    )
    assert (
        row.steps[0].approximate_kl
        == pytest.approx(0.0)
    )
    assert row.steps[1].approximate_kl > 0.0

    assert row.loss == row.steps[-1].loss
    assert (
        row.gradient_norm
        == row.steps[-1].gradient_norm
    )
    assert row.mean_ratio == row.steps[-1].mean_ratio
    assert (
        row.approximate_kl
        == row.steps[-1].approximate_kl
    )
    assert (
        row.clip_fraction
        == row.steps[-1].clip_fraction
    )
    assert (
        row.parameter_delta
        == row.steps[-1].parameter_delta
    )
    assert (
        row.steps[-1].parameter_delta
        > row.steps[0].parameter_delta
    )

    assert torch.equal(
        model.logit_bias.detach(),
        before,
    )
    assert model.training is False


@pytest.mark.parametrize(
    "steps",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_rejects_invalid_step_count(
    steps: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="steps must be a positive integer",
    ):
        run_training_comparison(
            model=TinyCausalModel(),
            batch=make_batch(),
            source_artifact="artifact.json",
            model_name="tiny-model",
            trainable_parameter_names=(
                "logit_bias",
            ),
            steps=steps,  # type: ignore[arg-type]
        )


def test_rejects_duplicate_normalizations() -> None:
    with pytest.raises(
        ValueError,
        match="normalizations must be unique",
    ):
        run_training_comparison(
            model=TinyCausalModel(),
            batch=make_batch(),
            source_artifact="artifact.json",
            model_name="tiny-model",
            trainable_parameter_names=(
                "logit_bias",
            ),
            normalizations=(
                "token",
                "token",
            ),
        )


def test_rejects_unknown_trainable_parameter() -> None:
    with pytest.raises(
        ValueError,
        match="unknown trainable parameters",
    ):
        run_training_comparison(
            model=TinyCausalModel(),
            batch=make_batch(),
            source_artifact="artifact.json",
            model_name="tiny-model",
            trainable_parameter_names=(
                "missing.weight",
            ),
        )
