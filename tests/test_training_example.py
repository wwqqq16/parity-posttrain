"""Tests for trajectory training examples."""

from __future__ import annotations

import math

import pytest

from parity_posttrain.training.example import (
    IGNORE_INDEX,
    build_generated_token_labels,
    build_generated_token_loss_mask,
    build_trajectory_training_example,
)


def test_build_generated_token_loss_mask() -> None:
    mask = build_generated_token_loss_mask(
        prompt_token_count=3,
        generated_token_count=2,
    )

    assert mask == (0, 0, 0, 1, 1)


def test_build_generated_token_labels() -> None:
    labels = build_generated_token_labels(
        prompt_token_ids=[10, 11, 12],
        generated_token_ids=[20, 21],
    )

    assert labels == (
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        21,
    )


def test_build_trajectory_training_example() -> None:
    example = build_trajectory_training_example(
        task_id="calculator_001",
        turn_index=1,
        status="completed",
        model_name="test-model",
        reward=1.0,
        prompt_token_ids=[10, 11, 12],
        generated_token_ids=[20, 21],
        rollout_logprobs=[-0.1, -0.2],
    )

    assert example.task_id == "calculator_001"
    assert example.turn_index == 1
    assert example.reward == 1.0
    assert example.input_ids == (10, 11, 12, 20, 21)
    assert example.labels == (
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        21,
    )
    assert example.loss_mask == (0, 0, 0, 1, 1)
    assert example.rollout_logprobs == (-0.1, -0.2)


def test_example_rejects_logprob_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must have equal length",
    ):
        build_trajectory_training_example(
            task_id="calculator_001",
            turn_index=0,
            status="completed",
            model_name="test-model",
            reward=1.0,
            prompt_token_ids=[10, 11],
            generated_token_ids=[20, 21],
            rollout_logprobs=[-0.1],
        )


def test_example_rejects_empty_generation() -> None:
    with pytest.raises(
        ValueError,
        match="generated_token_ids must not be empty",
    ):
        build_trajectory_training_example(
            task_id="calculator_001",
            turn_index=0,
            status="completed",
            model_name="test-model",
            reward=1.0,
            prompt_token_ids=[10, 11],
            generated_token_ids=[],
            rollout_logprobs=[],
        )


def test_example_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="reward must be finite",
    ):
        build_trajectory_training_example(
            task_id="calculator_001",
            turn_index=0,
            status="completed",
            model_name="test-model",
            reward=math.inf,
            prompt_token_ids=[10, 11],
            generated_token_ids=[20],
            rollout_logprobs=[-0.1],
        )

    with pytest.raises(
        ValueError,
        match="rollout_logprobs must contain finite",
    ):
        build_trajectory_training_example(
            task_id="calculator_001",
            turn_index=0,
            status="completed",
            model_name="test-model",
            reward=1.0,
            prompt_token_ids=[10, 11],
            generated_token_ids=[20],
            rollout_logprobs=[math.nan],
        )
