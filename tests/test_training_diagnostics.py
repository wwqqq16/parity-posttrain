"""Tests for token-level clipping diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from parity_posttrain.training import (
    TokenClippingDiagnostic,
    build_token_clipping_diagnostics,
)
from parity_posttrain.training.batch import (
    collate_training_examples,
)
from parity_posttrain.training.example import (
    build_trajectory_training_example,
)


def make_batch():
    """Create a two-trajectory diagnostic batch."""

    positive = build_trajectory_training_example(
        task_id="positive",
        turn_index=3,
        status="completed",
        model_name="test-model",
        reward=1.0,
        prompt_token_ids=[10],
        generated_token_ids=[20, 21],
        rollout_logprobs=[0.0, 0.0],
    )
    negative = build_trajectory_training_example(
        task_id="negative",
        turn_index=7,
        status="completed",
        model_name="test-model",
        reward=0.0,
        prompt_token_ids=[30],
        generated_token_ids=[40, 41],
        rollout_logprobs=[0.0, 0.0],
    )

    return collate_training_examples(
        (positive, negative),
        pad_token_id=0,
    )


def test_builds_active_and_inactive_diagnostics() -> None:
    batch = make_batch()
    trainer_logprobs = torch.zeros_like(
        batch.rollout_logprobs
    )

    trainer_logprobs[0, 1] = math.log(1.5)
    trainer_logprobs[0, 2] = math.log(0.5)
    trainer_logprobs[1, 1] = math.log(0.5)
    trainer_logprobs[1, 2] = math.log(1.5)

    diagnostics = build_token_clipping_diagnostics(
        trainer_logprobs=trainer_logprobs,
        batch=batch,
        clip_epsilon=0.2,
        normalization="trajectory",
    )

    assert len(diagnostics) == 4

    positive_high = diagnostics[0]
    assert isinstance(
        positive_high,
        TokenClippingDiagnostic,
    )
    assert positive_high.flat_token_index == 0
    assert positive_high.batch_row == 0
    assert positive_high.task_id == "positive"
    assert positive_high.turn_index == 3
    assert positive_high.generated_token_index == 0
    assert positive_high.sequence_position == 1
    assert positive_high.token_id == 20
    assert positive_high.advantage == pytest.approx(0.5)
    assert positive_high.ratio == pytest.approx(1.5)
    assert positive_high.out_of_range is True
    assert positive_high.active_clipped is True
    assert (
        positive_high.clip_direction
        == "positive_high"
    )

    positive_low = diagnostics[1]
    assert positive_low.generated_token_index == 1
    assert positive_low.token_id == 21
    assert positive_low.ratio == pytest.approx(0.5)
    assert positive_low.out_of_range is True
    assert positive_low.active_clipped is False
    assert positive_low.clip_direction == "none"

    negative_low = diagnostics[2]
    assert negative_low.batch_row == 1
    assert negative_low.task_id == "negative"
    assert negative_low.turn_index == 7
    assert negative_low.generated_token_index == 0
    assert negative_low.sequence_position == 1
    assert negative_low.token_id == 40
    assert negative_low.advantage == pytest.approx(-0.5)
    assert negative_low.ratio == pytest.approx(0.5)
    assert negative_low.out_of_range is True
    assert negative_low.active_clipped is True
    assert (
        negative_low.clip_direction
        == "negative_low"
    )

    negative_high = diagnostics[3]
    assert negative_high.generated_token_index == 1
    assert negative_high.token_id == 41
    assert negative_high.ratio == pytest.approx(1.5)
    assert negative_high.out_of_range is True
    assert negative_high.active_clipped is False
    assert negative_high.clip_direction == "none"


def test_rejects_mismatched_shape() -> None:
    batch = make_batch()

    with pytest.raises(
        ValueError,
        match="must match input_ids shape",
    ):
        build_token_clipping_diagnostics(
            trainer_logprobs=torch.zeros((1, 2)),
            batch=batch,
        )


def test_rejects_invalid_clip_epsilon() -> None:
    batch = make_batch()

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        build_token_clipping_diagnostics(
            trainer_logprobs=torch.zeros_like(
                batch.rollout_logprobs
            ),
            batch=batch,
            clip_epsilon=1.0,
        )
