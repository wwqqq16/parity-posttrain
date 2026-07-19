"""Tests for a single trajectory policy update."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from parity_posttrain.training import (
    build_trajectory_training_example,
    collate_training_examples,
    run_clipped_policy_step,
)


class TrainableLogitModel(torch.nn.Module):
    """Return a trainable fixed logit tensor."""

    def __init__(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.zeros(
                (
                    batch_size,
                    sequence_length,
                    vocabulary_size,
                ),
                dtype=torch.float32,
            )
        )
        self.training_mode_seen = False

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Any:
        """Return trainable logits."""

        if self.logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "logits and input_ids shapes do not match"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape does not match"
            )

        if use_cache:
            raise ValueError(
                "training must disable the KV cache"
            )

        if not return_dict:
            raise ValueError(
                "training requires dictionary output"
            )

        self.training_mode_seen = self.training

        return SimpleNamespace(logits=self.logits)


def make_batch():
    """Create positive and zero-reward examples."""

    initial_logprob = -math.log(4.0)

    positive = build_trajectory_training_example(
        task_id="positive",
        turn_index=0,
        status="completed",
        model_name="test-model",
        reward=1.0,
        prompt_token_ids=[1],
        generated_token_ids=[2],
        rollout_logprobs=[initial_logprob],
    )
    negative = build_trajectory_training_example(
        task_id="negative",
        turn_index=0,
        status="protocol_error",
        model_name="test-model",
        reward=0.0,
        prompt_token_ids=[1],
        generated_token_ids=[3],
        rollout_logprobs=[initial_logprob],
    )

    return collate_training_examples(
        (positive, negative),
        pad_token_id=0,
    )


def test_training_step_updates_parameters() -> None:
    batch = make_batch()
    model = TrainableLogitModel(
        batch_size=2,
        sequence_length=2,
        vocabulary_size=4,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    before = model.logits.detach().clone()

    result = run_clipped_policy_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        clip_epsilon=0.2,
        max_gradient_norm=1.0,
    )

    assert result.trainable_token_count == 2
    assert result.mean_ratio == pytest.approx(1.0)
    assert result.approximate_kl == pytest.approx(0.0)
    assert result.clip_fraction == 0.0
    assert result.gradient_norm > 0.0
    assert result.normalization == "token"
    assert model.training_mode_seen

    assert not torch.equal(
        before,
        model.logits.detach(),
    )
    assert model.logits.grad is not None


def test_training_step_rejects_invalid_gradient_norm() -> None:
    batch = make_batch()
    model = TrainableLogitModel(
        batch_size=2,
        sequence_length=2,
        vocabulary_size=4,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    with pytest.raises(
        ValueError,
        match="max_gradient_norm must be finite",
    ):
        run_clipped_policy_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            max_gradient_norm=0.0,
        )


def test_training_step_supports_sequence_normalization() -> None:
    batch = make_batch()
    model = TrainableLogitModel(
        batch_size=2,
        sequence_length=2,
        vocabulary_size=4,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    result = run_clipped_policy_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        normalization="sequence",
    )

    assert result.normalization == "sequence"
    assert result.trainable_token_count == 2
    assert result.gradient_norm > 0.0
