"""Tests for repeated trajectory policy updates."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from parity_posttrain.training import (
    build_trajectory_training_example,
    collate_training_examples,
    run_clipped_policy_training,
)


class TrainableLogitModel(torch.nn.Module):
    """Return a trainable fixed logit tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.zeros(
                (2, 2, 4),
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
    ) -> Any:
        """Return trainable causal logits."""

        if input_ids.shape != self.logits.shape[:2]:
            raise ValueError(
                "input shape does not match logits"
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention mask shape does not match"
            )

        if use_cache:
            raise ValueError(
                "training must disable the KV cache"
            )

        if not return_dict:
            raise ValueError(
                "training requires dictionary output"
            )

        return SimpleNamespace(logits=self.logits)


def make_batch():
    """Create one positive and one negative example."""

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


def test_training_loop_runs_multiple_steps() -> None:
    model = TrainableLogitModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    before = model.logits.detach().clone()

    result = run_clipped_policy_training(
        model=model,
        optimizer=optimizer,
        batch=make_batch(),
        steps=3,
        normalization="token",
    )

    assert result.requested_steps == 3
    assert result.completed_steps == 3
    assert len(result.steps) == 3
    assert result.final_step is result.steps[-1]

    assert result.steps[0].mean_ratio == pytest.approx(
        1.0
    )
    assert result.steps[0].approximate_kl == pytest.approx(
        0.0
    )
    assert result.steps[-1].approximate_kl > 0.0

    assert all(
        step.normalization == "token"
        for step in result.steps
    )
    assert all(
        step.trainable_token_count == 2
        for step in result.steps
    )
    assert not torch.equal(
        before,
        model.logits.detach(),
    )


def test_training_loop_supports_trajectory_normalization() -> None:
    model = TrainableLogitModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    result = run_clipped_policy_training(
        model=model,
        optimizer=optimizer,
        batch=make_batch(),
        steps=2,
        normalization="trajectory",
    )

    assert result.completed_steps == 2
    assert result.normalization == "trajectory"
    assert all(
        step.normalization == "trajectory"
        for step in result.steps
    )


@pytest.mark.parametrize(
    "steps",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_training_loop_rejects_invalid_steps(
    steps: object,
) -> None:
    model = TrainableLogitModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    with pytest.raises(
        ValueError,
        match="steps must be a positive integer",
    ):
        run_clipped_policy_training(
            model=model,
            optimizer=optimizer,
            batch=make_batch(),
            steps=steps,  # type: ignore[arg-type]
        )
