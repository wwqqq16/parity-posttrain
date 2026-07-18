"""Tests for trajectory policy objectives."""

from __future__ import annotations

import math

import pytest
import torch

from parity_posttrain.training import (
    build_trajectory_training_example,
    centered_reward_advantages,
    clipped_policy_loss,
    collate_training_examples,
)


def make_batch():
    """Create a two-example reward batch."""

    positive = build_trajectory_training_example(
        task_id="positive",
        turn_index=0,
        status="completed",
        model_name="test-model",
        reward=1.0,
        prompt_token_ids=[1],
        generated_token_ids=[2],
        rollout_logprobs=[0.0],
    )
    negative = build_trajectory_training_example(
        task_id="negative",
        turn_index=0,
        status="protocol_error",
        model_name="test-model",
        reward=0.0,
        prompt_token_ids=[1],
        generated_token_ids=[3],
        rollout_logprobs=[0.0],
    )

    return collate_training_examples(
        (positive, negative),
        pad_token_id=0,
    )


def test_centered_reward_advantages() -> None:
    rewards = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    advantages = centered_reward_advantages(
        rewards
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([0.5, -0.5]),
    )


def test_clipped_policy_loss_is_differentiable() -> None:
    batch = make_batch()
    trainer_logprobs = torch.zeros_like(
        batch.rollout_logprobs
    )
    trainer_logprobs[0, 1] = math.log(1.1)
    trainer_logprobs[1, 1] = math.log(0.9)
    trainer_logprobs.requires_grad_()

    result = clipped_policy_loss(
        trainer_logprobs=trainer_logprobs,
        batch=batch,
        clip_epsilon=0.2,
    )

    assert result.loss.item() == pytest.approx(
        -0.05
    )
    assert result.mean_ratio == pytest.approx(1.0)
    assert result.clip_fraction == 0.0
    assert result.trainable_token_count == 2
    assert result.approximate_kl >= 0.0

    result.loss.backward()

    assert trainer_logprobs.grad is not None
    assert trainer_logprobs.grad[0, 1].item() < 0
    assert trainer_logprobs.grad[1, 1].item() > 0


def test_clipped_policy_loss_clips_ratios() -> None:
    batch = make_batch()
    trainer_logprobs = torch.zeros_like(
        batch.rollout_logprobs
    )
    trainer_logprobs[0, 1] = math.log(1.5)
    trainer_logprobs[1, 1] = math.log(0.5)

    result = clipped_policy_loss(
        trainer_logprobs=trainer_logprobs,
        batch=batch,
        clip_epsilon=0.2,
    )

    assert result.loss.item() == pytest.approx(
        -0.1
    )
    assert result.clip_fraction == 1.0
    assert result.trainable_token_count == 2


def test_clipped_policy_loss_rejects_shape() -> None:
    batch = make_batch()

    with pytest.raises(
        ValueError,
        match="must match input_ids shape",
    ):
        clipped_policy_loss(
            trainer_logprobs=torch.zeros((1, 2)),
            batch=batch,
        )
