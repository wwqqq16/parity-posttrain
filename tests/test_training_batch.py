"""Tests for trajectory training batch collation."""

from __future__ import annotations

import pytest
import torch

from parity_posttrain.training.batch import (
    collate_training_examples,
)
from parity_posttrain.training.example import (
    IGNORE_INDEX,
    build_trajectory_training_example,
)


def make_example(
    *,
    task_id: str,
    turn_index: int,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    rollout_logprobs: list[float],
    reward: float,
):
    """Create one training example."""

    return build_trajectory_training_example(
        task_id=task_id,
        turn_index=turn_index,
        status="completed",
        model_name="test-model",
        reward=reward,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        rollout_logprobs=rollout_logprobs,
    )


def test_collate_training_examples_right_pads() -> None:
    first = make_example(
        task_id="first",
        turn_index=0,
        prompt_token_ids=[10, 11],
        generated_token_ids=[20, 21],
        rollout_logprobs=[-0.1, -0.2],
        reward=1.0,
    )
    second = make_example(
        task_id="second",
        turn_index=1,
        prompt_token_ids=[30],
        generated_token_ids=[40],
        rollout_logprobs=[-0.3],
        reward=0.0,
    )

    batch = collate_training_examples(
        (first, second),
        pad_token_id=99,
    )

    assert batch.input_ids.tolist() == [
        [10, 11, 20, 21],
        [30, 40, 99, 99],
    ]
    assert batch.attention_mask.tolist() == [
        [True, True, True, True],
        [True, True, False, False],
    ]
    assert batch.labels.tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 20, 21],
        [IGNORE_INDEX, 40, IGNORE_INDEX, IGNORE_INDEX],
    ]
    assert batch.loss_mask.tolist() == [
        [False, False, True, True],
        [False, True, False, False],
    ]
    torch.testing.assert_close(
        batch.rollout_logprobs,
        torch.tensor(
            [
                [0.0, 0.0, -0.1, -0.2],
                [0.0, -0.3, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert batch.rewards.tolist() == [1.0, 0.0]
    assert batch.sequence_lengths.tolist() == [4, 2]
    assert batch.task_ids == ("first", "second")
    assert batch.turn_indices == (0, 1)
    assert batch.batch_size == 2
    assert batch.max_sequence_length == 4
    assert batch.trainable_token_count == 3


def test_collate_preserves_tensor_dtypes() -> None:
    example = make_example(
        task_id="test",
        turn_index=0,
        prompt_token_ids=[10],
        generated_token_ids=[20],
        rollout_logprobs=[-0.1],
        reward=1.0,
    )

    batch = collate_training_examples(
        (example,),
        pad_token_id=0,
    )

    assert batch.input_ids.dtype == torch.long
    assert batch.labels.dtype == torch.long
    assert batch.attention_mask.dtype == torch.bool
    assert batch.loss_mask.dtype == torch.bool
    assert batch.rollout_logprobs.dtype == torch.float32
    assert batch.rewards.dtype == torch.float32
    assert batch.sequence_lengths.dtype == torch.long


def test_collate_rejects_empty_examples() -> None:
    with pytest.raises(
        ValueError,
        match="examples must not be empty",
    ):
        collate_training_examples(
            (),
            pad_token_id=0,
        )


def test_collate_rejects_invalid_pad_token() -> None:
    example = make_example(
        task_id="test",
        turn_index=0,
        prompt_token_ids=[10],
        generated_token_ids=[20],
        rollout_logprobs=[-0.1],
        reward=1.0,
    )

    with pytest.raises(
        ValueError,
        match="pad_token_id must be a non-negative",
    ):
        collate_training_examples(
            (example,),
            pad_token_id=-1,
        )
