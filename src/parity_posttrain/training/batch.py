"""Collate variable-length trajectory training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from parity_posttrain.training.example import (
    IGNORE_INDEX,
    TrajectoryTrainingExample,
)


@dataclass(frozen=True)
class TrajectoryTrainingBatch:
    """Right-padded tensors for trajectory post-training."""

    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    labels: torch.LongTensor
    loss_mask: torch.BoolTensor
    rollout_logprobs: torch.FloatTensor
    rewards: torch.FloatTensor
    sequence_lengths: torch.LongTensor
    task_ids: tuple[str, ...]
    turn_indices: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        """Return the number of examples."""

        return self.input_ids.shape[0]

    @property
    def max_sequence_length(self) -> int:
        """Return the padded sequence length."""

        return self.input_ids.shape[1]

    @property
    def trainable_token_count(self) -> int:
        """Return the number of generated tokens."""

        return int(self.loss_mask.sum().item())

    def validate(self) -> None:
        """Validate batch tensor shapes and masks."""

        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must be rank two")

        expected_shape = self.input_ids.shape

        for name, tensor in (
            ("attention_mask", self.attention_mask),
            ("labels", self.labels),
            ("loss_mask", self.loss_mask),
            ("rollout_logprobs", self.rollout_logprobs),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must match input_ids shape"
                )

        batch_size = expected_shape[0]

        if self.rewards.shape != (batch_size,):
            raise ValueError(
                "rewards must have shape [batch_size]"
            )

        if self.sequence_lengths.shape != (batch_size,):
            raise ValueError(
                "sequence_lengths must have shape "
                "[batch_size]"
            )

        if len(self.task_ids) != batch_size:
            raise ValueError(
                "task_ids must match batch size"
            )

        if len(self.turn_indices) != batch_size:
            raise ValueError(
                "turn_indices must match batch size"
            )

        if torch.any(
            self.loss_mask & ~self.attention_mask
        ):
            raise ValueError(
                "loss_mask cannot select padding positions"
            )

        if torch.any(
            self.labels[~self.loss_mask] != IGNORE_INDEX
        ):
            raise ValueError(
                "non-trainable labels must use IGNORE_INDEX"
            )


def collate_training_examples(
    examples: tuple[TrajectoryTrainingExample, ...],
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> TrajectoryTrainingBatch:
    """Right-pad trajectory examples into one tensor batch."""

    if not examples:
        raise ValueError("examples must not be empty")

    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or pad_token_id < 0
    ):
        raise ValueError(
            "pad_token_id must be a non-negative integer"
        )

    for example in examples:
        example.validate()

    batch_size = len(examples)
    max_length = max(
        len(example.input_ids)
        for example in examples
    )

    input_ids = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
        device=device,
    )
    labels = torch.full(
        (batch_size, max_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    loss_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
        device=device,
    )
    rollout_logprobs = torch.zeros(
        (batch_size, max_length),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.empty(
        batch_size,
        dtype=torch.float32,
        device=device,
    )
    sequence_lengths = torch.empty(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    task_ids: list[str] = []
    turn_indices: list[int] = []

    for row, example in enumerate(examples):
        sequence_length = len(example.input_ids)
        prompt_length = len(example.prompt_token_ids)
        generated_length = len(
            example.generated_token_ids
        )
        generated_slice = slice(
            prompt_length,
            prompt_length + generated_length,
        )

        input_ids[row, :sequence_length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :sequence_length] = True
        labels[row, :sequence_length] = torch.tensor(
            example.labels,
            dtype=torch.long,
            device=device,
        )
        loss_mask[row, generated_slice] = True
        rollout_logprobs[
            row,
            generated_slice,
        ] = torch.tensor(
            example.rollout_logprobs,
            dtype=torch.float32,
            device=device,
        )

        rewards[row] = example.reward
        sequence_lengths[row] = sequence_length
        task_ids.append(example.task_id)
        turn_indices.append(example.turn_index)

    batch = TrajectoryTrainingBatch(
        input_ids=cast(torch.LongTensor, input_ids),
        attention_mask=cast(
            torch.BoolTensor,
            attention_mask,
        ),
        labels=cast(torch.LongTensor, labels),
        loss_mask=cast(torch.BoolTensor, loss_mask),
        rollout_logprobs=cast(
            torch.FloatTensor,
            rollout_logprobs,
        ),
        rewards=cast(torch.FloatTensor, rewards),
        sequence_lengths=cast(
            torch.LongTensor,
            sequence_lengths,
        ),
        task_ids=tuple(task_ids),
        turn_indices=tuple(turn_indices),
    )
    batch.validate()

    return batch
