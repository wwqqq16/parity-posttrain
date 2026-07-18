"""Tests for batched trainer-side logprob computation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from parity_posttrain.training import (
    build_trajectory_training_example,
    collate_training_examples,
)
from parity_posttrain.training.logprobs import (
    gather_masked_causal_logprobs,
    rescore_training_batch,
)


class FixedLogitModel(torch.nn.Module):
    """Return a trainable fixed logit tensor."""

    def __init__(
        self,
        logits: torch.Tensor,
    ) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
        self.last_attention_mask: torch.Tensor | None = None
        self.last_use_cache: bool | None = None
        self.last_return_dict: bool | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Any:
        """Return fixed logits and record model arguments."""

        if self.logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "fixed logits do not match input shape"
            )

        self.last_attention_mask = attention_mask
        self.last_use_cache = use_cache
        self.last_return_dict = return_dict

        return SimpleNamespace(logits=self.logits)


def test_gather_handles_variable_prompt_lengths() -> None:
    logits = torch.zeros((2, 5, 6))

    logits[0, 1, 1] = 2.0
    logits[0, 2, 2] = 3.0
    logits[1, 2, 3] = 4.0

    input_ids = torch.tensor(
        [
            [4, 4, 1, 2, 0],
            [4, 4, 4, 3, 0],
        ],
        dtype=torch.long,
    )
    loss_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, False, True, False],
        ],
        dtype=torch.bool,
    )

    result = gather_masked_causal_logprobs(
        logits=logits,
        input_ids=input_ids,
        loss_mask=loss_mask,
    )

    expected = torch.zeros((2, 5))
    expected[0, 2] = torch.log_softmax(
        logits[0, 1],
        dim=-1,
    )[1]
    expected[0, 3] = torch.log_softmax(
        logits[0, 2],
        dim=-1,
    )[2]
    expected[1, 3] = torch.log_softmax(
        logits[1, 2],
        dim=-1,
    )[3]

    torch.testing.assert_close(result, expected)


def test_gather_rejects_first_token_selection() -> None:
    logits = torch.zeros((1, 2, 4))
    input_ids = torch.tensor([[1, 2]])
    loss_mask = torch.tensor([[True, False]])

    with pytest.raises(
        ValueError,
        match="cannot select the first token",
    ):
        gather_masked_causal_logprobs(
            logits=logits,
            input_ids=input_ids,
            loss_mask=loss_mask,
        )


def test_gather_rejects_empty_selection() -> None:
    logits = torch.zeros((1, 2, 4))
    input_ids = torch.tensor([[1, 2]])
    loss_mask = torch.tensor([[False, False]])

    with pytest.raises(
        ValueError,
        match="must select at least one token",
    ):
        gather_masked_causal_logprobs(
            logits=logits,
            input_ids=input_ids,
            loss_mask=loss_mask,
        )


def test_rescore_training_batch_is_differentiable() -> None:
    example = build_trajectory_training_example(
        task_id="calculator_001",
        turn_index=0,
        status="completed",
        model_name="test-model",
        reward=1.0,
        prompt_token_ids=[4, 4],
        generated_token_ids=[1],
        rollout_logprobs=[-0.1],
    )
    batch = collate_training_examples(
        (example,),
        pad_token_id=0,
    )

    logits = torch.zeros((1, 3, 5))
    logits[0, 1, 1] = 2.0
    model = FixedLogitModel(logits)

    result = rescore_training_batch(
        model=model,
        batch=batch,
    )

    expected = torch.log_softmax(
        model.logits[0, 1].float(),
        dim=-1,
    )[1]

    assert result.shape == batch.input_ids.shape
    assert result[0, 2].item() == pytest.approx(
        expected.item()
    )
    assert result[0, 0].item() == 0.0
    assert result[0, 1].item() == 0.0

    assert model.last_use_cache is False
    assert model.last_return_dict is True
    assert model.last_attention_mask is not None
    torch.testing.assert_close(
        model.last_attention_mask,
        batch.attention_mask,
    )

    result.sum().backward()

    assert model.logits.grad is not None
    assert torch.count_nonzero(
        model.logits.grad
    ).item() > 0
