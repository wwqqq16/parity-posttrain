import pytest
import torch

from parity_posttrain.rollout.hf_backend import (
    GenerationResult,
    gather_generated_logprobs,
    select_dtype,
)


def test_gather_generated_logprobs() -> None:
    scores = (
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[3.0, 2.0, 1.0]]),
    )
    generated_ids = torch.tensor([[2, 0]])

    result = gather_generated_logprobs(
        scores,
        generated_ids,
    )

    expected_first = torch.log_softmax(
        scores[0],
        dim=-1,
    )[0, 2]
    expected_second = torch.log_softmax(
        scores[1],
        dim=-1,
    )[0, 0]

    assert result.shape == (1, 2)
    assert result[0, 0].item() == pytest.approx(
        expected_first.item()
    )
    assert result[0, 1].item() == pytest.approx(
        expected_second.item()
    )


def test_gather_generated_logprobs_rejects_empty_scores() -> None:
    generated_ids = torch.tensor([[0]])

    with pytest.raises(ValueError, match="must not be empty"):
        gather_generated_logprobs(
            (),
            generated_ids,
        )


def test_gather_generated_logprobs_rejects_shape_mismatch() -> None:
    scores = (
        torch.tensor([[1.0, 2.0]]),
    )
    generated_ids = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="matching shapes"):
        gather_generated_logprobs(
            scores,
            generated_ids,
        )


def test_generation_result_requires_aligned_metadata() -> None:
    result = GenerationResult(
        model_name="test-model",
        device="cpu",
        prompt_text="prompt",
        generated_text="answer",
        prompt_token_ids=[1, 2],
        generated_token_ids=[3, 4],
        generated_token_logprobs=[-0.1],
        latency_ms=10.0,
        tokens_per_second=20.0,
    )

    with pytest.raises(ValueError, match="must have equal lengths"):
        result.validate()


def test_cpu_uses_float32() -> None:
    assert select_dtype(torch.device("cpu")) == torch.float32


def test_accelerators_use_float16() -> None:
    assert select_dtype(torch.device("mps")) == torch.float16
    assert select_dtype(torch.device("cuda")) == torch.float16
