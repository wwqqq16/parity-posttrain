import pytest
import torch

from parity_posttrain.parity.forced_sequence import (
    ForcedContinuationProcessor,
    ForcedRolloutResult,
)


def test_processor_forces_current_token() -> None:
    processor = ForcedContinuationProcessor(
        prompt_length=3,
        token_ids=[2, 1],
    )
    scores = torch.zeros((1, 4))

    first = processor(
        torch.tensor([[10, 11, 12]]),
        scores,
    )
    second = processor(
        torch.tensor([[10, 11, 12, 2]]),
        scores,
    )

    assert first[0, 2].item() == 0.0
    assert torch.isneginf(first[0, 0])
    assert torch.isneginf(first[0, 1])
    assert torch.isneginf(first[0, 3])

    assert second[0, 1].item() == 0.0
    assert torch.isneginf(second[0, 2])


def test_processor_rejects_out_of_range_step() -> None:
    processor = ForcedContinuationProcessor(
        prompt_length=2,
        token_ids=[1],
    )

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        processor(
            torch.tensor([[10, 11, 1]]),
            torch.zeros((1, 4)),
        )


def test_processor_rejects_empty_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        ForcedContinuationProcessor(
            prompt_length=2,
            token_ids=[],
        )


def test_forced_result_requires_logprobs() -> None:
    result = ForcedRolloutResult(
        use_cache=True,
        token_logprobs=[],
        latency_ms=10.0,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        result.validate()
