import pytest
import torch

from parity_posttrain.parity.logprob_parity import (
    build_parity_report,
    gather_teacher_forced_logprobs,
)


def test_teacher_forced_gather_uses_causal_shift() -> None:
    logits = torch.zeros((1, 5, 4))

    logits[0, 2] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    logits[0, 3] = torch.tensor([0.0, 0.0, 3.0, 0.0])

    generated_ids = torch.tensor([[1, 2]])

    result = gather_teacher_forced_logprobs(
        logits=logits,
        prompt_length=3,
        generated_token_ids=generated_ids,
    )

    expected_first = torch.log_softmax(
        logits[0, 2],
        dim=-1,
    )[1]
    expected_second = torch.log_softmax(
        logits[0, 3],
        dim=-1,
    )[2]

    assert result.shape == (1, 2)
    assert result[0, 0].item() == pytest.approx(
        expected_first.item()
    )
    assert result[0, 1].item() == pytest.approx(
        expected_second.item()
    )


def test_teacher_forced_gather_rejects_zero_prompt() -> None:
    logits = torch.zeros((1, 2, 4))
    generated_ids = torch.tensor([[1]])

    with pytest.raises(
        ValueError,
        match="prompt_length must be positive",
    ):
        gather_teacher_forced_logprobs(
            logits=logits,
            prompt_length=0,
            generated_token_ids=generated_ids,
        )


def test_teacher_forced_gather_rejects_short_logits() -> None:
    logits = torch.zeros((1, 3, 4))
    generated_ids = torch.tensor([[1, 2]])

    with pytest.raises(
        ValueError,
        match="too short",
    ):
        gather_teacher_forced_logprobs(
            logits=logits,
            prompt_length=3,
            generated_token_ids=generated_ids,
        )


def test_build_parity_report_computes_errors() -> None:
    report = build_parity_report(
        model_name="test-model",
        device="cpu",
        dtype="float32",
        token_ids=[10, 11],
        token_texts=["a", "b"],
        rollout_logprobs=[-0.1, -0.2],
        trainer_logprobs=[-0.1, -0.202],
        tolerance=0.001,
    )

    assert report.token_count == 2
    assert report.mean_absolute_error == pytest.approx(0.001)
    assert report.max_absolute_error == pytest.approx(0.002)
    assert report.p95_absolute_error == pytest.approx(0.002)
    assert report.tokens_over_tolerance == 1
    assert report.within_tolerance is False


def test_build_parity_report_detects_exact_parity() -> None:
    report = build_parity_report(
        model_name="test-model",
        device="cpu",
        dtype="float32",
        token_ids=[10],
        token_texts=["a"],
        rollout_logprobs=[-0.5],
        trainer_logprobs=[-0.5],
    )

    assert report.max_absolute_error == 0.0
    assert report.within_tolerance is True


def test_build_parity_report_rejects_misalignment() -> None:
    with pytest.raises(ValueError, match="must align"):
        build_parity_report(
            model_name="test-model",
            device="cpu",
            dtype="float32",
            token_ids=[10, 11],
            token_texts=["a"],
            rollout_logprobs=[-0.1, -0.2],
            trainer_logprobs=[-0.1, -0.2],
        )
