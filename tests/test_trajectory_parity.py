import pytest

from parity_posttrain.parity.logprob_parity import (
    build_parity_report,
)
from parity_posttrain.parity.trajectory_parity import (
    TurnParityReport,
    aggregate_turn_reports,
)


def make_turn(
    *,
    turn_index: int,
    rollout_logprobs: list[float],
    trainer_logprobs: list[float],
    tolerance: float = 0.001,
) -> TurnParityReport:
    token_count = len(rollout_logprobs)

    parity = build_parity_report(
        model_name="test-model",
        device="cpu",
        dtype="torch.float32",
        token_ids=list(range(token_count)),
        token_texts=[
            f"token-{index}"
            for index in range(token_count)
        ],
        rollout_logprobs=rollout_logprobs,
        trainer_logprobs=trainer_logprobs,
        tolerance=tolerance,
    )

    return TurnParityReport(
        turn_index=turn_index,
        prompt_token_count=10,
        generated_text="generated",
        rescore_latency_ms=5.0,
        parity=parity,
    )


def test_aggregate_turn_reports() -> None:
    first = make_turn(
        turn_index=0,
        rollout_logprobs=[-0.1, -0.2],
        trainer_logprobs=[-0.1, -0.202],
    )
    second = make_turn(
        turn_index=1,
        rollout_logprobs=[-0.3],
        trainer_logprobs=[-0.301],
    )

    report = aggregate_turn_reports([first, second])

    assert report.generation_count == 2
    assert report.total_token_count == 3
    assert report.mean_absolute_error == pytest.approx(
        0.001
    )
    assert report.max_absolute_error == pytest.approx(
        0.002
    )
    assert report.p95_absolute_error == pytest.approx(
        0.002
    )
    assert report.tokens_over_tolerance == 1
    assert report.within_tolerance is False
    assert report.total_rescore_latency_ms == 10.0


def test_aggregate_exact_parity() -> None:
    turn = make_turn(
        turn_index=0,
        rollout_logprobs=[-0.5],
        trainer_logprobs=[-0.5],
    )

    report = aggregate_turn_reports([turn])

    assert report.max_absolute_error == 0.0
    assert report.within_tolerance is True


def test_aggregate_rejects_empty_turns() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        aggregate_turn_reports([])


def test_aggregate_rejects_mixed_tolerances() -> None:
    first = make_turn(
        turn_index=0,
        rollout_logprobs=[-0.1],
        trainer_logprobs=[-0.1],
        tolerance=0.001,
    )
    second = make_turn(
        turn_index=1,
        rollout_logprobs=[-0.1],
        trainer_logprobs=[-0.1],
        tolerance=0.01,
    )

    with pytest.raises(
        ValueError,
        match="same tolerance",
    ):
        aggregate_turn_reports([first, second])


def test_turn_rejects_negative_index() -> None:
    turn = make_turn(
        turn_index=-1,
        rollout_logprobs=[-0.1],
        trainer_logprobs=[-0.1],
    )

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        turn.validate()
