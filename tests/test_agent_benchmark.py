import pytest

from parity_posttrain.benchmarks.agent_benchmark import (
    AgentBenchmarkRecord,
    build_benchmark_summary,
)


def make_record(
    *,
    task_id: str,
    category: str,
    status: str = "completed",
    answer_correct: bool = True,
    reward: float = 1.0,
    parity_passed: bool = True,
) -> AgentBenchmarkRecord:
    assert status in {
        "completed",
        "protocol_error",
        "max_steps",
    }

    return AgentBenchmarkRecord(
        task_id=task_id,
        category=category,
        status=status,
        answer_correct=answer_correct,
        reward=reward,
        tool_coverage=1.0,
        used_tools=["calculator"],
        missing_tools=[],
        unexpected_tools=[],
        latency_ms=100.0,
        generation_count=2,
        generated_token_count=10,
        parity_within_tolerance=parity_passed,
        parity_mean_absolute_error=0.0001,
        parity_max_absolute_error=0.0002,
        parity_tokens_over_tolerance=(
            0 if parity_passed else 1
        ),
    )


def test_build_benchmark_summary() -> None:
    records = [
        make_record(
            task_id="calculator_001",
            category="calculator",
        ),
        make_record(
            task_id="shopping_001",
            category="shopping",
            answer_correct=False,
            reward=0.3,
        ),
        make_record(
            task_id="shopping_002",
            category="shopping",
            status="protocol_error",
            answer_correct=False,
            reward=0.0,
            parity_passed=False,
        ),
    ]

    summary = build_benchmark_summary(records)

    assert summary.task_count == 3
    assert summary.completed_count == 2
    assert summary.protocol_error_count == 1
    assert summary.max_steps_count == 0
    assert summary.answer_correct_count == 1
    assert summary.full_reward_count == 1
    assert summary.completion_rate == pytest.approx(
        2 / 3,
        abs=1e-6,
    )
    assert summary.answer_accuracy == pytest.approx(
        1 / 3,
        abs=1e-6,
    )
    assert summary.mean_reward == pytest.approx(
        1.3 / 3,
        abs=1e-6,
    )
    assert summary.total_generated_tokens == 30
    assert summary.parity_pass_count == 2
    assert summary.parity_pass_rate == pytest.approx(
        2 / 3,
        abs=1e-6,
    )
    assert summary.total_tokens_over_tolerance == 1

    assert set(summary.by_category) == {
        "calculator",
        "shopping",
    }
    assert (
        summary.by_category["calculator"].answer_accuracy
        == 1.0
    )
    assert (
        summary.by_category["shopping"].answer_accuracy
        == 0.0
    )


def test_empty_benchmark_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_benchmark_summary([])


def test_invalid_reward_is_rejected() -> None:
    record = make_record(
        task_id="invalid",
        category="calculator",
    )
    record.reward = 1.5

    with pytest.raises(
        ValueError,
        match="reward must be between",
    ):
        build_benchmark_summary([record])
