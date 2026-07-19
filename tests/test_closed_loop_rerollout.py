"""Tests for agent-task closed-loop rerollout."""

from __future__ import annotations

import pytest

from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.core.task import AgentTask
from parity_posttrain.rollout.hf_backend import (
    GenerationResult,
)
from parity_posttrain.training import (
    rerollout_agent_tasks,
)


def make_generation(
    text: str,
    token_ids: list[int],
) -> GenerationResult:
    """Create one deterministic fake generation."""

    return GenerationResult(
        model_name="fake-model",
        device="cpu",
        prompt_text="prompt",
        generated_text=text,
        prompt_token_ids=[1, 2],
        generated_token_ids=token_ids,
        generated_token_logprobs=[
            -0.1
            for _ in token_ids
        ],
        latency_ms=10.0,
        tokens_per_second=100.0,
    )


class FakeBackend:
    """Return a predetermined generation sequence."""

    def __init__(
        self,
        generations: list[GenerationResult],
    ) -> None:
        self.generations = generations.copy()
        self.call_count = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        """Return the next fake generation."""

        del messages
        del max_new_tokens

        self.call_count += 1

        if not self.generations:
            raise RuntimeError(
                "fake backend has no generation left"
            )

        return self.generations.pop(0)


def make_task(
    task_id: str,
    *,
    category: str = "calculator",
) -> AgentTask:
    """Create one deterministic calculator task."""

    return AgentTask(
        task_id=task_id,
        prompt=(
            "Use the calculator to evaluate 17 * 6."
        ),
        expected_answer="102",
        required_tools=["calculator"],
        metadata={
            "category": category,
        },
    )


def test_rerollout_builds_snapshots_in_task_order() -> None:
    backend = FakeBackend(
        [
            make_generation(
                (
                    '{"type":"tool_call",'
                    '"name":"calculator",'
                    '"arguments":{'
                    '"expression":"17 * 6"}}'
                ),
                [10, 11],
            ),
            make_generation(
                '{"type":"final","answer":"102"}',
                [12],
            ),
            make_generation(
                "The answer is probably 102.",
                [20, 21, 22],
            ),
        ]
    )
    runner = AgentRunner(
        backend,
        max_steps=3,
        max_new_tokens=64,
    )

    snapshots = rerollout_agent_tasks(
        runner=runner,
        tasks=(
            make_task("completed_task"),
            make_task("failed_task"),
        ),
    )

    assert [
        snapshot.task_id
        for snapshot in snapshots
    ] == [
        "completed_task",
        "failed_task",
    ]

    completed = snapshots[0]

    assert completed.category == "calculator"
    assert completed.status == "completed"
    assert completed.reward == pytest.approx(1.0)
    assert completed.answer_correct is True
    assert completed.generation_count == 2
    assert completed.generated_token_count == 3

    failed = snapshots[1]

    assert failed.status == "protocol_error"
    # A protocol error can still receive answer credit.
    # Missing tool coverage leaves the reward at 0.7.
    assert failed.reward == pytest.approx(0.7)
    assert failed.answer_correct is True
    assert failed.generation_count == 1
    assert failed.generated_token_count == 3
    assert backend.call_count == 3


def test_rejects_empty_task_sequence() -> None:
    runner = AgentRunner(FakeBackend([]))

    with pytest.raises(
        ValueError,
        match="tasks must not be empty",
    ):
        rerollout_agent_tasks(
            runner=runner,
            tasks=(),
        )


def test_rejects_duplicate_task_ids() -> None:
    runner = AgentRunner(FakeBackend([]))
    task = make_task("duplicate")

    with pytest.raises(
        ValueError,
        match="task IDs must be unique",
    ):
        rerollout_agent_tasks(
            runner=runner,
            tasks=(task, task),
        )


def test_rejects_missing_task_category() -> None:
    runner = AgentRunner(FakeBackend([]))
    task = make_task(
        "missing_category",
        category="",
    )

    with pytest.raises(
        ValueError,
        match="has no valid category",
    ):
        rerollout_agent_tasks(
            runner=runner,
            tasks=(task,),
        )
