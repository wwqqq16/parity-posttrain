"""Tests for the online enterprise RL control plane."""

from __future__ import annotations

import pytest

from enterprise_eval.cases import CASES
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    Difficulty,
    FailureProfile,
)
from enterprise_eval.rl_environment import (
    EnterpriseRefundRLEnvironment,
)
from enterprise_eval.task_factory import (
    TaskFactoryConfig,
    generate_failure_surface_cases,
)


def _refund_actions(
    order_id: str = "ORD-1001",
) -> tuple[AgentAction, ...]:
    return (
        AgentAction(
            ActionType.GET_ORDER,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.CHECK_REFUND_POLICY,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.GET_PAYMENT_STATUS,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.ISSUE_REFUND,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.RESPOND,
            {"message": "Your refund has been issued."},
        ),
    )


def test_reset_has_deterministic_state_fingerprint() -> None:
    first = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"]
    )
    second = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"]
    )

    first_reset = first.reset()
    second_reset = second.reset()

    assert first_reset.observation == second_reset.observation
    assert (
        first_reset.state_fingerprint
        == second_reset.state_fingerprint
    )
    assert (
        first.environment.run is not None
        and second.environment.run is not None
    )
    assert (
        first.environment.run.run_id
        != second.environment.run.run_id
    )


def test_successful_episode_returns_dense_online_rewards() -> None:
    env = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"]
    )
    env.reset()

    transitions = [
        env.step(action)
        for action in _refund_actions()
    ]

    assert [
        transition.reward
        for transition in transitions
    ] == [
        0.1,
        0.1,
        0.1,
        0.25,
        1.0,
    ]
    assert sum(
        transition.reward
        for transition in transitions
    ) == 1.55
    assert transitions[-1].terminated
    assert not transitions[-1].truncated
    assert transitions[-1].info[
        "reward_components"
    ] == {
        "terminal_outcome_reward": 1.0,
    }


def test_snapshot_restore_replays_identical_transition() -> None:
    env = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"]
    )
    env.reset()
    env.step(_refund_actions()[0])
    snapshot = env.snapshot()

    first = env.step(_refund_actions()[1])
    restored = env.restore(snapshot)
    second = env.step(_refund_actions()[1])

    assert restored == snapshot.state_fingerprint
    assert first.reward == second.reward
    assert first.observation == second.observation
    assert (
        first.state_fingerprint
        == second.state_fingerprint
    )


def test_execution_guard_blocks_unsafe_side_effect() -> None:
    env = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"],
        enforce_execution_guard=True,
    )
    env.reset()

    transition = env.step(
        AgentAction(
            ActionType.ISSUE_REFUND,
            {"order_id": "ORD-1001"},
        )
    )

    assert transition.reward == -0.5
    assert not transition.terminated
    assert not env.environment.state.refund_issued
    assert (
        transition.info["tool_metadata"][
            "error_type"
        ]
        == "execution_guard_rejection"
    )


def test_retryable_failure_receives_recovery_credit() -> None:
    case = next(
        case
        for case in generate_failure_surface_cases(
            TaskFactoryConfig(variants_per_cell=1)
        )
        if case.difficulty is Difficulty.EASY
        and case.failure_profile
        is FailureProfile.TRANSIENT_TOOL_TIMEOUT
        and case.failure_injection_step == 2
    )
    env = EnterpriseRefundRLEnvironment(case)
    env.reset()
    order_id = case.order_id

    env.step(
        AgentAction(
            ActionType.GET_ORDER,
            {"order_id": order_id},
        )
    )
    env.step(
        AgentAction(
            ActionType.CHECK_REFUND_POLICY,
            {"order_id": order_id},
        )
    )
    failed = env.step(
        AgentAction(
            ActionType.GET_PAYMENT_STATUS,
            {"order_id": order_id},
        )
    )
    recovered = env.step(
        AgentAction(
            ActionType.GET_PAYMENT_STATUS,
            {"order_id": order_id},
        )
    )

    assert failed.reward == 0.0
    assert recovered.reward == 0.35
    assert recovered.info["reward_components"] == {
        "evidence_progress": 0.1,
        "recovery_credit": 0.25,
    }


def test_step_budget_returns_truncation_signal() -> None:
    env = EnterpriseRefundRLEnvironment(
        CASES["eligible_standard"],
        max_steps=2,
    )
    env.reset()
    env.step(_refund_actions()[0])
    truncated = env.step(_refund_actions()[1])

    assert not truncated.terminated
    assert truncated.truncated
    assert truncated.reward == -0.4
    assert truncated.info["reward_components"][
        "truncation_penalty"
    ] == -0.5

    with pytest.raises(
        RuntimeError,
        match="truncated",
    ):
        env.step(_refund_actions()[2])
