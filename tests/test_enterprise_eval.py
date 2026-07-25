from __future__ import annotations

from enterprise_eval.cases import CASES
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.scripted_agent import ScriptedRefundAgent


def run(case_id: str):
    env = RefundEnvironment(CASES[case_id])
    ScriptedRefundAgent().run(env)
    return env, RefundEvaluator().evaluate(env)


def test_eligible_order_is_refunded() -> None:
    env, result = run("eligible_standard")
    assert result.task_success
    assert result.expected_outcome == "refund"
    assert env.state.refund_issued
    assert not env.state.human_review_requested


def test_high_value_order_is_escalated() -> None:
    env, result = run("high_value")
    assert result.task_success
    assert result.expected_outcome == "escalate"
    assert env.state.human_review_requested
    assert not env.state.refund_issued


def test_outside_window_is_denied_without_escalation() -> None:
    env, result = run("outside_window")
    assert result.task_success
    assert result.expected_outcome == "deny"
    assert not env.state.refund_issued
    assert not env.state.human_review_requested


def test_disputed_payment_is_escalated() -> None:
    env, result = run("disputed_payment")
    assert result.task_success
    assert env.state.human_review_requested


def test_conflicting_delivery_is_escalated() -> None:
    env, result = run("conflicting_delivery")
    assert result.task_success
    assert env.state.human_review_requested
