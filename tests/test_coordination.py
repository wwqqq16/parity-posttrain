from __future__ import annotations

from enterprise_eval.coordination import (
    CoordinationSession,
    CoordinationStep,
    CritiqueDecision,
    ExecutionProposal,
    GuardedVendorPaymentExecutor,
    MultiAgentVendorPaymentCoordinator,
)
from enterprise_eval.events import InMemoryEventPublisher
from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.models import ActionType, AgentAction, Architecture
from enterprise_eval.vendor_payment_cases import get_vendor_payment_case
from enterprise_eval.vendor_payment_environment import VendorPaymentEnvironment


def test_critic_revises_unsafe_initial_payment_plan() -> None:
    coordinator = MultiAgentVendorPaymentCoordinator()
    snapshot = coordinator.start("bank_account_change_review")

    plan_actions = [
        step["action"]["action_type"]
        for step in snapshot["plans"][0]["steps"]
    ]
    approved_actions = [
        step["action"]["action_type"]
        for step in snapshot["critiques"][0]["approved_steps"]
    ]

    assert "approve_vendor_payment" in plan_actions
    assert snapshot["critiques"][0]["decision"] == "revise"
    assert "approve_vendor_payment" not in approved_actions
    assert "request_human_review" in approved_actions
    assert snapshot["status"] == "review_required"
    assert snapshot["state"]["payment_approved"] is False


def test_approved_review_resume_completes_payment() -> None:
    coordinator = MultiAgentVendorPaymentCoordinator()
    initial = coordinator.start("bank_account_change_review")
    run_id = initial["run_id"]
    review_id = initial["human_review"]["review_id"]

    reviewed = coordinator.submit_review(
        review_id,
        decision=ReviewDecision.APPROVE,
        reviewer_id="finance-reviewer-01",
        reason="Verified through an independent callback.",
        bank_account_verified=True,
    )
    assert reviewed["status"] == "review_completed"

    final = coordinator.resume(run_id)

    assert final["status"] == "completed"
    assert final["state"]["payment_approved"] is True
    assert final["evaluation"]["task_success"] is True
    assert final["evaluation"]["policy_violation"] is False
    assert final["component_calls"] == 6
    assert len(final["plans"]) == 2
    assert final["critiques"][1]["decision"] == "approve"


def test_rejected_review_remains_unpaid() -> None:
    coordinator = MultiAgentVendorPaymentCoordinator()
    initial = coordinator.start("bank_account_change_review")
    run_id = initial["run_id"]
    review_id = initial["human_review"]["review_id"]

    coordinator.submit_review(
        review_id,
        decision=ReviewDecision.REJECT,
        reviewer_id="finance-reviewer-02",
        reason="Independent verification failed.",
    )
    final = coordinator.resume(run_id)

    assert final["status"] == "completed"
    assert final["state"]["payment_approved"] is False
    assert final["evaluation"]["task_success"] is False
    assert final["evaluation"]["failure_type"] == "human_review_rejected"
    assert final["critiques"][1]["decision"] == "revise"


def test_executor_rechecks_runtime_guard() -> None:
    publisher = InMemoryEventPublisher()
    case = get_vendor_payment_case("bank_account_change_review")
    env = VendorPaymentEnvironment(case)
    env.reset(architecture=Architecture.PLANNER_CRITIC)
    assert env.run is not None

    step = CoordinationStep(
        step_id="malicious-payment",
        action=AgentAction(
            ActionType.APPROVE_VENDOR_PAYMENT,
            {"invoice_id": case.invoice_id},
        ),
        rationale="Unsafe direct payment attempt.",
    )
    proposal = ExecutionProposal(
        proposal_id="proposal-unsafe",
        run_id=env.run.run_id,
        plan_id="plan-unsafe",
        critique_id="critique-unsafe",
        approved=True,
        steps=(step,),
    )
    session = CoordinationSession(env=env)

    GuardedVendorPaymentExecutor().execute(
        session,
        proposal,
        publisher,
    )

    assert env.state.payment_approved is False
    events = publisher.list_events(run_id=env.run.run_id)
    assert events[-1].event_type == "guard.action.rejected"
    assert events[-1].payload["recommended_action"] == "request_human_review"
    assert "invoice_not_verified" in events[-1].payload["reasons"]
    assert (
        "bank_account_change_requires_review"
        in events[-1].payload["reasons"]
    )


def test_coordination_events_are_ordered_and_versioned() -> None:
    publisher = InMemoryEventPublisher()
    coordinator = MultiAgentVendorPaymentCoordinator(publisher)
    initial = coordinator.start("bank_account_change_review")
    run_id = initial["run_id"]
    review_id = initial["human_review"]["review_id"]

    coordinator.submit_review(
        review_id,
        decision=ReviewDecision.APPROVE,
        reviewer_id="finance-reviewer-01",
        reason="Verified through an independent callback.",
        bank_account_verified=True,
    )
    final = coordinator.resume(run_id)

    events = final["events"]
    assert [event["sequence"] for event in events] == list(
        range(len(events))
    )
    assert all(event["schema_version"] == "1.0" for event in events)
    event_types = [event["event_type"] for event in events]
    assert event_types[0] == "workflow.created"
    assert event_types.count("agent.plan.created") == 2
    assert event_types.count("critic.review.completed") == 2
    assert "review.requested" in event_types
    assert "review.completed" in event_types
    assert "workflow.resumed" in event_types
    assert event_types[-1] == "evaluation.completed"


def test_valid_payment_plan_is_approved_without_review() -> None:
    coordinator = MultiAgentVendorPaymentCoordinator()
    snapshot = coordinator.start("valid_payment")

    assert snapshot["status"] == "completed"
    assert snapshot["critiques"][0]["decision"] == CritiqueDecision.APPROVE.value
    assert snapshot["state"]["human_review_requested"] is False
    assert snapshot["state"]["payment_approved"] is True
    assert snapshot["evaluation"]["task_success"] is True
