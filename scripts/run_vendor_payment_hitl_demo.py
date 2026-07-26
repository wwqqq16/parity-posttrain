from __future__ import annotations

import argparse
from pathlib import Path

from enterprise_eval.artifacts import write_run_artifact
from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.models import ActionType, AgentAction
from enterprise_eval.vendor_payment_cases import get_vendor_payment_case
from enterprise_eval.vendor_payment_environment import VendorPaymentEnvironment
from enterprise_eval.vendor_payment_evaluator import VendorPaymentEvaluator


def _action(action_type: ActionType, **arguments: object) -> AgentAction:
    return AgentAction(action_type, arguments)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic vendor-payment HITL demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vendor_payment_hitl"),
    )
    args = parser.parse_args()

    case = get_vendor_payment_case("bank_account_change_review")
    env = VendorPaymentEnvironment(case)
    env.reset()

    for action_type in (
        ActionType.GET_INVOICE,
        ActionType.VERIFY_PURCHASE_ORDER,
        ActionType.CHECK_DUPLICATE_INVOICE,
        ActionType.CHECK_BUDGET,
        ActionType.VERIFY_VENDOR_BANK_ACCOUNT,
    ):
        env.step(_action(action_type, invoice_id=case.invoice_id))

    payment_action = _action(
        ActionType.APPROVE_VENDOR_PAYMENT,
        invoice_id=case.invoice_id,
    )
    guard_result = env.inspect_execution_guard(payment_action)
    if guard_result is None:
        raise RuntimeError("Expected the bank-account guard to block payment.")
    assert env.run is not None
    env.run.add_step(payment_action, guard_result)

    env.step(
        _action(
            ActionType.REQUEST_HUMAN_REVIEW,
            reason=(
                "Changed vendor bank account requires independent finance "
                "verification."
            ),
        )
    )
    env.submit_human_review(
        reviewer_id="finance-reviewer-01",
        decision=ReviewDecision.APPROVE,
        reason="New account verified through an independent callback.",
        bank_account_verified=True,
    )
    env.resume_after_human_review()

    second_guard = env.inspect_execution_guard(payment_action)
    if second_guard is not None:
        raise RuntimeError(second_guard.observation)
    env.step(payment_action)
    env.step(
        _action(
            ActionType.RESPOND,
            message=(
                "The independently verified vendor payment was approved."
            ),
        )
    )

    evaluation = VendorPaymentEvaluator().evaluate(env)
    artifact_path = write_run_artifact(env.run, evaluation, args.output_dir)

    print("VENDOR PAYMENT HITL DEMO")
    print("=" * 52)
    print("Case:", case.case_id)
    print("Initial guard:", guard_result.metadata["recommended_action"])
    print("Human review requested:", env.state.human_review_requested)
    print("Human review approved:", env.state.human_review_approved)
    print("Bank account verified:", env.state.bank_account_verified_by_human)
    print("Workflow resumed:", env.state.workflow_resumed)
    print("Payment approved:", env.state.payment_approved)
    print("Task success:", evaluation.task_success)
    print("Policy violation:", evaluation.policy_violation)
    print("Final reward:", evaluation.final_reward)
    print("Artifact:", artifact_path)
    print()
    print("HUMAN REVIEW EVENTS")
    events = env.run.metadata["human_review_events"]
    assert isinstance(events, list)
    for event in events:
        print(f"{event['sequence']}: {event['event_type']}")


if __name__ == "__main__":
    main()
