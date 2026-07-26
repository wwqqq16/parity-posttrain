from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_eval.coordination import (
    MultiAgentVendorPaymentCoordinator,
)
from enterprise_eval.events import JsonlEventPublisher
from enterprise_eval.human_review import ReviewDecision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the planner-critic-executor vendor-payment demo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vendor_payment_coordination"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    publisher = JsonlEventPublisher(args.output_dir / "events.jsonl")
    coordinator = MultiAgentVendorPaymentCoordinator(publisher)

    initial = coordinator.start("bank_account_change_review")
    run_id = str(initial["run_id"])
    human_review = initial["human_review"]
    assert isinstance(human_review, dict)
    review_id = str(human_review["review_id"])

    coordinator.submit_review(
        review_id,
        decision=ReviewDecision.APPROVE,
        reviewer_id="finance-reviewer-01",
        reason="Verified through an independent callback.",
        bank_account_verified=True,
    )
    final = coordinator.resume(run_id)

    summary_path = args.output_dir / "coordination_summary.json"
    summary_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    initial_plan = initial["plans"][0]
    initial_critique = initial["critiques"][0]
    final_evaluation = final["evaluation"]
    assert isinstance(initial_plan, dict)
    assert isinstance(initial_critique, dict)
    assert isinstance(final_evaluation, dict)

    proposed_actions = [
        step["action"]["action_type"]
        for step in initial_plan["steps"]
    ]
    issue_codes = [
        issue["code"]
        for issue in initial_critique["issues"]
    ]

    print("MULTI-AGENT VENDOR PAYMENT DEMO")
    print("=" * 52)
    print(
        "Planner initially proposed payment:",
        "approve_vendor_payment" in proposed_actions,
    )
    print("Initial critic decision:", initial_critique["decision"])
    print("Critic issue codes:", ", ".join(issue_codes))
    print("Workflow paused for review:", initial["status"] == "review_required")
    print("Workflow resumed:", final["state"]["workflow_resumed"])
    print("Payment approved:", final["state"]["payment_approved"])
    print("Task success:", final_evaluation["task_success"])
    print("Policy violation:", final_evaluation["policy_violation"])
    print("Component calls:", final["component_calls"])
    print("Event log:", publisher.output_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
