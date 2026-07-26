from __future__ import annotations

import argparse

from enterprise_eval.coordination import (
    MultiAgentVendorPaymentCoordinator,
)
from enterprise_eval.event_transport import (
    KafkaEventPublisher,
    build_event_publisher_from_env,
)
from enterprise_eval.human_review import ReviewDecision


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a complete vendor-payment workflow to a configured "
            "Kafka or Redpanda broker."
        )
    )
    parser.add_argument(
        "--case-id",
        default="bank_account_change_review",
    )
    args = parser.parse_args()

    publisher = build_event_publisher_from_env()
    if not isinstance(publisher, KafkaEventPublisher):
        raise RuntimeError(
            "Set EVENT_TRANSPORT=kafka or EVENT_TRANSPORT=redpanda."
        )

    coordinator = MultiAgentVendorPaymentCoordinator(publisher)
    initial = coordinator.start(args.case_id)
    run_id = str(initial["run_id"])

    review = initial["human_review"]
    if isinstance(review, dict):
        coordinator.submit_review(
            str(review["review_id"]),
            decision=ReviewDecision.APPROVE,
            reviewer_id="finance-reviewer-01",
            reason="Verified through an independent callback.",
            bank_account_verified=True,
        )
        final = coordinator.resume(run_id)
    else:
        final = initial

    evaluation = final["evaluation"]
    assert isinstance(evaluation, dict)
    events = publisher.list_events(run_id=run_id)

    print("KAFKA WORKFLOW SMOKE TEST")
    print("=" * 52)
    print("Run ID:", run_id)
    print("Confirmed events:", len(events))
    print("Task success:", evaluation["task_success"])
    print("Policy violation:", evaluation["policy_violation"])


if __name__ == "__main__":
    main()
