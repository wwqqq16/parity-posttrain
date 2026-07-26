from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterprise_eval.coordination import (
    MultiAgentVendorPaymentCoordinator,
)
from enterprise_eval.event_transport import (
    EventTopicRouter,
    KafkaEventPublisher,
)
from enterprise_eval.human_review import ReviewDecision


@dataclass(frozen=True)
class DemoMessageMetadata:
    topic_name: str
    partition_id: int
    offset_value: int

    def topic(self) -> str:
        return self.topic_name

    def partition(self) -> int:
        return self.partition_id

    def offset(self) -> int:
        return self.offset_value


class DemoProducer:
    """Local producer double that exercises the real transport contract."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def produce(
        self,
        *,
        topic: str,
        key: str,
        value: bytes,
        on_delivery: Any,
        headers: list[tuple[str, bytes]],
    ) -> None:
        offset = len(self.messages)
        self.messages.append(
            {
                "topic": topic,
                "key": key,
                "value": json.loads(value.decode("utf-8")),
                "headers": [
                    [name, header_value.decode("utf-8")]
                    for name, header_value in headers
                ],
                "partition": 0,
                "offset": offset,
            }
        )
        on_delivery(
            None,
            DemoMessageMetadata(topic, 0, offset),
        )

    def poll(self, timeout: float) -> int:
        del timeout
        return 0

    def flush(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise Kafka/Redpanda event contracts with a local "
            "delivery-confirming producer double."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/event_transport"),
    )
    args = parser.parse_args()

    producer = DemoProducer()
    publisher = KafkaEventPublisher(
        producer,
        router=EventTopicRouter("enterprise.agent"),
    )
    coordinator = MultiAgentVendorPaymentCoordinator(publisher)

    initial = coordinator.start("bank_account_change_review")
    run_id = str(initial["run_id"])
    review = initial["human_review"]
    assert isinstance(review, dict)
    review_id = str(review["review_id"])

    coordinator.submit_review(
        review_id,
        decision=ReviewDecision.APPROVE,
        reviewer_id="finance-reviewer-01",
        reason="Verified through an independent callback.",
        bank_account_verified=True,
    )
    final = coordinator.resume(run_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "published_messages.json"
    output_path.write_text(
        json.dumps(producer.messages, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    topics = sorted(
        {str(message["topic"]) for message in producer.messages}
    )
    events = publisher.list_events(run_id=run_id)
    sequences = [event.sequence for event in events]
    evaluation = final["evaluation"]
    assert isinstance(evaluation, dict)

    print("EVENT TRANSPORT DEMO")
    print("=" * 52)
    print("Delivery-confirmed messages:", len(producer.messages))
    print("Topics:", ", ".join(topics))
    print("Run-keyed messages:", all(
        message["key"] == run_id for message in producer.messages
    ))
    print("Ordered sequences:", sequences == list(range(len(sequences))))
    print("Task success:", evaluation["task_success"])
    print("Policy violation:", evaluation["policy_violation"])
    print("Artifact:", output_path)


if __name__ == "__main__":
    main()
