from __future__ import annotations

import json
from typing import Any

import pytest

from enterprise_eval.coordination import (
    MultiAgentVendorPaymentCoordinator,
)
from enterprise_eval.event_transport import (
    EventTopicRouter,
    KafkaEventPublisher,
    KafkaPublishError,
    KafkaTransportConfig,
    build_event_publisher_from_env,
)
from enterprise_eval.events import (
    InMemoryEventPublisher,
    JsonlEventPublisher,
)
from enterprise_eval.human_review import ReviewDecision


class FakeMessage:
    def __init__(
        self,
        *,
        partition: int = 0,
        offset: int = 0,
    ) -> None:
        self._partition = partition
        self._offset = offset

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class FakeProducer:
    def __init__(
        self,
        *,
        delivery_error: object | None = None,
        flush_remaining: int = 0,
        confirm_delivery: bool = True,
    ) -> None:
        self.delivery_error = delivery_error
        self.flush_remaining = flush_remaining
        self.confirm_delivery = confirm_delivery
        self.messages: list[dict[str, Any]] = []
        self.poll_calls: list[float] = []
        self.flush_calls: list[float | None] = []

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
                "value": value,
                "headers": headers,
            }
        )
        if self.confirm_delivery:
            on_delivery(
                self.delivery_error,
                FakeMessage(offset=offset),
            )

    def poll(self, timeout: float) -> int:
        self.poll_calls.append(timeout)
        return 0

    def flush(self, timeout: float | None = None) -> int:
        self.flush_calls.append(timeout)
        return self.flush_remaining


def test_topic_router_separates_event_categories() -> None:
    router = EventTopicRouter("platform.agent")

    assert router.route("workflow.created") == (
        "platform.agent.workflows.v1"
    )
    assert router.route("review.requested") == (
        "platform.agent.reviews.v1"
    )
    assert router.route("guard.action.rejected") == (
        "platform.agent.security.v1"
    )
    assert router.route("evaluation.completed") == (
        "platform.agent.evaluations.v1"
    )


def test_publisher_serializes_envelope_and_confirms_delivery() -> None:
    producer = FakeProducer()
    publisher = KafkaEventPublisher(
        producer,
        router=EventTopicRouter("platform.agent"),
        flush_timeout_seconds=3.0,
    )

    event = publisher.publish(
        "review.requested",
        run_id="run-123",
        payload={"review_id": "review-456"},
    )

    assert event.sequence == 0
    assert producer.messages[0]["topic"] == (
        "platform.agent.reviews.v1"
    )
    assert producer.messages[0]["key"] == "run-123"
    encoded = json.loads(
        producer.messages[0]["value"].decode("utf-8")
    )
    assert encoded == event.to_dict()
    assert producer.flush_calls == [3.0]

    receipt = publisher.receipt_for(event.event_id)
    assert receipt is not None
    assert receipt.topic == "platform.agent.reviews.v1"
    assert receipt.partition == 0
    assert receipt.offset == 0


def test_failed_delivery_does_not_advance_local_sequence() -> None:
    producer = FakeProducer(delivery_error="broker unavailable")
    publisher = KafkaEventPublisher(producer)

    with pytest.raises(KafkaPublishError, match="rejected"):
        publisher.publish(
            "workflow.created",
            run_id="run-failed",
            payload={},
        )

    assert publisher.list_events(run_id="run-failed") == ()

    producer.delivery_error = None
    event = publisher.publish(
        "workflow.created",
        run_id="run-failed",
        payload={},
    )
    assert event.sequence == 0


def test_flush_timeout_and_missing_confirmation_raise() -> None:
    timed_out = KafkaEventPublisher(
        FakeProducer(flush_remaining=1)
    )
    with pytest.raises(KafkaPublishError, match="flush timeout"):
        timed_out.publish(
            "workflow.created",
            run_id="run-timeout",
            payload={},
        )

    unconfirmed = KafkaEventPublisher(
        FakeProducer(confirm_delivery=False)
    )
    with pytest.raises(KafkaPublishError, match="confirmation"):
        unconfirmed.publish(
            "workflow.created",
            run_id="run-unconfirmed",
            payload={},
        )


def test_transport_config_enables_idempotent_producer() -> None:
    config = KafkaTransportConfig.from_env(
        {
            "KAFKA_BOOTSTRAP_SERVERS": "broker:19092",
            "KAFKA_CLIENT_ID": "workflow-service",
            "KAFKA_TOPIC_PREFIX": "platform.agent",
            "KAFKA_FLUSH_TIMEOUT_SECONDS": "4.5",
        }
    )

    producer_config = config.producer_config()
    assert producer_config["bootstrap.servers"] == "broker:19092"
    assert producer_config["client.id"] == "workflow-service"
    assert producer_config["enable.idempotence"] is True
    assert producer_config["acks"] == "all"
    assert config.topic_prefix == "platform.agent"
    assert config.flush_timeout_seconds == 4.5


def test_environment_factory_supports_all_transports(
    tmp_path: Any,
) -> None:
    memory = build_event_publisher_from_env(
        {"EVENT_TRANSPORT": "memory"}
    )
    assert isinstance(memory, InMemoryEventPublisher)

    jsonl = build_event_publisher_from_env(
        {
            "EVENT_TRANSPORT": "jsonl",
            "EVENT_JSONL_PATH": str(tmp_path / "events.jsonl"),
        }
    )
    assert isinstance(jsonl, JsonlEventPublisher)

    captured_config: dict[str, Any] = {}

    def factory(config: dict[str, Any]) -> FakeProducer:
        captured_config.update(config)
        return FakeProducer()

    kafka = build_event_publisher_from_env(
        {
            "EVENT_TRANSPORT": "redpanda",
            "KAFKA_BOOTSTRAP_SERVERS": "redpanda:9092",
        },
        producer_factory=factory,
    )
    assert isinstance(kafka, KafkaEventPublisher)
    assert captured_config["bootstrap.servers"] == "redpanda:9092"


def test_multi_role_workflow_publishes_confirmed_ordered_events() -> None:
    producer = FakeProducer()
    publisher = KafkaEventPublisher(producer)
    coordinator = MultiAgentVendorPaymentCoordinator(publisher)

    initial = coordinator.start("bank_account_change_review")
    run_id = str(initial["run_id"])
    review = initial["human_review"]
    assert isinstance(review, dict)

    coordinator.submit_review(
        str(review["review_id"]),
        decision=ReviewDecision.APPROVE,
        reviewer_id="finance-reviewer-01",
        reason="Verified through an independent callback.",
        bank_account_verified=True,
    )
    final = coordinator.resume(run_id)

    events = publisher.list_events(run_id=run_id)
    assert len(events) == len(producer.messages)
    assert [event.sequence for event in events] == list(
        range(len(events))
    )
    assert all(
        message["key"] == run_id
        for message in producer.messages
    )
    topics = {
        str(message["topic"])
        for message in producer.messages
    }
    assert "enterprise.agent.reviews.v1" in topics
    assert "enterprise.agent.evaluations.v1" in topics
    assert "enterprise.agent.workflows.v1" in topics
    assert final["evaluation"]["task_success"] is True
    assert final["evaluation"]["policy_violation"] is False
