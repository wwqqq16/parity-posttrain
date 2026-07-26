from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from enterprise_eval.events import (
    EventEnvelope,
    EventPublisher,
    InMemoryEventPublisher,
    JsonlEventPublisher,
)


class KafkaPublishError(RuntimeError):
    """Raised when an event cannot be confirmed by the broker transport."""


class ProducerProtocol(Protocol):
    def produce(
        self,
        *,
        topic: str,
        key: str,
        value: bytes,
        on_delivery: Callable[[Any, Any], None],
        headers: list[tuple[str, bytes]],
    ) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True)
class KafkaDeliveryReceipt:
    event_id: str
    topic: str
    partition: int | None
    offset: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KafkaTransportConfig:
    bootstrap_servers: str = "127.0.0.1:9092"
    client_id: str = "enterprise-agent-reliability"
    topic_prefix: str = "enterprise.agent"
    flush_timeout_seconds: float = 10.0
    security_protocol: str | None = None
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> KafkaTransportConfig:
        values = os.environ if env is None else env
        timeout_raw = values.get("KAFKA_FLUSH_TIMEOUT_SECONDS", "10")
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                "KAFKA_FLUSH_TIMEOUT_SECONDS must be numeric."
            ) from exc
        if timeout <= 0:
            raise ValueError(
                "KAFKA_FLUSH_TIMEOUT_SECONDS must be greater than zero."
            )

        return cls(
            bootstrap_servers=values.get(
                "KAFKA_BOOTSTRAP_SERVERS",
                "127.0.0.1:9092",
            ),
            client_id=values.get(
                "KAFKA_CLIENT_ID",
                "enterprise-agent-reliability",
            ),
            topic_prefix=values.get(
                "KAFKA_TOPIC_PREFIX",
                "enterprise.agent",
            ).strip("."),
            flush_timeout_seconds=timeout,
            security_protocol=values.get("KAFKA_SECURITY_PROTOCOL"),
            sasl_mechanism=values.get("KAFKA_SASL_MECHANISM"),
            sasl_username=values.get("KAFKA_SASL_USERNAME"),
            sasl_password=values.get("KAFKA_SASL_PASSWORD"),
        )

    def producer_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": self.client_id,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 5,
        }
        if self.security_protocol:
            config["security.protocol"] = self.security_protocol
        if self.sasl_mechanism:
            config["sasl.mechanism"] = self.sasl_mechanism
        if self.sasl_username:
            config["sasl.username"] = self.sasl_username
        if self.sasl_password:
            config["sasl.password"] = self.sasl_password
        return config


@dataclass(frozen=True)
class EventTopicRouter:
    topic_prefix: str = "enterprise.agent"

    def route(self, event_type: str) -> str:
        if event_type.startswith("review."):
            category = "reviews"
        elif event_type.startswith("evaluation."):
            category = "evaluations"
        elif event_type.startswith("guard."):
            category = "security"
        else:
            category = "workflows"
        return f"{self.topic_prefix}.{category}.v1"


class KafkaEventPublisher:
    """Synchronous Kafka/Redpanda publisher with delivery confirmation.

    Confirmed events are also retained in a process-local cache so the
    existing control-plane read API remains usable in a single-process demo.
    Kafka remains the durable transport; the local cache is not a replacement
    for a shared event store.
    """

    def __init__(
        self,
        producer: ProducerProtocol,
        *,
        router: EventTopicRouter | None = None,
        flush_timeout_seconds: float = 10.0,
    ) -> None:
        if flush_timeout_seconds <= 0:
            raise ValueError(
                "flush_timeout_seconds must be greater than zero."
            )
        self.producer = producer
        self.router = router or EventTopicRouter()
        self.flush_timeout_seconds = flush_timeout_seconds
        self._events: list[EventEnvelope] = []
        self._sequence_by_run: dict[str, int] = {}
        self._receipts: dict[str, KafkaDeliveryReceipt] = {}

    def publish(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        event = self._build_event(
            event_type,
            run_id=run_id,
            payload=payload,
        )
        topic = self.router.route(event_type)
        callback_called = False
        delivery_error: Any = None
        delivery_message: Any = None

        def on_delivery(error: Any, message: Any) -> None:
            nonlocal callback_called, delivery_error, delivery_message
            callback_called = True
            delivery_error = error
            delivery_message = message

        try:
            self.producer.produce(
                topic=topic,
                key=run_id,
                value=json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                on_delivery=on_delivery,
                headers=[
                    ("event_type", event_type.encode("utf-8")),
                    (
                        "schema_version",
                        event.schema_version.encode("utf-8"),
                    ),
                    ("event_id", event.event_id.encode("utf-8")),
                ],
            )
            self.producer.poll(0.0)
            remaining = self.producer.flush(
                self.flush_timeout_seconds
            )
        except Exception as exc:
            raise KafkaPublishError(
                f"Kafka publish failed for event_type={event_type!r}."
            ) from exc

        if remaining != 0:
            raise KafkaPublishError(
                "Kafka delivery was not completed before the flush timeout."
            )
        if not callback_called:
            raise KafkaPublishError(
                "Kafka producer returned without delivery confirmation."
            )
        if delivery_error is not None:
            raise KafkaPublishError(
                "Kafka broker rejected event delivery: "
                f"{delivery_error}"
            )

        receipt = KafkaDeliveryReceipt(
            event_id=event.event_id,
            topic=topic,
            partition=self._message_number(
                delivery_message,
                "partition",
            ),
            offset=self._message_number(
                delivery_message,
                "offset",
            ),
        )
        self._events.append(event)
        self._sequence_by_run[run_id] = event.sequence + 1
        self._receipts[event.event_id] = receipt
        return event

    def list_events(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[EventEnvelope, ...]:
        if run_id is None:
            return tuple(self._events)
        return tuple(
            event for event in self._events if event.run_id == run_id
        )

    def receipt_for(
        self,
        event_id: str,
    ) -> KafkaDeliveryReceipt | None:
        return self._receipts.get(event_id)

    def _build_event(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        return EventEnvelope(
            event_id=str(uuid4()),
            event_type=event_type,
            run_id=run_id,
            sequence=self._sequence_by_run.get(run_id, 0),
            timestamp=datetime.now(UTC).isoformat(),
            schema_version="1.0",
            payload=payload,
        )

    @staticmethod
    def _message_number(
        message: Any,
        attribute_name: str,
    ) -> int | None:
        if message is None:
            return None
        attribute = getattr(message, attribute_name, None)
        value = attribute() if callable(attribute) else attribute
        return value if isinstance(value, int) else None


ProducerFactory = Callable[[dict[str, Any]], ProducerProtocol]


def build_kafka_event_publisher(
    config: KafkaTransportConfig,
    *,
    producer_factory: ProducerFactory | None = None,
) -> KafkaEventPublisher:
    if producer_factory is None:
        try:
            module = importlib.import_module("confluent_kafka")
        except ImportError as exc:
            raise RuntimeError(
                "Kafka transport requires the optional "
                "'confluent-kafka' package."
            ) from exc
        producer_type = module.Producer
        producer = cast(
            ProducerProtocol,
            producer_type(config.producer_config()),
        )
    else:
        producer = producer_factory(config.producer_config())

    return KafkaEventPublisher(
        producer,
        router=EventTopicRouter(config.topic_prefix),
        flush_timeout_seconds=config.flush_timeout_seconds,
    )


def build_event_publisher_from_env(
    env: Mapping[str, str] | None = None,
    *,
    producer_factory: ProducerFactory | None = None,
) -> EventPublisher:
    values = os.environ if env is None else env
    transport = values.get("EVENT_TRANSPORT", "memory").strip().lower()

    if transport == "memory":
        return InMemoryEventPublisher()
    if transport == "jsonl":
        output_path = Path(
            values.get(
                "EVENT_JSONL_PATH",
                "artifacts/event_transport/events.jsonl",
            )
        )
        return JsonlEventPublisher(output_path)
    if transport in {"kafka", "redpanda"}:
        return build_kafka_event_publisher(
            KafkaTransportConfig.from_env(values),
            producer_factory=producer_factory,
        )

    raise ValueError(
        "EVENT_TRANSPORT must be one of: memory, jsonl, kafka, redpanda."
    )
