"""Tests for Kafka publishing, metrics, and event replay."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from enterprise_eval.distributed.event_replay import (
    replay_fingerprint,
)
from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
    InMemoryEventPublisher,
)
from enterprise_eval.distributed.kafka_events import (
    EventStreamMetrics,
    KafkaEventPublisher,
)
from enterprise_eval.distributed.service import EpisodeService


class _FakeProducer:
    def __init__(self) -> None:
        self.started = False
        self.records: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        key: bytes,
        headers: list[tuple[str, bytes]],
    ) -> object:
        self.records.append(
            {
                "topic": topic,
                "value": value,
                "key": key,
                "headers": headers,
            }
        )
        return object()


def test_kafka_publisher_keys_events_by_episode() -> None:
    async def scenario() -> _FakeProducer:
        producer = _FakeProducer()
        publisher = KafkaEventPublisher(
            producer=producer
        )
        event = ControlPlaneEvent.create(
            episode_id="episode-1",
            sequence=0,
            event_type="episode.started",
            state_fingerprint="abc",
            payload={"case_id": "eligible_standard"},
        )

        await publisher.start()
        await publisher.publish(event)
        await publisher.stop()
        return producer

    producer = asyncio.run(scenario())
    record = producer.records[0]
    decoded = json.loads(record["value"])

    assert record["key"] == b"episode-1"
    assert decoded["event_type"] == "episode.started"
    assert record["headers"] == [
        (
            "schema_version",
            b"control-plane-event.v1",
        )
    ]
    assert not producer.started


def test_metrics_detect_sequence_gaps() -> None:
    metrics = EventStreamMetrics()
    for sequence in (0, 2):
        metrics.observe(
            ControlPlaneEvent.create(
                episode_id="episode-1",
                sequence=sequence,
                event_type="reward.assigned",
                state_fingerprint=str(sequence),
            )
        )

    assert metrics.events_seen == 2
    assert metrics.sequence_gaps == 1


def test_event_stream_replays_to_original_fingerprint() -> None:
    async def scenario() -> tuple[
        tuple[ControlPlaneEvent, ...],
        str,
    ]:
        publisher = InMemoryEventPublisher()
        service = EpisodeService(publisher)
        await service.create_episode(
            case_id="eligible_standard",
            episode_id="replay-episode",
        )
        actions = (
            ("get_order", {"order_id": "ORD-1001"}),
            (
                "check_refund_policy",
                {"order_id": "ORD-1001"},
            ),
            (
                "get_payment_status",
                {"order_id": "ORD-1001"},
            ),
            (
                "issue_refund",
                {"order_id": "ORD-1001"},
            ),
            (
                "respond",
                {
                    "message": (
                        "Your refund has been issued."
                    )
                },
            ),
        )
        transition: dict[str, Any] = {}
        for index, (
            action_type,
            arguments,
        ) in enumerate(actions):
            transition = await service.step_episode(
                episode_id="replay-episode",
                request_id=f"step-{index}",
                action_type=action_type,
                arguments=arguments,
            )
        return publisher.events, str(
            transition["state_fingerprint"]
        )

    events, original = asyncio.run(scenario())

    assert len(events) == 12
    assert replay_fingerprint(events) == original
    assert original.startswith("cc445b8608da8196")
