"""Kafka-compatible publishing and event-stream metrics."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
)

DEFAULT_EVENT_TOPIC = "enterprise-control-plane-events"


class AsyncProducer(Protocol):
    """Small producer surface used for broker-independent tests."""

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        key: bytes,
        headers: list[tuple[str, bytes]],
    ) -> object:
        ...


class KafkaEventPublisher:
    """Publish versioned events with episode-key ordering."""

    def __init__(
        self,
        *,
        bootstrap_servers: str = "127.0.0.1:19092",
        topic: str = DEFAULT_EVENT_TOPIC,
        client_id: str = "parity-posttrain-control-plane",
        producer: AsyncProducer | None = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.client_id = client_id
        self._producer = producer
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                enable_idempotence=True,
            )
        await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        assert self._producer is not None
        await self._producer.stop()
        self._started = False

    async def publish(
        self,
        event: ControlPlaneEvent,
    ) -> None:
        if not self._started or self._producer is None:
            raise RuntimeError(
                "KafkaEventPublisher.start() must be called "
                "before publish()"
            )
        encoded = json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        await self._producer.send_and_wait(
            self.topic,
            encoded,
            key=event.episode_id.encode("utf-8"),
            headers=[
                (
                    "schema_version",
                    event.schema_version.encode("utf-8"),
                )
            ],
        )


class EventStreamMetrics:
    """Validate ordering and aggregate operational events."""

    def __init__(self) -> None:
        self.events_seen = 0
        self.completed_episodes = 0
        self.guard_rejections = 0
        self.tool_failures = 0
        self.sequence_gaps = 0
        self._last_sequence: dict[str, int] = {}

    def observe(
        self,
        event: ControlPlaneEvent,
    ) -> None:
        expected = (
            self._last_sequence.get(event.episode_id, -1)
            + 1
        )
        if event.sequence != expected:
            self.sequence_gaps += 1
        self._last_sequence[event.episode_id] = (
            event.sequence
        )
        self.events_seen += 1

        if event.event_type == "episode.completed":
            self.completed_episodes += 1
        elif event.event_type == "guard.rejected":
            self.guard_rejections += 1
        elif event.event_type == "tool.failed":
            self.tool_failures += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "events_seen": self.events_seen,
            "completed_episodes": (
                self.completed_episodes
            ),
            "guard_rejections": self.guard_rejections,
            "tool_failures": self.tool_failures,
            "sequence_gaps": self.sequence_gaps,
        }


async def position_consumer_at_end(
    consumer: AIOKafkaConsumer,
    *,
    timeout: float,
) -> None:
    """Complete assignment before publishing a bounded demo run."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not consumer.assignment():
        if loop.time() >= deadline:
            raise TimeoutError(
                "Kafka consumer did not receive a partition "
                "assignment"
            )
        await asyncio.sleep(0.05)
    await consumer.seek_to_end(*consumer.assignment())
