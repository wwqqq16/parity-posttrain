# ruff: noqa: E402
"""Publish, consume, validate, and replay one real Kafka event stream."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from aiokafka import AIOKafkaConsumer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.distributed.event_replay import (
    replay_fingerprint,
)
from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
)
from enterprise_eval.distributed.kafka_events import (
    DEFAULT_EVENT_TOPIC,
    EventStreamMetrics,
    KafkaEventPublisher,
    position_consumer_at_end,
)
from enterprise_eval.distributed.service import EpisodeService

_ACTIONS = (
    ("get_order", {"order_id": "ORD-1001"}),
    ("check_refund_policy", {"order_id": "ORD-1001"}),
    ("get_payment_status", {"order_id": "ORD-1001"}),
    ("issue_refund", {"order_id": "ORD-1001"}),
    (
        "respond",
        {"message": "Your refund has been issued."},
    ),
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Kafka-compatible event demo."
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="127.0.0.1:19092",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_EVENT_TOPIC,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
    )
    return parser.parse_args(argv)


async def run(
    *,
    bootstrap_servers: str,
    topic: str,
    timeout: float,
) -> int:
    episode_id = f"kafka-demo-{uuid4()}"
    group_id = f"kafka-demo-consumer-{uuid4()}"
    publisher = KafkaEventPublisher(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
    )
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )

    await publisher.start()
    await consumer.start()
    try:
        await position_consumer_at_end(
            consumer,
            timeout=timeout,
        )
        service = EpisodeService(publisher)
        await service.create_episode(
            case_id="eligible_standard",
            episode_id=episode_id,
        )
        final_fingerprint = ""
        for index, (
            action_type,
            arguments,
        ) in enumerate(_ACTIONS):
            transition = await service.step_episode(
                episode_id=episode_id,
                request_id=f"kafka-step-{index}",
                action_type=action_type,
                arguments=arguments,
            )
            final_fingerprint = str(
                transition["state_fingerprint"]
            )

        events = await _consume_episode(
            consumer=consumer,
            episode_id=episode_id,
            timeout=timeout,
        )
    finally:
        await consumer.stop()
        await publisher.stop()

    metrics = EventStreamMetrics()
    for event in events:
        metrics.observe(event)
    replayed_fingerprint = replay_fingerprint(events)
    replay_identical = (
        replayed_fingerprint == final_fingerprint
    )

    print("Enterprise Kafka-compatible event demo")
    print("=" * 72)
    print("Broker:", bootstrap_servers)
    print("Topic:", topic)
    print("Events observed:", metrics.events_seen)
    print("Completed episodes:", metrics.completed_episodes)
    print("Sequence gaps:", metrics.sequence_gaps)
    print(
        "Final fingerprint:",
        final_fingerprint[:16],
    )
    print(
        "Replay fingerprint:",
        replayed_fingerprint[:16],
    )
    print("Replay parity:", replay_identical)

    passed = (
        metrics.events_seen == 12
        and metrics.completed_episodes == 1
        and metrics.sequence_gaps == 0
        and replay_identical
    )
    return 0 if passed else 1


async def _consume_episode(
    *,
    consumer: AIOKafkaConsumer,
    episode_id: str,
    timeout: float,
) -> list[ControlPlaneEvent]:
    events: list[ControlPlaneEvent] = []
    while not any(
        event.event_type == "episode.completed"
        for event in events
    ):
        message = await asyncio.wait_for(
            consumer.getone(),
            timeout=timeout,
        )
        decoded = json.loads(message.value)
        event = ControlPlaneEvent.from_dict(decoded)
        if event.episode_id == episode_id:
            events.append(event)
    return sorted(events, key=lambda event: event.sequence)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    return asyncio.run(
        run(
            bootstrap_servers=args.bootstrap_servers,
            topic=args.topic,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
