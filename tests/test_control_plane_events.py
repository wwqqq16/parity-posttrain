"""Tests for versioned control-plane events."""

from __future__ import annotations

import asyncio

from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
    InMemoryEventPublisher,
)


def test_in_memory_publisher_preserves_episode_order() -> None:
    publisher = InMemoryEventPublisher()

    first = ControlPlaneEvent.create(
        episode_id="episode-1",
        sequence=0,
        event_type="episode.started",
        state_fingerprint="initial",
    )
    second = ControlPlaneEvent.create(
        episode_id="episode-1",
        sequence=1,
        event_type="action.requested",
        state_fingerprint="initial",
        payload={"action_type": "get_order"},
    )

    asyncio.run(publisher.publish(first))
    asyncio.run(publisher.publish(second))

    events = publisher.events_for_episode("episode-1")

    assert [event.sequence for event in events] == [0, 1]
    assert [event.event_type for event in events] == [
        "episode.started",
        "action.requested",
    ]
    assert events[1].payload == {
        "action_type": "get_order"
    }


def test_event_has_version_and_unique_identity() -> None:
    first = ControlPlaneEvent.create(
        episode_id="episode-1",
        sequence=0,
        event_type="episode.started",
        state_fingerprint="abc",
    )
    second = ControlPlaneEvent.create(
        episode_id="episode-1",
        sequence=0,
        event_type="episode.started",
        state_fingerprint="abc",
    )

    assert first.schema_version == "control-plane-event.v1"
    assert first.event_id != second.event_id
    assert first.to_dict()["episode_id"] == "episode-1"
