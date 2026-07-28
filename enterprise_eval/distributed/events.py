"""Versioned control-plane events and publisher interfaces."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class ControlPlaneEvent:
    """One ordered and auditable episode event."""

    schema_version: str
    event_id: str
    episode_id: str
    sequence: int
    event_type: str
    state_fingerprint: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        sequence: int,
        event_type: str,
        state_fingerprint: str,
        payload: dict[str, Any] | None = None,
    ) -> ControlPlaneEvent:
        return cls(
            schema_version="control-plane-event.v1",
            event_id=str(uuid4()),
            episode_id=episode_id,
            sequence=sequence,
            event_type=event_type,
            state_fingerprint=state_fingerprint,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> ControlPlaneEvent:
        """Parse an event received from an external transport."""

        return cls(
            schema_version=str(value["schema_version"]),
            event_id=str(value["event_id"]),
            episode_id=str(value["episode_id"]),
            sequence=int(value["sequence"]),
            event_type=str(value["event_type"]),
            state_fingerprint=str(
                value["state_fingerprint"]
            ),
            payload=copy.deepcopy(value["payload"]),
        )


class EventPublisher(Protocol):
    """Transport-independent event publishing contract."""

    async def publish(
        self,
        event: ControlPlaneEvent,
    ) -> None:
        ...


class InMemoryEventPublisher:
    """Deterministic test publisher used before Kafka is connected."""

    def __init__(self) -> None:
        self._events: list[ControlPlaneEvent] = []

    async def publish(
        self,
        event: ControlPlaneEvent,
    ) -> None:
        self._events.append(copy.deepcopy(event))

    @property
    def events(self) -> tuple[ControlPlaneEvent, ...]:
        return tuple(copy.deepcopy(self._events))

    def events_for_episode(
        self,
        episode_id: str,
    ) -> tuple[ControlPlaneEvent, ...]:
        return tuple(
            copy.deepcopy(event)
            for event in self._events
            if event.episode_id == episode_id
        )
