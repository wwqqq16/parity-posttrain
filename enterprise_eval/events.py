from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_type: str
    run_id: str
    sequence: int
    timestamp: str
    schema_version: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventPublisher(Protocol):
    def publish(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: dict[str, Any],
    ) -> EventEnvelope: ...

    def list_events(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[EventEnvelope, ...]: ...


class InMemoryEventPublisher:
    """Kafka-compatible event contract with an in-memory local sink."""

    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._sequence_by_run: dict[str, int] = {}

    def publish(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        sequence = self._sequence_by_run.get(run_id, 0)
        event = EventEnvelope(
            event_id=str(uuid4()),
            event_type=event_type,
            run_id=run_id,
            sequence=sequence,
            timestamp=datetime.now(UTC).isoformat(),
            schema_version="1.0",
            payload=payload,
        )
        self._sequence_by_run[run_id] = sequence + 1
        self._events.append(event)
        return event

    def list_events(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[EventEnvelope, ...]:
        if run_id is None:
            return tuple(self._events)
        return tuple(event for event in self._events if event.run_id == run_id)


class JsonlEventPublisher(InMemoryEventPublisher):
    """Append-only JSONL sink using the same event envelope as Kafka."""

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path

    def publish(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        event = super().publish(
            event_type,
            run_id=run_id,
            payload=payload,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return event
