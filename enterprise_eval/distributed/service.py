"""Shared episode lifecycle and idempotency service."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from enterprise_eval.cases import CASES
from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
    EventPublisher,
    InMemoryEventPublisher,
)
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    Architecture,
)
from enterprise_eval.rl_environment import (
    EnterpriseRefundRLEnvironment,
    RLStepResult,
)


class EpisodeNotFoundError(KeyError):
    """Raised when an episode ID does not exist."""


class IdempotencyConflictError(ValueError):
    """Raised when one request ID is reused for another action."""


@dataclass(frozen=True)
class _CachedRequest:
    action_digest: str
    response: dict[str, Any]


@dataclass
class _EpisodeRecord:
    environment: EnterpriseRefundRLEnvironment
    architecture: Architecture
    next_sequence: int = 0
    requests: dict[str, _CachedRequest] = field(
        default_factory=dict
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EpisodeService:
    """Transport-independent episode management service."""

    def __init__(
        self,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.publisher = publisher or InMemoryEventPublisher()
        self._episodes: dict[str, _EpisodeRecord] = {}

    async def create_episode(
        self,
        *,
        case_id: str,
        architecture: str = Architecture.SINGLE.value,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        if case_id not in CASES:
            raise ValueError(f"unknown case_id: {case_id}")

        try:
            architecture_value = Architecture(architecture)
        except ValueError as error:
            raise ValueError(
                f"unknown architecture: {architecture}"
            ) from error

        resolved_episode_id = episode_id or str(uuid4())
        if resolved_episode_id in self._episodes:
            raise ValueError(
                f"episode already exists: {resolved_episode_id}"
            )

        environment = EnterpriseRefundRLEnvironment(
            CASES[case_id],
            enforce_execution_guard=True,
        )
        reset = environment.reset(
            architecture=architecture_value
        )
        record = _EpisodeRecord(
            environment=environment,
            architecture=architecture_value,
        )
        self._episodes[resolved_episode_id] = record

        await self._publish(
            episode_id=resolved_episode_id,
            record=record,
            event_type="episode.started",
            state_fingerprint=reset.state_fingerprint,
            payload={
                "case_id": case_id,
                "architecture": architecture_value.value,
            },
        )

        return {
            "episode_id": resolved_episode_id,
            "case_id": case_id,
            "architecture": architecture_value.value,
            "observation": reset.observation,
            "state_fingerprint": reset.state_fingerprint,
            "info": copy.deepcopy(reset.info),
        }

    async def step_episode(
        self,
        *,
        episode_id: str,
        request_id: str,
        action_type: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")

        try:
            action_type_value = ActionType(action_type)
        except ValueError as error:
            raise ValueError(
                f"unknown action_type: {action_type}"
            ) from error

        record = self._get_record(episode_id)
        action_digest = _action_digest(
            action_type=action_type_value,
            arguments=arguments,
        )

        async with record.lock:
            cached = record.requests.get(request_id)
            if cached is not None:
                if cached.action_digest != action_digest:
                    raise IdempotencyConflictError(
                        "request_id was already used for "
                        "a different action"
                    )
                return copy.deepcopy(cached.response)

            before_fingerprint = (
                record.environment.state_fingerprint()
            )
            await self._publish(
                episode_id=episode_id,
                record=record,
                event_type="action.requested",
                state_fingerprint=before_fingerprint,
                payload={
                    "request_id": request_id,
                    "action_type": action_type_value.value,
                    "arguments": copy.deepcopy(arguments),
                },
            )

            transition = record.environment.step(
                AgentAction(
                    action_type=action_type_value,
                    arguments=copy.deepcopy(arguments),
                )
            )
            response = _transition_response(
                episode_id=episode_id,
                request_id=request_id,
                transition=transition,
            )

            await self._publish_outcome_event(
                episode_id=episode_id,
                record=record,
                transition=transition,
            )
            await self._publish(
                episode_id=episode_id,
                record=record,
                event_type="reward.assigned",
                state_fingerprint=(
                    transition.state_fingerprint
                ),
                payload={
                    "reward": transition.reward,
                    "components": copy.deepcopy(
                        transition.info[
                            "reward_components"
                        ]
                    ),
                },
            )

            if transition.terminated or transition.truncated:
                await self._publish(
                    episode_id=episode_id,
                    record=record,
                    event_type="episode.completed",
                    state_fingerprint=(
                        transition.state_fingerprint
                    ),
                    payload={
                        "terminated": transition.terminated,
                        "truncated": transition.truncated,
                    },
                )

            record.requests[request_id] = _CachedRequest(
                action_digest=action_digest,
                response=copy.deepcopy(response),
            )
            return response

    async def get_episode(
        self,
        episode_id: str,
    ) -> dict[str, Any]:
        record = self._get_record(episode_id)

        async with record.lock:
            snapshot = record.environment.snapshot()
            run = record.environment.environment.run
            assert run is not None

            return {
                "episode_id": episode_id,
                "case_id": record.environment.case.case_id,
                "architecture": record.architecture.value,
                "step_count": len(run.steps),
                "terminated": (
                    record.environment.environment.state.terminated
                ),
                "truncated": snapshot.truncated,
                "state_fingerprint": (
                    snapshot.state_fingerprint
                ),
                "next_event_sequence": record.next_sequence,
            }

    async def _publish_outcome_event(
        self,
        *,
        episode_id: str,
        record: _EpisodeRecord,
        transition: RLStepResult,
    ) -> None:
        metadata = transition.info["tool_metadata"]
        reward_components = transition.info[
            "reward_components"
        ]
        error_type = metadata.get("error_type")

        event_type: str | None = None
        if error_type == "execution_guard_rejection":
            event_type = "guard.rejected"
        elif error_type == "transient_tool_failure":
            event_type = "tool.failed"
        elif "recovery_credit" in reward_components:
            event_type = "tool.recovered"

        if event_type is not None:
            await self._publish(
                episode_id=episode_id,
                record=record,
                event_type=event_type,
                state_fingerprint=(
                    transition.state_fingerprint
                ),
                payload=copy.deepcopy(metadata),
            )

    async def _publish(
        self,
        *,
        episode_id: str,
        record: _EpisodeRecord,
        event_type: str,
        state_fingerprint: str,
        payload: dict[str, Any],
    ) -> None:
        event = ControlPlaneEvent.create(
            episode_id=episode_id,
            sequence=record.next_sequence,
            event_type=event_type,
            state_fingerprint=state_fingerprint,
            payload=payload,
        )
        await self.publisher.publish(event)
        record.next_sequence += 1

    def _get_record(
        self,
        episode_id: str,
    ) -> _EpisodeRecord:
        try:
            return self._episodes[episode_id]
        except KeyError as error:
            raise EpisodeNotFoundError(
                f"unknown episode_id: {episode_id}"
            ) from error


def _action_digest(
    *,
    action_type: ActionType,
    arguments: dict[str, Any],
) -> str:
    payload = {
        "action_type": action_type.value,
        "arguments": arguments,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transition_response(
    *,
    episode_id: str,
    request_id: str,
    transition: RLStepResult,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "request_id": request_id,
        "action_type": transition.info["action_type"],
        "observation": transition.observation,
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "state_fingerprint": (
            transition.state_fingerprint
        ),
        "info": copy.deepcopy(transition.info),
    }
