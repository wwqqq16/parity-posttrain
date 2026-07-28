"""Tests for shared distributed episode management."""

from __future__ import annotations

import asyncio

import pytest

from enterprise_eval.distributed.events import (
    InMemoryEventPublisher,
)
from enterprise_eval.distributed.service import (
    EpisodeService,
    IdempotencyConflictError,
)


def test_duplicate_request_executes_action_once() -> None:
    async def scenario() -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        InMemoryEventPublisher,
    ]:
        publisher = InMemoryEventPublisher()
        service = EpisodeService(publisher)

        await service.create_episode(
            case_id="eligible_standard",
            episode_id="episode-1",
        )
        first = await service.step_episode(
            episode_id="episode-1",
            request_id="request-1",
            action_type="get_order",
            arguments={"order_id": "ORD-1001"},
        )
        duplicate = await service.step_episode(
            episode_id="episode-1",
            request_id="request-1",
            action_type="get_order",
            arguments={"order_id": "ORD-1001"},
        )
        state = await service.get_episode("episode-1")
        return first, duplicate, state, publisher

    first, duplicate, state, publisher = asyncio.run(
        scenario()
    )

    assert first == duplicate
    assert state["step_count"] == 1
    assert [
        event.event_type
        for event in publisher.events
    ] == [
        "episode.started",
        "action.requested",
        "reward.assigned",
    ]
    assert [
        event.sequence
        for event in publisher.events
    ] == [0, 1, 2]


def test_request_id_cannot_be_reused_for_new_action() -> None:
    async def scenario() -> None:
        service = EpisodeService()
        await service.create_episode(
            case_id="eligible_standard",
            episode_id="episode-1",
        )
        await service.step_episode(
            episode_id="episode-1",
            request_id="same-request",
            action_type="get_order",
            arguments={"order_id": "ORD-1001"},
        )

        with pytest.raises(
            IdempotencyConflictError,
            match="different action",
        ):
            await service.step_episode(
                episode_id="episode-1",
                request_id="same-request",
                action_type="check_refund_policy",
                arguments={"order_id": "ORD-1001"},
            )

    asyncio.run(scenario())


def test_unsafe_refund_emits_guard_event() -> None:
    async def scenario() -> tuple[
        dict[str, object],
        InMemoryEventPublisher,
    ]:
        publisher = InMemoryEventPublisher()
        service = EpisodeService(publisher)
        await service.create_episode(
            case_id="eligible_standard",
            episode_id="unsafe-episode",
        )
        transition = await service.step_episode(
            episode_id="unsafe-episode",
            request_id="unsafe-request",
            action_type="issue_refund",
            arguments={"order_id": "ORD-1001"},
        )
        return transition, publisher

    transition, publisher = asyncio.run(scenario())

    assert transition["reward"] == -0.5
    assert transition["info"]["tool_metadata"][
        "error_type"
    ] == "execution_guard_rejection"
    assert [
        event.event_type
        for event in publisher.events
    ] == [
        "episode.started",
        "action.requested",
        "guard.rejected",
        "reward.assigned",
    ]
    assert [
        event.sequence
        for event in publisher.events
    ] == [0, 1, 2, 3]
