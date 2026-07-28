"""HTTP contract tests for the enterprise control plane."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterprise_eval.distributed.api import create_app
from enterprise_eval.distributed.events import (
    InMemoryEventPublisher,
)
from enterprise_eval.distributed.service import EpisodeService


def _client() -> tuple[TestClient, InMemoryEventPublisher]:
    publisher = InMemoryEventPublisher()
    service = EpisodeService(publisher)
    return TestClient(create_app(service)), publisher


def test_health_create_and_get_episode() -> None:
    client, _publisher = _client()

    with client:
        health = client.get("/healthz")
        created = client.post(
            "/v1/episodes",
            json={
                "case_id": "eligible_standard",
                "episode_id": "http-episode",
            },
        )
        state = client.get(
            "/v1/episodes/http-episode"
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert created.status_code == 201
    assert created.json()["episode_id"] == "http-episode"
    assert state.status_code == 200
    assert state.json()["step_count"] == 0
    assert (
        created.json()["state_fingerprint"]
        == state.json()["state_fingerprint"]
    )


def test_http_idempotency_prevents_duplicate_execution() -> None:
    client, publisher = _client()
    request = {
        "request_id": "request-1",
        "action_type": "get_order",
        "arguments": {"order_id": "ORD-1001"},
    }

    with client:
        client.post(
            "/v1/episodes",
            json={
                "case_id": "eligible_standard",
                "episode_id": "http-episode",
            },
        )
        first = client.post(
            "/v1/episodes/http-episode/steps",
            json=request,
        )
        duplicate = client.post(
            "/v1/episodes/http-episode/steps",
            json=request,
        )
        state = client.get(
            "/v1/episodes/http-episode"
        )

    assert first.status_code == 200
    assert first.json() == duplicate.json()
    assert state.json()["step_count"] == 1
    assert [
        event.event_type for event in publisher.events
    ] == [
        "episode.started",
        "action.requested",
        "reward.assigned",
    ]


def test_http_maps_domain_errors_to_stable_status_codes() -> None:
    client, _publisher = _client()

    with client:
        missing = client.get("/v1/episodes/missing")
        invalid_case = client.post(
            "/v1/episodes",
            json={"case_id": "not-a-case"},
        )
        client.post(
            "/v1/episodes",
            json={
                "case_id": "eligible_standard",
                "episode_id": "http-episode",
            },
        )
        client.post(
            "/v1/episodes/http-episode/steps",
            json={
                "request_id": "same-request",
                "action_type": "get_order",
                "arguments": {"order_id": "ORD-1001"},
            },
        )
        conflict = client.post(
            "/v1/episodes/http-episode/steps",
            json={
                "request_id": "same-request",
                "action_type": "check_refund_policy",
                "arguments": {"order_id": "ORD-1001"},
            },
        )

    assert missing.status_code == 404
    assert invalid_case.status_code == 400
    assert conflict.status_code == 409
    assert "different action" in conflict.json()["detail"]


def test_http_guard_rejection_does_not_issue_refund() -> None:
    client, publisher = _client()

    with client:
        client.post(
            "/v1/episodes",
            json={
                "case_id": "eligible_standard",
                "episode_id": "guarded-episode",
            },
        )
        response = client.post(
            "/v1/episodes/guarded-episode/steps",
            json={
                "request_id": "unsafe-request",
                "action_type": "issue_refund",
                "arguments": {"order_id": "ORD-1001"},
            },
        )

    assert response.status_code == 200
    assert response.json()["reward"] == -0.5
    assert response.json()["info"]["tool_metadata"][
        "error_type"
    ] == "execution_guard_rejection"
    assert "guard.rejected" in [
        event.event_type for event in publisher.events
    ]
