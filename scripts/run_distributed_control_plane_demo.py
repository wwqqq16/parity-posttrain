# ruff: noqa: E402
"""Verify REST, gRPC, Kafka, replay, idempotency, and guarding."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import grpc
import httpx
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
    position_consumer_at_end,
)
from enterprise_eval.distributed.proto import (
    control_plane_pb2,
    control_plane_pb2_grpc,
)

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
        description=(
            "Run the complete distributed control-plane demo."
        )
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--grpc-target",
        default="127.0.0.1:50051",
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
        default=15.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/distributed_control_plane/demo.json"
        ),
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    run_suffix = str(uuid4())
    rest_episode_id = f"rest-{run_suffix}"
    grpc_episode_id = f"grpc-{run_suffix}"
    guard_episode_id = f"guard-{run_suffix}"

    consumer = AIOKafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=f"distributed-demo-{run_suffix}",
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )
    await consumer.start()

    channel = grpc.aio.insecure_channel(args.grpc_target)
    try:
        await _wait_for_http(args.api_url, args.timeout)
        await asyncio.wait_for(
            channel.channel_ready(),
            timeout=args.timeout,
        )
        await position_consumer_at_end(
            consumer,
            timeout=args.timeout,
        )

        async with httpx.AsyncClient(
            base_url=args.api_url,
            timeout=args.timeout,
        ) as client:
            rest_result = await _run_rest_episode(
                client=client,
                episode_id=rest_episode_id,
            )
            guard_result = await _run_guard_probe(
                client=client,
                episode_id=guard_episode_id,
            )

        stub = control_plane_pb2_grpc.RolloutServiceStub(
            channel
        )
        grpc_fingerprint = await _run_grpc_episode(
            stub=stub,
            episode_id=grpc_episode_id,
        )
        event_groups = await _consume_targets(
            consumer=consumer,
            rest_episode_id=rest_episode_id,
            grpc_episode_id=grpc_episode_id,
            guard_episode_id=guard_episode_id,
            timeout=args.timeout,
        )
    finally:
        await channel.close()
        await consumer.stop()

    rest_events = event_groups[rest_episode_id]
    metrics = EventStreamMetrics()
    for event in rest_events:
        metrics.observe(event)

    replayed_fingerprint = replay_fingerprint(rest_events)
    transport_parity = (
        rest_result["final_fingerprint"]
        == grpc_fingerprint
    )
    replay_parity = (
        replayed_fingerprint
        == rest_result["final_fingerprint"]
    )
    guard_event_emitted = any(
        event.event_type == "guard.rejected"
        for event in event_groups[guard_episode_id]
    )
    duplicate_refund_blocked = bool(
        rest_result["duplicate_refund_idempotent"]
    )
    guard_safe = (
        guard_result == "execution_guard_rejection"
        and guard_event_emitted
    )

    payload = {
        "schema_version": (
            "distributed-control-plane-demo.v1"
        ),
        "rest_final_fingerprint": rest_result[
            "final_fingerprint"
        ],
        "grpc_final_fingerprint": grpc_fingerprint,
        "transport_parity": transport_parity,
        "duplicate_refund_blocked": (
            duplicate_refund_blocked
        ),
        "kafka_metrics": metrics.to_dict(),
        "replay_fingerprint": replayed_fingerprint,
        "replay_parity": replay_parity,
        "guard_rejection_emitted": guard_safe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Distributed enterprise control-plane demo")
    print("=" * 72)
    print(
        "REST final fingerprint:  ",
        str(rest_result["final_fingerprint"])[:16],
    )
    print(
        "gRPC final fingerprint:  ",
        grpc_fingerprint[:16],
    )
    print(
        "Transport parity:        ",
        _status(transport_parity),
    )
    print(
        "Duplicate refund blocked:",
        _status(duplicate_refund_blocked),
    )
    print(
        "Kafka events observed:   ",
        metrics.events_seen,
    )
    print(
        "Event sequence gaps:      ",
        metrics.sequence_gaps,
    )
    print(
        "Replay parity:           ",
        _status(replay_parity),
    )
    print(
        "Guard rejection emitted: ",
        _status(guard_safe),
    )
    print("Artifact:", args.output)

    passed = (
        transport_parity
        and duplicate_refund_blocked
        and metrics.events_seen == 12
        and metrics.sequence_gaps == 0
        and replay_parity
        and guard_safe
    )
    return 0 if passed else 1


async def _run_rest_episode(
    *,
    client: httpx.AsyncClient,
    episode_id: str,
) -> dict[str, Any]:
    created = await client.post(
        "/v1/episodes",
        json={
            "case_id": "eligible_standard",
            "episode_id": episode_id,
        },
    )
    created.raise_for_status()

    final_fingerprint = ""
    duplicate_refund_idempotent = False
    for index, (action_type, arguments) in enumerate(
        _ACTIONS
    ):
        request = {
            "request_id": f"rest-step-{index}",
            "action_type": action_type,
            "arguments": arguments,
        }
        response = await client.post(
            f"/v1/episodes/{episode_id}/steps",
            json=request,
        )
        response.raise_for_status()
        transition = response.json()
        final_fingerprint = str(
            transition["state_fingerprint"]
        )

        if action_type == "issue_refund":
            duplicate = await client.post(
                f"/v1/episodes/{episode_id}/steps",
                json=request,
            )
            duplicate.raise_for_status()
            duplicate_refund_idempotent = (
                duplicate.json() == transition
            )

    state = await client.get(
        f"/v1/episodes/{episode_id}"
    )
    state.raise_for_status()
    duplicate_refund_idempotent = (
        duplicate_refund_idempotent
        and state.json()["step_count"] == len(_ACTIONS)
    )
    return {
        "final_fingerprint": final_fingerprint,
        "duplicate_refund_idempotent": (
            duplicate_refund_idempotent
        ),
    }


async def _run_grpc_episode(
    *,
    stub: control_plane_pb2_grpc.RolloutServiceStub,
    episode_id: str,
) -> str:
    await stub.Reset(
        control_plane_pb2.ResetRequest(
            case_id="eligible_standard",
            episode_id=episode_id,
        )
    )
    final_fingerprint = ""
    for index, (action_type, arguments) in enumerate(
        _ACTIONS
    ):
        response = await stub.Step(
            control_plane_pb2.StepRequest(
                episode_id=episode_id,
                request_id=f"grpc-step-{index}",
                action_type=action_type,
                arguments=arguments,
            )
        )
        final_fingerprint = response.state_fingerprint
    return final_fingerprint


async def _run_guard_probe(
    *,
    client: httpx.AsyncClient,
    episode_id: str,
) -> str:
    created = await client.post(
        "/v1/episodes",
        json={
            "case_id": "eligible_standard",
            "episode_id": episode_id,
        },
    )
    created.raise_for_status()
    response = await client.post(
        f"/v1/episodes/{episode_id}/steps",
        json={
            "request_id": "unsafe-refund",
            "action_type": "issue_refund",
            "arguments": {"order_id": "ORD-1001"},
        },
    )
    response.raise_for_status()
    return str(
        response.json()["info"]["tool_metadata"][
            "error_type"
        ]
    )


async def _consume_targets(
    *,
    consumer: AIOKafkaConsumer,
    rest_episode_id: str,
    grpc_episode_id: str,
    guard_episode_id: str,
    timeout: float,
) -> dict[str, list[ControlPlaneEvent]]:
    groups = {
        rest_episode_id: [],
        grpc_episode_id: [],
        guard_episode_id: [],
    }
    while not _target_events_complete(
        groups=groups,
        rest_episode_id=rest_episode_id,
        grpc_episode_id=grpc_episode_id,
        guard_episode_id=guard_episode_id,
    ):
        message = await asyncio.wait_for(
            consumer.getone(),
            timeout=timeout,
        )
        event = ControlPlaneEvent.from_dict(
            json.loads(message.value)
        )
        if event.episode_id in groups:
            groups[event.episode_id].append(event)
    for events in groups.values():
        events.sort(key=lambda event: event.sequence)
    return groups


def _target_events_complete(
    *,
    groups: dict[str, list[ControlPlaneEvent]],
    rest_episode_id: str,
    grpc_episode_id: str,
    guard_episode_id: str,
) -> bool:
    return (
        any(
            event.event_type == "episode.completed"
            for event in groups[rest_episode_id]
        )
        and any(
            event.event_type == "episode.completed"
            for event in groups[grpc_episode_id]
        )
        and any(
            event.event_type == "guard.rejected"
            for event in groups[guard_episode_id]
        )
    )


async def _wait_for_http(
    api_url: str,
    timeout: float,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            try:
                response = await client.get(
                    f"{api_url}/healthz"
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if loop.time() >= deadline:
                raise TimeoutError(
                    "FastAPI service did not become ready"
                )
            await asyncio.sleep(0.1)


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
