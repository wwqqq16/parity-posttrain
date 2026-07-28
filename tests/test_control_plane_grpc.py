"""gRPC transport and cross-transport parity tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import grpc
import pytest
from fastapi.testclient import TestClient

from enterprise_eval.distributed.api import create_app
from enterprise_eval.distributed.grpc_service import (
    create_grpc_server,
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


async def _with_stub() -> tuple[
    grpc.aio.Server,
    grpc.aio.Channel,
    control_plane_pb2_grpc.RolloutServiceStub,
]:
    server, port = create_grpc_server()
    await server.start()
    channel = grpc.aio.insecure_channel(
        f"127.0.0.1:{port}"
    )
    stub = control_plane_pb2_grpc.RolloutServiceStub(
        channel
    )
    return server, channel, stub


def test_grpc_reset_and_step_are_idempotent() -> None:
    async def scenario() -> tuple[str, str]:
        server, channel, stub = await _with_stub()
        try:
            await stub.Reset(
                control_plane_pb2.ResetRequest(
                    case_id="eligible_standard",
                    episode_id="grpc-episode",
                )
            )
            request = control_plane_pb2.StepRequest(
                episode_id="grpc-episode",
                request_id="step-1",
                action_type="get_order",
                arguments={"order_id": "ORD-1001"},
            )
            first = await stub.Step(request)
            duplicate = await stub.Step(request)
            return (
                first.state_fingerprint,
                duplicate.state_fingerprint,
            )
        finally:
            await channel.close()
            await server.stop(grace=0)

    first, duplicate = asyncio.run(scenario())

    assert first == duplicate


def test_grpc_idempotency_conflict_has_stable_code() -> None:
    async def scenario() -> grpc.StatusCode:
        server, channel, stub = await _with_stub()
        try:
            await stub.Reset(
                control_plane_pb2.ResetRequest(
                    case_id="eligible_standard",
                    episode_id="grpc-episode",
                )
            )
            await stub.Step(
                control_plane_pb2.StepRequest(
                    episode_id="grpc-episode",
                    request_id="same-request",
                    action_type="get_order",
                    arguments={"order_id": "ORD-1001"},
                )
            )
            with pytest.raises(
                grpc.aio.AioRpcError
            ) as captured:
                await stub.Step(
                    control_plane_pb2.StepRequest(
                        episode_id="grpc-episode",
                        request_id="same-request",
                        action_type="check_refund_policy",
                        arguments={
                            "order_id": "ORD-1001"
                        },
                    )
                )
            return captured.value.code()
        finally:
            await channel.close()
            await server.stop(grace=0)

    assert (
        asyncio.run(scenario())
        is grpc.StatusCode.ALREADY_EXISTS
    )


def test_bidirectional_stream_runs_ordered_steps() -> None:
    async def scenario() -> list[str]:
        server, channel, stub = await _with_stub()
        try:
            await stub.Reset(
                control_plane_pb2.ResetRequest(
                    case_id="eligible_standard",
                    episode_id="stream-episode",
                )
            )

            async def requests() -> AsyncIterator[
                control_plane_pb2.StepRequest
            ]:
                for index, (
                    action_type,
                    arguments,
                ) in enumerate(_ACTIONS[:2]):
                    yield control_plane_pb2.StepRequest(
                        episode_id="stream-episode",
                        request_id=f"stream-{index}",
                        action_type=action_type,
                        arguments=arguments,
                    )

            return [
                response.action_type
                async for response in stub.RunEpisode(
                    requests()
                )
            ]
        finally:
            await channel.close()
            await server.stop(grace=0)

    assert asyncio.run(scenario()) == [
        "get_order",
        "check_refund_policy",
    ]


def test_rest_and_grpc_have_final_fingerprint_parity() -> None:
    rest_client = TestClient(create_app())
    with rest_client:
        rest_client.post(
            "/v1/episodes",
            json={
                "case_id": "eligible_standard",
                "episode_id": "rest-parity",
            },
        )
        rest_fingerprint = ""
        for index, (action_type, arguments) in enumerate(
            _ACTIONS
        ):
            response = rest_client.post(
                "/v1/episodes/rest-parity/steps",
                json={
                    "request_id": f"rest-{index}",
                    "action_type": action_type,
                    "arguments": arguments,
                },
            )
            assert response.status_code == 200
            rest_fingerprint = response.json()[
                "state_fingerprint"
            ]

    async def grpc_episode() -> str:
        server, channel, stub = await _with_stub()
        try:
            await stub.Reset(
                control_plane_pb2.ResetRequest(
                    case_id="eligible_standard",
                    episode_id="grpc-parity",
                )
            )
            fingerprint = ""
            for index, (
                action_type,
                arguments,
            ) in enumerate(_ACTIONS):
                response = await stub.Step(
                    control_plane_pb2.StepRequest(
                        episode_id="grpc-parity",
                        request_id=f"grpc-{index}",
                        action_type=action_type,
                        arguments=arguments,
                    )
                )
                fingerprint = response.state_fingerprint
            return fingerprint
        finally:
            await channel.close()
            await server.stop(grace=0)

    grpc_fingerprint = asyncio.run(grpc_episode())

    assert rest_fingerprint == grpc_fingerprint
    assert rest_fingerprint.startswith(
        "cc445b8608da8196"
    )
