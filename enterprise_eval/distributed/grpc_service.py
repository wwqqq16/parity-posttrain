"""Asynchronous gRPC adapter for enterprise rollout workers."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from enterprise_eval.distributed.kafka_events import (
    KafkaEventPublisher,
)
from enterprise_eval.distributed.proto import (
    control_plane_pb2,
    control_plane_pb2_grpc,
)
from enterprise_eval.distributed.service import (
    EpisodeNotFoundError,
    EpisodeService,
    IdempotencyConflictError,
)


class RolloutService(
    control_plane_pb2_grpc.RolloutServiceServicer
):
    """Map protobuf requests onto the shared episode service."""

    def __init__(
        self,
        service: EpisodeService | None = None,
    ) -> None:
        self.service = service or EpisodeService()

    async def Reset(
        self,
        request: control_plane_pb2.ResetRequest,
        context: grpc.aio.ServicerContext[
            control_plane_pb2.ResetRequest,
            control_plane_pb2.ResetResponse,
        ],
    ) -> control_plane_pb2.ResetResponse:
        try:
            result = await self.service.create_episode(
                case_id=request.case_id,
                architecture=request.architecture or "single",
                episode_id=request.episode_id or None,
            )
        except ValueError as error:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                _error_message(error),
            )

        return control_plane_pb2.ResetResponse(
            episode_id=result["episode_id"],
            case_id=result["case_id"],
            architecture=result["architecture"],
            observation=result["observation"],
            state_fingerprint=result["state_fingerprint"],
            info=_to_struct(result["info"]),
        )

    async def Step(
        self,
        request: control_plane_pb2.StepRequest,
        context: grpc.aio.ServicerContext[
            control_plane_pb2.StepRequest,
            control_plane_pb2.StepResponse,
        ],
    ) -> control_plane_pb2.StepResponse:
        try:
            result = await self.service.step_episode(
                episode_id=request.episode_id,
                request_id=request.request_id,
                action_type=request.action_type,
                arguments=MessageToDict(
                    request.arguments,
                    preserving_proto_field_name=True,
                ),
            )
        except EpisodeNotFoundError as error:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                _error_message(error),
            )
        except IdempotencyConflictError as error:
            await context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                _error_message(error),
            )
        except ValueError as error:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                _error_message(error),
            )

        return _step_response(result)

    async def RunEpisode(
        self,
        request_iterator: AsyncIterator[
            control_plane_pb2.StepRequest
        ],
        context: grpc.aio.ServicerContext[
            control_plane_pb2.StepRequest,
            control_plane_pb2.StepResponse,
        ],
    ) -> AsyncIterator[control_plane_pb2.StepResponse]:
        async for request in request_iterator:
            yield await self.Step(request, context)


def create_grpc_server(
    service: EpisodeService | None = None,
    *,
    address: str = "127.0.0.1:0",
) -> tuple[grpc.aio.Server, int]:
    """Build a server and return its resolved listening port."""

    server = grpc.aio.server()
    control_plane_pb2_grpc.add_RolloutServiceServicer_to_server(
        RolloutService(service),
        server,
    )
    port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError(
            f"gRPC could not bind to {address}"
        )
    return server, port


async def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 50051,
    kafka_bootstrap_servers: str | None = None,
) -> None:
    """Run the rollout worker until interrupted."""

    publisher = (
        KafkaEventPublisher(
            bootstrap_servers=kafka_bootstrap_servers
        )
        if kafka_bootstrap_servers is not None
        else None
    )
    if publisher is not None:
        await publisher.start()
    service = EpisodeService(publisher)
    server, bound_port = create_grpc_server(
        service,
        address=f"{host}:{port}"
    )
    await server.start()
    print(
        "gRPC rollout service listening on "
        f"{host}:{bound_port}"
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=1.0)
        if publisher is not None:
            await publisher.stop()


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the enterprise gRPC rollout service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--kafka-bootstrap-servers",
        default=None,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(
            serve(
                host=args.host,
                port=args.port,
                kafka_bootstrap_servers=(
                    args.kafka_bootstrap_servers
                ),
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


def _step_response(
    result: dict[str, Any],
) -> control_plane_pb2.StepResponse:
    return control_plane_pb2.StepResponse(
        episode_id=result["episode_id"],
        request_id=result["request_id"],
        action_type=result["action_type"],
        observation=result["observation"],
        reward=result["reward"],
        terminated=result["terminated"],
        truncated=result["truncated"],
        state_fingerprint=result["state_fingerprint"],
        info=_to_struct(result["info"]),
    )


def _to_struct(value: dict[str, Any]) -> Struct:
    result = Struct()
    result.update(value)
    return result


def _error_message(error: Exception) -> str:
    if error.args:
        return str(error.args[0])
    return str(error)


if __name__ == "__main__":
    raise SystemExit(main())
