"""FastAPI adapter for the shared enterprise episode service."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from enterprise_eval.distributed.service import (
    EpisodeNotFoundError,
    EpisodeService,
    IdempotencyConflictError,
)
from enterprise_eval.models import Architecture


class CreateEpisodeRequest(BaseModel):
    """Request body for deterministic environment reset."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    architecture: str = Architecture.SINGLE.value
    episode_id: str | None = None


class StepEpisodeRequest(BaseModel):
    """One idempotent action request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    action_type: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def create_app(
    service: EpisodeService | None = None,
    *,
    lifespan: (
        Callable[
            [FastAPI],
            AbstractAsyncContextManager[None],
        ]
        | None
    ) = None,
) -> FastAPI:
    """Create an HTTP adapter around one episode service instance."""

    episode_service = service or EpisodeService()
    app = FastAPI(
        title="ParityPostTrain Enterprise Control Plane",
        version="0.1.0",
        description=(
            "Production-shaped HTTP transport for the deterministic "
            "enterprise RL environment."
        ),
        lifespan=lifespan,
    )
    app.state.episode_service = episode_service

    @app.exception_handler(EpisodeNotFoundError)
    async def handle_episode_not_found(
        _request: Request,
        error: EpisodeNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": _error_message(error)},
        )

    @app.exception_handler(IdempotencyConflictError)
    async def handle_idempotency_conflict(
        _request: Request,
        error: IdempotencyConflictError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": _error_message(error)},
        )

    @app.exception_handler(ValueError)
    async def handle_invalid_request(
        _request: Request,
        error: ValueError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": _error_message(error)},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/episodes",
        status_code=status.HTTP_201_CREATED,
    )
    async def create_episode(
        body: CreateEpisodeRequest,
    ) -> dict[str, Any]:
        return await episode_service.create_episode(
            case_id=body.case_id,
            architecture=body.architecture,
            episode_id=body.episode_id,
        )

    @app.get("/v1/episodes/{episode_id}")
    async def get_episode(
        episode_id: str,
    ) -> dict[str, Any]:
        return await episode_service.get_episode(episode_id)

    @app.post("/v1/episodes/{episode_id}/steps")
    async def step_episode(
        episode_id: str,
        body: StepEpisodeRequest,
    ) -> dict[str, Any]:
        return await episode_service.step_episode(
            episode_id=episode_id,
            request_id=body.request_id,
            action_type=body.action_type,
            arguments=body.arguments,
        )

    return app


def _error_message(error: Exception) -> str:
    if error.args:
        return str(error.args[0])
    return str(error)


app = create_app()
