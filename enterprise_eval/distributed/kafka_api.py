"""FastAPI application wired to a Kafka-compatible event publisher."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from enterprise_eval.distributed.api import create_app
from enterprise_eval.distributed.kafka_events import (
    DEFAULT_EVENT_TOPIC,
    KafkaEventPublisher,
)
from enterprise_eval.distributed.service import EpisodeService

publisher = KafkaEventPublisher(
    bootstrap_servers=os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "127.0.0.1:19092",
    ),
    topic=os.getenv(
        "KAFKA_EVENT_TOPIC",
        DEFAULT_EVENT_TOPIC,
    ),
)
service = EpisodeService(publisher)


@asynccontextmanager
async def lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    """Own the producer lifecycle with the HTTP process."""

    await publisher.start()
    try:
        yield
    finally:
        await publisher.stop()


app = create_app(service, lifespan=lifespan)
