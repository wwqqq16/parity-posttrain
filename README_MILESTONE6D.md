# Milestone 6D — FastAPI Control Plane and Event Contracts

This milestone exposes the vendor-payment workflow through a local FastAPI
control plane and emits versioned lifecycle events through a publisher
interface.

## Endpoints

- `GET /health`
- `POST /workflows/vendor-payments`
- `GET /runs/{run_id}`
- `GET /reviews`
- `POST /reviews/{review_id}/approve`
- `POST /reviews/{review_id}/reject`
- `POST /runs/{run_id}/resume`
- `GET /events?run_id=...`

## Event contract

The local implementation uses `InMemoryEventPublisher` and
`JsonlEventPublisher`, but every event has a Kafka-compatible envelope:

- `event_id`
- `event_type`
- `run_id`
- `sequence`
- `timestamp`
- `schema_version`
- `payload`

Core event types include:

- `workflow.created`
- `tool.execution.completed`
- `guard.action.rejected`
- `review.requested`
- `review.completed`
- `workflow.resumed`
- `evaluation.completed`

## Install service dependencies

```bash
./.venv/bin/python -m pip install -r requirements-service.txt
```

## Run tests

```bash
./.venv/bin/python -m pytest -q tests/test_vendor_payment_api.py
```

## Run API

```bash
./.venv/bin/python -m scripts.run_vendor_payment_api
```

Then open the generated OpenAPI documentation at `/docs`.

## Honest framing

This is a locally deployable control-plane prototype with deterministic
business semantics and versioned event contracts. The in-memory publisher is
not itself Kafka. A later adapter can publish the same envelopes to
Kafka/Redpanda without changing workflow logic.
