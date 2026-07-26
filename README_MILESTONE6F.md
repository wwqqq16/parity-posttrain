# Milestone 6F — Kafka / Redpanda Event Transport

This milestone adds a broker-backed implementation of the existing
`EventPublisher` contract.

## Capabilities

- Kafka and Redpanda-compatible producer adapter
- Versioned JSON event envelopes
- Run-ID message keys for stable per-workflow partition ordering
- Topic routing for workflow, review, security, and evaluation events
- Idempotent-producer configuration
- Synchronous delivery confirmation before a publish is considered successful
- Delivery receipts with topic, partition, and offset
- Environment-driven runtime selection between memory, JSONL, Kafka, and
  Redpanda transports
- Process-local confirmed-event cache for the existing control-plane read API

## Topics

With the default `enterprise.agent` prefix:

- `enterprise.agent.workflows.v1`
- `enterprise.agent.reviews.v1`
- `enterprise.agent.security.v1`
- `enterprise.agent.evaluations.v1`

## Local contract demo

The local demo uses a delivery-confirming producer double. It exercises the
real serialization, routing, keying, receipt, and failure semantics without
requiring an external broker.

```bash
./.venv/bin/python -m scripts.run_event_transport_demo
```

## Unit and workflow tests

```bash
./.venv/bin/python -m pytest -q tests/test_event_transport.py
```

## Optional real broker dependency

```bash
./.venv/bin/python -m pip install -r requirements-event-transport.txt
```

Configure a Kafka or Redpanda endpoint:

```bash
export EVENT_TRANSPORT=redpanda
export KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
export KAFKA_TOPIC_PREFIX=enterprise.agent
```

Run a real broker smoke workflow:

```bash
./.venv/bin/python -m scripts.run_kafka_workflow_smoke
```

Run the FastAPI service with the configured transport:

```bash
./.venv/bin/python -m uvicorn services.api.runtime_app:app \
  --host 127.0.0.1 \
  --port 8000
```

## Security configuration

The runtime factory accepts these optional environment variables without
printing their values:

- `KAFKA_SECURITY_PROTOCOL`
- `KAFKA_SASL_MECHANISM`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`

## Honest framing

The adapter publishes the same versioned contracts to Kafka or Redpanda and
waits for broker delivery confirmation. Unit and workflow tests use a producer
double; they do not prove connectivity to an external cluster. The local
event cache supports the single-process demo only and is not presented as a
shared production event store.
