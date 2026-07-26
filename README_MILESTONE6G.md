# Milestone 6G — gRPC Internal Workflow Service

This milestone adds a typed internal RPC surface over the existing
vendor-payment workflow service.

## Architecture

- FastAPI remains the external REST control plane.
- gRPC provides typed internal service-to-service calls.
- Both transports inject and reuse the same
  `VendorPaymentWorkflowService` business object.
- Workflow, review, resume, guard, evaluation, and event logic are not
  duplicated in the transport adapters.

## Protocol

The protobuf contract defines:

- `Health`
- `CreateWorkflow`
- `GetRun`
- `ListReviews`
- `SubmitReview`
- `ResumeWorkflow`

Requests use typed protobuf fields and a typed review-decision enum. Responses
surface stable identity and status fields plus a `google.protobuf.Struct`
snapshot containing the complete existing workflow representation.

## Error mapping

- Missing workflow or review: `NOT_FOUND`
- Invalid identifiers, empty fields, or review decisions: `INVALID_ARGUMENT`
- Invalid workflow transitions: `FAILED_PRECONDITION`

## Install

```bash
./.venv/bin/python -m pip install -r requirements-grpc.txt
```

## Test

```bash
./.venv/bin/python -m pytest -q tests/test_grpc_workflow.py
```

The integration tests start a real in-process gRPC server on an ephemeral
local port. They exercise protobuf serialization, RPC dispatch, status-code
mapping, review/resume behavior, and REST/gRPC reuse of the same business
service.

## Demo

```bash
./.venv/bin/python -m scripts.run_grpc_workflow_demo
```

## Run the service

Memory transport:

```bash
./.venv/bin/python -m services.grpc.runtime_server
```

Kafka-compatible transport:

```bash
EVENT_TRANSPORT=kafka \
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 \
./.venv/bin/python -m services.grpc.runtime_server
```

Configuration:

- `GRPC_HOST` defaults to `127.0.0.1`
- `GRPC_PORT` defaults to `50051`
- `GRPC_MAX_WORKERS` defaults to `8`

## Honest framing

This is a locally tested gRPC service contract and adapter. The tests run a
real local gRPC server and client, but they do not claim a production cluster,
TLS deployment, service mesh, load balancer, or distributed persistence.
