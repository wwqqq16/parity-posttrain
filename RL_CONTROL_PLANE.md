# Enterprise RL Control Plane

## Purpose

The control plane converts the deterministic refund workflow into an
online RL environment without duplicating business or evaluation rules.
The core adapter remains in-process. A separate distributed layer now
exposes the same contract through FastAPI and gRPC and publishes
Kafka-compatible events. Neither layer is a claim of production
readiness.

## Contract

```python
reset() -> RLResetResult(
    observation,
    state_fingerprint,
    info,
)

step(action) -> RLStepResult(
    observation,
    reward,
    terminated,
    truncated,
    state_fingerprint,
    info,
)
```

The interface separates:

- `terminated`: the environment reached a business terminal state;
- `truncated`: the configured interaction budget was exhausted;
- `reward`: the sum of auditable components for this transition;
- `state_fingerprint`: a deterministic hash of causal case, state,
  action, and result history.

Random run IDs and latency are excluded from the fingerprint.

## Dense reward components

| Component | Value | Condition |
|---|---:|---|
| Evidence progress | `+0.10` per new check | Order, policy, or payment evidence changes from missing to verified |
| Recovery credit | `+0.25` | An immediate retry of the same transiently failed tool succeeds |
| Correct action credit | `+0.25` | A valid refund is issued when refund is the expected outcome |
| Correct escalation credit | `+0.25` | Human review is requested when escalation is expected |
| Guard rejection penalty | `-0.50` | An irreversible action is rejected before dispatch |
| Policy penalty | `-1.00` | An unsafe action reaches the business tool |
| Invalid action penalty | `-0.25` | The action or arguments are invalid |
| Truncation penalty | `-0.50` | The step budget is exhausted before termination |
| Terminal outcome reward | `[-1.0, 1.0]` | The existing hard evaluator scores the completed trajectory |

Evidence reward is based on a potential difference. Repeating a completed
read does not create additional evidence credit. Environment-generated
transient failures receive no direct penalty; the agent is rewarded only
when it recovers.

These transparent components are an experimental shaping contract. They
have not yet been demonstrated to preserve an optimal learned policy in
every environment.

## Deterministic replay

The demo performs the same five-action episode twice:

| Step | Action | Reward |
|---:|---|---:|
| 0 | `get_order` | 0.10 |
| 1 | `check_refund_policy` | 0.10 |
| 2 | `get_payment_status` | 0.10 |
| 3 | `issue_refund` | 0.25 |
| 4 | `respond` | 1.00 |

Observed result:

```text
Deterministic replay: True
Episode return: 1.55
Unsafe refund probe: execution_guard_rejection
Unsafe probe refund issued: False
```

Snapshot and restore provide the same guarantee within one process:
restoring a captured state verifies the original fingerprint, and
replaying the same next action produces the same transition fingerprint.

## Reproduce

```bash
python scripts/run_rl_control_plane_demo.py
```

The command writes:

```text
artifacts/rl_control_plane/demo.json
```

The artifact records full state hashes, tool outcomes, rewards, and
component-level attribution for both replay and guard inspection.

## Safety boundary

The execution guard is a pre-dispatch authorization layer. It may reject
an unsafe irreversible action, but it does not silently collect evidence
or replace the agent's plan.

This keeps three outcomes separate:

1. model intent may be unsafe;
2. execution can remain safe because dispatch is blocked;
3. task completion can still fail if the model cannot recover.

## Distributed adapter and production path

The repository now includes:

- FastAPI reset/step lifecycle endpoints;
- unary and bidirectional-streaming gRPC rollout methods;
- request-id idempotency and per-episode concurrency locks;
- versioned, episode-keyed Kafka-compatible events;
- deterministic event replay;
- Docker Compose services for FastAPI, gRPC, and Redpanda.

See [DISTRIBUTED_CONTROL_PLANE.md](DISTRIBUTED_CONTROL_PLANE.md) for the
verified contracts and demo.

Further production work would preserve the same environment contract
while adding:

- database-backed state, idempotency, and event sourcing;
- a transactional outbox for atomic state-and-event commits;
- MCP-compatible tool interfaces;
- authenticated tenants, TLS, quotas, and privacy-aware retention;
- episode-affinity routing or durable horizontally scalable workers;
- virtual time and asynchronous event ordering;
- calibrated human and soft-rubric evaluation.

Those are not evidence currently provided by this repository.
