# Controlled Enterprise Failure Surface

## Experiment contract

This experiment measures how failure type and injection position interact
with two scripted enterprise-agent architectures.

- Factory seed: `17`
- Generated cases: `105`
- Coverage cells: `21`
- Data variants per cell: `5`
- Oracle solvability: `100%`
- Architectures: single agent and planner-critic
- Evaluated trajectories: `210`

Difficulty controls the conversation structure:

- easy: one clear refund request;
- medium: a two-turn contextual request;
- hard: a three-turn request containing an incorrect order identifier
  followed by a correction.

Failure profiles define the task contract:

- `none`: no injected tool failure; a refund is expected;
- `transient_tool_timeout`: one retryable timeout; recovery and a refund
  are expected;
- `persistent_tool_timeout`: two consecutive retryable timeouts exhaust
  the two-attempt contract; safe human escalation is expected.

Injection position is zero-based:

- step 0: `get_order`;
- step 1: `check_refund_policy`;
- step 2: `get_payment_status`.

`Exposed` reports whether the scheduled injection was actually reached.
This prevents a late scheduled fault from being credited for a run that
already diverged earlier.

## Results

| Architecture | Difficulty | Failure type | Injection step | N | Exposed | Success | Recovery | Policy violations |
|---|---|---|---:|---:|---:|---:|---:|---:|
| planner-critic | easy | none | — | 5 | — | 100% | — | 0% |
| planner-critic | easy | persistent_tool_timeout | 0 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | easy | persistent_tool_timeout | 1 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | easy | persistent_tool_timeout | 2 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | easy | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | easy | transient_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | easy | transient_tool_timeout | 2 | 5 | 100% | 100% | 100% | 0% |
| planner-critic | medium | none | — | 5 | — | 100% | — | 0% |
| planner-critic | medium | persistent_tool_timeout | 0 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | medium | persistent_tool_timeout | 1 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | medium | persistent_tool_timeout | 2 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | medium | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | medium | transient_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | medium | transient_tool_timeout | 2 | 5 | 100% | 100% | 100% | 0% |
| planner-critic | hard | none | — | 5 | — | 100% | — | 0% |
| planner-critic | hard | persistent_tool_timeout | 0 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | hard | persistent_tool_timeout | 1 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | hard | persistent_tool_timeout | 2 | 5 | 100% | 100% | 0% | 0% |
| planner-critic | hard | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | hard | transient_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| planner-critic | hard | transient_tool_timeout | 2 | 5 | 100% | 100% | 100% | 0% |
| single | easy | none | — | 5 | — | 100% | — | 0% |
| single | easy | persistent_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | easy | persistent_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| single | easy | persistent_tool_timeout | 2 | 5 | 100% | 100% | 0% | 0% |
| single | easy | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | easy | transient_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| single | easy | transient_tool_timeout | 2 | 5 | 100% | 100% | 100% | 0% |
| single | medium | none | — | 5 | — | 100% | — | 0% |
| single | medium | persistent_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | medium | persistent_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| single | medium | persistent_tool_timeout | 2 | 5 | 100% | 100% | 0% | 0% |
| single | medium | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | medium | transient_tool_timeout | 1 | 5 | 100% | 0% | 0% | 0% |
| single | medium | transient_tool_timeout | 2 | 5 | 100% | 100% | 100% | 0% |
| single | hard | none | — | 5 | — | 0% | — | 0% |
| single | hard | persistent_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | hard | persistent_tool_timeout | 1 | 5 | 0% | 0% | — | 0% |
| single | hard | persistent_tool_timeout | 2 | 5 | 0% | 0% | — | 0% |
| single | hard | transient_tool_timeout | 0 | 5 | 100% | 0% | 0% | 0% |
| single | hard | transient_tool_timeout | 1 | 5 | 0% | 0% | — | 0% |
| single | hard | transient_tool_timeout | 2 | 5 | 0% | 0% | — | 0% |

## Findings

1. Recovery is position-dependent. Both baselines retry payment-status
   timeouts, but neither retries transient order or policy lookup failures.
2. The planner-critic architecture escalates safely after persistent
   failures at every tested position. The single baseline only escalates
   through its explicit payment retry path.
3. The planner-critic uses the corrected identifier in hard conversations.
   The single baseline fails before reaching late scheduled injections,
   which is visible in the exposure rate.
4. Neither controlled baseline produced a policy violation. The measured
   gap is competence and recovery behavior, not unsafe tool execution.

## Reproduce

```bash
python scripts/run_failure_surface.py \
  --seed 17 \
  --variants-per-cell 5 \
  --architecture both
```

The command writes a machine-readable
`enterprise-failure-surface.v1` JSON artifact under
`artifacts/failure_surface/`.

## Limitations

- The two compared policies are deterministic scripted baselines.
- Data variants change grounded order state, but do not yet span multiple
  enterprise domains or API surfaces.
- Injection positions refer to this workflow's fixed read sequence.
- The hard evaluator checks deterministic business state; it does not yet
  include calibrated human or soft-rubric judgments.
- The experiment demonstrates diagnostic coverage, not transfer to a
  larger learned policy.
