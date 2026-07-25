# Milestone 5D: Runtime Execution Guard

This milestone separates model intent from execution authority.

The `none` guard profile preserves the prior behavior: a model-generated
`issue_refund` action is dispatched to the environment and can become a
`policy_violation_attempt`.

The `prerequisite` guard profile checks the environment state before dispatching
`issue_refund`. Missing verification or unsafe case conditions produce an
`execution_guard_rejection`. The sensitive tool is not executed, the rejection is
returned to the model, and the model can gather evidence, retry, or escalate.

## Controlled comparison

```bash
python scripts/run_model_enterprise_agent.py \
  --case payment_timeout_recoverable \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --max-steps 8 \
  --max-new-tokens 96 \
  --prompt-profile checklist \
  --guard-profile none
```

```bash
python scripts/run_model_enterprise_agent.py \
  --case payment_timeout_recoverable \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --max-steps 8 \
  --max-new-tokens 96 \
  --prompt-profile checklist \
  --guard-profile prerequisite
```

Artifacts record `guard_profile`, per-generation `dispatch_status`, and
`runtime_guard_rejections`. A blocked unsafe intent is distinguishable from an
actually dispatched policy violation.
