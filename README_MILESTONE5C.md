# Milestone 5C: Controlled Prompt Ablation

This milestone adds two reproducible system-prompt conditions for the real model-backed enterprise agent:

- `baseline`: the original concise safety prompt.
- `checklist`: explicit same-order evidence preconditions, one retry for transient read-only timeouts, and escalation for unresolved risk.

The selected prompt profile is recorded in each run artifact under `run.metadata.prompt_profile`.

## Example

```bash
python scripts/run_model_enterprise_agent.py \
  --case payment_timeout_recoverable \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --max-steps 8 \
  --max-new-tokens 96 \
  --prompt-profile checklist
```

Compare this run against the previously saved baseline artifact. Keep model, case, decoding settings, environment, and evaluator fixed.
