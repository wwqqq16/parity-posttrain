# Milestone 5A: Model-backed enterprise agent

This milestone adds a bounded model-driven action loop on top of the deterministic
enterprise refund environment.

It deliberately validates the protocol with a fake backend before loading a real
Hugging Face model. Every generation record stores the actual prompt token IDs,
generated token IDs, and generated token log-probabilities returned by the rollout
backend.

## Added files

- `enterprise_eval/model_agent.py`
- `scripts/run_model_enterprise_agent.py`
- `tests/test_enterprise_model_agent.py`

## Test

```bash
python -m pytest tests/test_enterprise_model_agent.py -q
```

## Real local model run

```bash
python scripts/run_model_enterprise_agent.py \
  --case eligible_standard \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --device mps
```

A protocol error or exhausted action budget safely escalates to human review rather
than claiming that an irreversible action succeeded.
