# Milestone 3: Multi-Difficulty Benchmark and Planner-Critic Ablation

This milestone extends the enterprise refund environment into a 14-case benchmark
with three difficulty levels and two agent architectures.

## Architectures

- `single`: one-pass baseline that follows the first user intent and first order ID.
- `planner-critic`: planner reads the latest intent, a policy critic checks business
  constraints and tool evidence, and an executor applies the reviewed outcome.

## Hard cases that expose coordination value

- corrected order identifier across turns
- stale payment status that must be re-queried
- user withdrawal of an earlier refund request
- prompt injection
- exhausted tool retries
- simultaneous high-value, dispute, delivery-conflict, and window constraints

## Run

```bash
python -m pytest tests/test_enterprise_eval.py tests/test_enterprise_benchmark.py -q
python scripts/run_enterprise_benchmark.py --architecture both --show-cases
```

The local scaffold should report 15 passing tests. Benchmark latency is local
Python control-flow latency; it is not intended to estimate remote LLM latency.
Use `component_calls` to represent the architectural overhead in this scripted
ablation.
