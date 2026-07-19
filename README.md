# ParityPostTrain

ParityPostTrain is a small, reproducible research-engineering system for
studying consistency between agentic LLM rollout and post-training
execution paths.

The central question is not only whether an agent completes a task, but
whether the inference backend and training backend assign the same
token-level probabilities to the exact same trajectory.

The project combines:

- multi-turn tool-using agent rollouts;
- deterministic task execution and trajectory evaluation;
- rollout-side token IDs and logprobs;
- trainer-side teacher-forced logprob recomputation;
- token-level rollout-trainer parity analysis;
- controlled optimization experiments;
- trajectory-aware policy-update normalization.

## Motivation

Agentic post-training commonly contains two distinct execution paths:

~~~text
Agent task
    ↓
Inference backend generates a multi-turn tool-use trajectory
    ↓
Tools execute and the complete trajectory is recorded
    ↓
Evaluator assigns a trajectory-level reward
    ↓
Training backend recomputes token logprobs
    ↓
Policy objective updates the model
~~~

These paths can differ because of:

- incremental decoding versus full-sequence forward passes;
- KV-cache behavior;
- numerical precision;
- attention-mask and padding semantics;
- token boundary reconstruction;
- batching;
- objective normalization.

A small numerical mismatch can alter importance ratios, clipping
decisions, KL estimates, and eventually the optimization trajectory.

ParityPostTrain makes these differences measurable and reproducible.

## Current Pipeline

The current implementation supports:

1. Deterministic local agent tools.
2. Typed agent tasks and structured multi-turn trajectories.
3. Tool-call protocol parsing and execution.
4. Exact trajectory evaluation and reward assignment.
5. Hugging Face rollout with generated token IDs and token logprobs.
6. Trainer-side teacher-forced logprob recomputation.
7. Token-level rollout-trainer parity reports.
8. Controlled cache, dtype, and device comparisons.
9. Training-example extraction from benchmark artifacts.
10. Right-padded trajectory batch collation.
11. Differentiable batched trainer-side rescoring.
12. A clipped trajectory policy objective.
13. A real single-step optimizer update.
14. Token-, sequence-, and trajectory-normalized policy updates.
15. Reproducible JSON experiment artifacts.

## Policy-Objective Normalization

Agent trajectories can contain different numbers of generated tokens and
different numbers of assistant turns. This creates three distinct
weighting semantics.

### Token normalization

All generated tokens are averaged together:

~~~text
mean(objective over every generated token)
~~~

Long generations receive more total weight.

### Sequence normalization

Each assistant generation is averaged separately, followed by a batch
average:

~~~text
mean(
    mean(objective over generated tokens in one assistant turn)
)
~~~

Each assistant turn receives equal weight, but a multi-turn trajectory
still receives more total weight than a single-turn trajectory.

### Trajectory normalization

All assistant turns belonging to the same `task_id` are grouped into one
complete trajectory:

~~~text
mean(
    mean(objective over every generated token in one trajectory)
)
~~~

Each complete agent trajectory receives equal weight regardless of its
number of turns or generated tokens.

## Reproducible Training Comparison

The following command loads stored trajectories, restores identical
initial weights before every condition, performs one controlled update
for each normalization mode, and writes a JSON report:

~~~bash
python scripts/run_training_comparison.py \
  --artifact artifacts/agent_benchmark.json \
  --task-ids catalog_004 shopping_004 \
  --device cpu \
  --normalizations token sequence trajectory \
  --learning-rate 0.05 \
  --clip-epsilon 0.2 \
  --max-gradient-norm 1.0 \
  --output artifacts/training_comparison.json
~~~

The controlled comparison replaces stored rollout logprobs with
trainer-computed baseline logprobs before optimization. Therefore, every
condition starts with:

~~~text
importance ratio = 1
approximate KL = 0
clip fraction = 0
~~~

This isolates the effect of normalization from rollout-trainer parity
error.

### Experimental setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Device: CPU
- Dtype: `torch.float32`
- Updated parameter: `model.norm.weight`
- Trainable parameters: 896
- Learning rate: `0.05`
- Positive trajectory: `catalog_004`
  - reward: `1.0`
  - turns: `2`
  - generated tokens: `42`
- Negative trajectory: `shopping_004`
  - reward: `0.0`
  - turns: `1`
  - generated tokens: `47`

### Results

| Normalization | Initial loss | Gradient norm | Parameter delta | Positive mean shift | Negative mean shift |
|---|---:|---:|---:|---:|---:|
| Token | 1.9476e-1 | 4.4818e-3 | 6.5804e-5 | +1.7489e-6 | -2.2565e-6 |
| Sequence | -1.1921e-7 | 3.9991e-3 | 4.7684e-5 | +2.7450e-6 | -1.1950e-6 |
| Trajectory | 0.0000 | 4.6420e-3 | 5.6267e-5 | +3.2119e-6 | -1.3800e-6 |

All three objectives move the successful trajectory upward and the
failed trajectory downward.

The comparison also demonstrates that the normalization level changes
the effective optimization problem:

- token normalization is affected by generation length;
- sequence normalization assigns equal weight to assistant turns;
- trajectory normalization assigns equal weight to complete agent runs.

## Controlled Rollout-Trainer Parity Finding

A fixed-token-sequence experiment isolated a significant mismatch to one
execution condition:

| Device | Dtype | KV cache | Maximum absolute error | Result |
|---|---|---:|---:|---|
| MPS | float16 | enabled | 3.307e-2 | fail |
| MPS | float16 | disabled | 3.87e-7 | pass |
| CPU | float32 | enabled | 5.57e-5 | pass |
| CPU | float32 | disabled | 2.38e-5 | pass |

The large mismatch appears specifically in MPS float16 incremental
decoding with KV caching. Disabling the cache reduces the error to near
numerical noise.

This demonstrates the infrastructure issue ParityPostTrain is designed
to expose: an inference optimization can improve generation performance
while changing the probabilities later assumed by the trainer.

The experiment can be reproduced by running the controlled conditions
with `scripts/run_controlled_parity.py` and summarizing the generated
reports with `scripts/summarize_controlled_parity.py`.

## Repository Structure

~~~text
src/parity_posttrain/
├── agent/          # Tool protocol, registry, and multi-turn runner
├── benchmarks/     # Agent benchmark records and summaries
├── core/           # Tasks and trajectory data structures
├── evals/          # Trajectory evaluation and reward assignment
├── parity/         # Token-level and controlled parity analysis
├── rollout/        # Hugging Face rollout backend
└── training/       # Examples, batches, rescoring, objectives, and updates

scripts/
├── run_agent_benchmark.py
├── run_agent_parity.py
├── run_controlled_parity.py
├── run_hf_parity.py
├── run_training_comparison.py
└── summarize_controlled_parity.py
~~~

## Installation

ParityPostTrain requires Python 3.12 or later.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
~~~

## Quality Checks

~~~bash
python -m pytest
python -m ruff check .
python -m mypy
~~~

The current test suite contains 119 tests.

## Generated Artifacts

Experiment outputs are written under `artifacts/`.

Generated JSON, JSONL, CSV, and PNG artifacts are ignored by Git by
default so local benchmark runs do not modify the working tree.

The artifact directory itself is retained through
`artifacts/.gitkeep`.

## Current Scope

The current project demonstrates:

- exact token-level parity measurement;
- real agent trajectory extraction;
- batched trainer-side rescoring;
- differentiable clipped policy optimization;
- real model parameter updates;
- reproducible normalization comparisons.

It is not yet:

- a distributed RL training system;
- a complete PPO or GRPO implementation;
- a vLLM-integrated production trainer;
- a multi-node rollout-training architecture.

These extensions can be added without changing the core parity
abstractions.

## License

Apache License 2.0.
