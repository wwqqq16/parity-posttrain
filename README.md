# ParityPostTrain

ParityPostTrain is a focused research-engineering project for studying
the interface between agentic LLM rollouts and post-training systems.

The project measures token-level probability alignment between an
inference rollout backend and a PyTorch training backend while
benchmarking trajectory quality, latency, throughput, and memory usage.

## Pipeline

1. Generate deterministic agent tasks.
2. Produce multi-turn tool-use rollouts.
3. Evaluate trajectory correctness and tool use.
4. Rescore rollout tokens through the training backend.
5. Measure rollout-trainer logprob parity.
6. Benchmark throughput, latency, and memory.
7. Perform a minimal model update.

## Planned v0.1 Features

- Typed multi-turn trajectory representation
- Deterministic local tools and agent tasks
- Exact trajectory evaluators
- Hugging Face rollout backend
- vLLM rollout backend
- PyTorch trainer-side rescoring
- Token-level parity metrics
- Throughput, latency, and memory benchmarks
- Minimal rollout-evaluate-update loop

## Status

Work in progress. The first release will provide a reproducible
single-device benchmark and a small end-to-end training demonstration.
