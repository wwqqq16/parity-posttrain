# Milestone 4: Semantic Post-Training Export

This milestone converts deterministic enterprise-agent trajectories into three
semantic post-training datasets without inventing model token IDs or rollout
log-probabilities.

## Outputs

```text
artifacts/enterprise_posttrain/
├── manifest.json
├── preferences.jsonl
├── sft.jsonl
└── step_rewards.jsonl
```

- `sft.jsonl`: successful, policy-compliant tool-use conversations.
- `preferences.jsonl`: same-case chosen/rejected trajectory pairs when one run
  succeeds and another fails.
- `step_rewards.jsonl`: deterministic verifier signals attached to each action.
- `manifest.json`: counts, paths, schema versions, and training-readiness notes.

## Why semantic export comes first

The current enterprise benchmark uses scripted agents. It therefore has real
business actions, observations, failure labels, and rewards, but no model
`prompt_token_ids`, `generated_token_ids`, or rollout log-probabilities.
Producing those fields now would fabricate model data. Token-level integration
will follow after a real model-backed agent adapter is added.

## Run

```bash
python scripts/run_enterprise_benchmark.py --architecture both
python scripts/export_enterprise_posttrain_data.py
```

To include the controlled safe/unsafe prompt-injection artifacts:

```bash
python scripts/export_enterprise_posttrain_data.py \
  --include-failure-injection
```

## Controlled benchmark expectations

With one fresh single-agent and one fresh planner-critic run for each of the 14
cases, the exporter should normally produce:

- 28 source artifacts
- 25 SFT records
- 3 preference pairs
- one step-reward record per trajectory step

The exact source count can be larger if the artifact directory contains runs
from earlier executions. Clean the output/input artifact directories or point
the exporter to a fresh directory for reproducible counts.
