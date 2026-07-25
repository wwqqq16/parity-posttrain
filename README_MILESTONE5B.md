# Milestone 5B — Real Token-Level Enterprise Parity

This milestone connects a real model-backed enterprise-agent artifact to the
existing post-training and parity stack.

It adds:

- conversion of every stored model generation into a validated
  `TrajectoryTrainingExample`;
- trainer-side teacher-forced rescoring;
- fixed-sequence rollout rescoring;
- three comparisons:
  - stored free-generation rollout vs trainer;
  - forced rollout vs trainer;
  - stored free-generation rollout vs forced rollout.

The three-way comparison helps distinguish generation-path discrepancies from
fixed-sequence numerical discrepancies.

Example:

```bash
python scripts/run_enterprise_model_parity.py \
  --artifact artifacts/enterprise_model/eligible_standard-<run-id>.json \
  --turn-index 0 \
  --device mps \
  --use-cache \
  --output artifacts/enterprise_model_parity/eligible-turn0.json
```
