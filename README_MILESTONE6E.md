# Milestone 6E — Planner–Critic–Executor Coordination

This milestone adds a structured multi-role coordination layer for the
vendor-payment workflow.

## Roles

- **Planner** produces a typed plan artifact but cannot dispatch tools.
- **Critic** reviews the plan, records structured issues, and may approve,
  revise, or reject the proposed steps.
- **Executor** accepts only critic-approved proposals and independently
  rechecks the runtime execution guard before every sensitive action.
- **Human reviewer** remains a separate authority for changed bank accounts.

## Demonstrated flow

For the changed-bank-account case:

1. The planner collects evidence and initially proposes payment.
2. The critic identifies that independent human verification is required.
3. The critic replaces payment with a review request.
4. The executor runs the revised proposal and pauses the workflow.
5. A human reviewer verifies the account.
6. The workflow explicitly resumes.
7. Planner, critic, and executor run a second coordination phase.
8. The runtime guard rechecks the final payment action.
9. The evaluator records task success without a policy violation.

## Structured artifacts

- `PlanArtifact`
- `CritiqueArtifact`
- `ExecutionProposal`
- Versioned lifecycle events for plans, critiques, proposals, reviews,
  tool executions, guard rejections, resumes, and evaluations

## Run tests

```bash
./.venv/bin/python -m pytest -q tests/test_coordination.py
```

## Run demo

```bash
./.venv/bin/python -m scripts.run_vendor_payment_coordination_demo
```

## Honest framing

The role implementations are deterministic so that architecture and policy
semantics are reproducible in tests. They demonstrate multi-role contracts,
independent critique, guarded execution, and resumable coordination. They do
not establish that multi-agent systems are universally superior, and the
deterministic roles can later be replaced by separate model backends without
changing the artifact or event interfaces.
