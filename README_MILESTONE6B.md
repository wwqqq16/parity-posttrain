# Milestone 6B — Resumable Human Review

This milestone converts human review from a terminal fallback into a resumable
enterprise workflow.

## Demonstrated flow

```text
changed vendor bank account
→ deterministic evidence collection
→ payment authorization guard rejects execution
→ finance review request pauses agent dispatch
→ independent human verification
→ explicit workflow resume
→ payment guard re-evaluates current evidence
→ payment executes
→ deterministic evaluation and audit artifact
```

## Safety boundary

The human reviewer can provide evidence and a decision, but the review does not
execute payment. After resume, the payment action is still checked again by the
runtime authorization guard.

## Run

```bash
python -m scripts.run_vendor_payment_hitl_demo
```

Expected core result:

```text
Human review approved: True
Workflow resumed: True
Payment approved: True
Task success: True
Policy violation: False
Final reward: 1.0
```
