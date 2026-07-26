# Milestone 6C — Privacy-Aware Synthetic Vendor Data

This milestone adds deterministic synthetic vendor records, tokenization and
redaction controls, provenance metadata, PII and sensitive-number scanning,
canary leakage detection, and audit-gated JSONL export.

## What this demonstrates

- Synthetic vendor records are generated from business schemas rather than
  copied from customer data.
- Contact fields are redacted and bank identifiers are deterministically
  tokenized.
- Every record includes synthesis provenance and an explicit
  `is_synthetic=true` marker.
- Export is blocked when the auditor detects email addresses, phone numbers,
  SSNs, sensitive account/routing numbers, or configured canaries.
- Audit reports contain masked previews rather than reproducing the detected
  secret.

## Run tests

```bash
./.venv/bin/python -m pytest -q tests/test_privacy_synthesis.py
```

## Run demo

```bash
./.venv/bin/python -m scripts.run_vendor_payment_privacy_demo
```

## Honest framing

This is a privacy-aware synthetic-data and leakage-auditing layer. It reduces
the risk of exporting direct identifiers and provides deterministic controls
for local enterprise-agent experiments. It does **not** provide a formal
differential-privacy guarantee, prove resistance to all re-identification
attacks, or replace a production data-governance review.
