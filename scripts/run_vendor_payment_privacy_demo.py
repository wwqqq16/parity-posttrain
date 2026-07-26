from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from enterprise_eval.privacy_audit import PrivacyAuditor
from enterprise_eval.synthetic_vendor_data import (
    SyntheticVendorGenerator,
    export_privacy_safe_jsonl,
)
from enterprise_eval.vendor_payment_cases import VENDOR_PAYMENT_CASES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and audit privacy-aware synthetic vendor data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vendor_payment_privacy"),
    )
    args = parser.parse_args()

    generator = SyntheticVendorGenerator(seed=2026)
    auditor = PrivacyAuditor()
    canary = "ENTERPRISE-CANARY-DO-NOT-EXPORT"

    records = generator.generate_for_cases(VENDOR_PAYMENT_CASES.values())
    export_result = export_privacy_safe_jsonl(
        records,
        args.output_dir / "synthetic_vendor_records.jsonl",
        auditor=auditor,
        canaries=(canary,),
    )

    safe_payload: dict[str, Any] = {
        "records": [record.to_dict() for record in records],
        "export_path": str(export_result.output_path),
    }
    safe_report = auditor.audit_payload(safe_payload, canaries=(canary,))

    unsafe_probe = {
        "contact_email": "finance.owner@example.com",
        "contact_phone": "(415) 555-0199",
        "tax_id": "123-45-6789",
        "bank_account": "987654321012",
        "notes": f"Internal marker: {canary}",
    }
    unsafe_report = auditor.audit_payload(
        unsafe_probe,
        canaries=(canary,),
    )

    audit_payload = {
        "schema_version": "1.0",
        "records_generated": len(records),
        "safe_export": safe_report.to_dict(),
        "negative_control": unsafe_report.to_dict(),
        "formal_differential_privacy_claimed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "privacy_audit_report.json"
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("VENDOR PAYMENT PRIVACY DEMO")
    print("=" * 52)
    print("Synthetic records:", len(records))
    print("Safe export passed:", safe_report.passed)
    print("Unsafe negative control passed:", unsafe_report.passed)
    print(
        "Detected finding types:",
        ", ".join(
            sorted(
                {
                    finding.finding_type
                    for finding in unsafe_report.findings
                }
            )
        ),
    )
    print("JSONL export:", export_result.output_path)
    print("Audit report:", audit_path)
    print("Formal differential privacy claimed: False")


if __name__ == "__main__":
    main()
