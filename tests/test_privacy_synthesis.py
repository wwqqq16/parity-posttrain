from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_eval.privacy_audit import PrivacyAuditError, PrivacyAuditor
from enterprise_eval.synthetic_vendor_data import (
    SyntheticVendorGenerator,
    export_privacy_safe_jsonl,
)
from enterprise_eval.vendor_payment_cases import VENDOR_PAYMENT_CASES


def test_generator_is_deterministic_and_case_aligned() -> None:
    first = SyntheticVendorGenerator(seed=17).generate_for_cases(
        VENDOR_PAYMENT_CASES.values()
    )
    second = SyntheticVendorGenerator(seed=17).generate_for_cases(
        reversed(tuple(VENDOR_PAYMENT_CASES.values()))
    )

    assert first == second
    assert {record.vendor_id for record in first} == {
        case.vendor_id for case in VENDOR_PAYMENT_CASES.values()
    }


def test_generated_records_are_marked_and_tokenized() -> None:
    records = SyntheticVendorGenerator().generate_for_cases(
        VENDOR_PAYMENT_CASES.values()
    )

    assert records
    for record in records:
        assert record.is_synthetic
        assert record.contact_email == "<REDACTED_EMAIL>"
        assert record.contact_phone == "<REDACTED_PHONE>"
        assert record.bank_account.startswith("<TOKENIZED_BANK_ACCOUNT:")
        assert record.routing_number.startswith("<TOKENIZED_ROUTING_NUMBER:")
        assert record.provenance.source == "schema_constrained_synthetic"


def test_generated_records_pass_privacy_audit() -> None:
    auditor = PrivacyAuditor()
    records = SyntheticVendorGenerator().generate_for_cases(
        VENDOR_PAYMENT_CASES.values()
    )

    for record in records:
        report = auditor.audit_payload(record.to_dict())
        assert report.passed
        assert report.findings == ()


def test_auditor_detects_email_phone_ssn_and_bank_account() -> None:
    auditor = PrivacyAuditor()
    report = auditor.audit_payload(
        {
            "email": "owner@example.com",
            "phone": "415-555-0199",
            "tax_id": "123-45-6789",
            "bank_account": "987654321012",
        }
    )

    assert not report.passed
    assert {finding.finding_type for finding in report.findings} == {
        "email",
        "phone",
        "ssn",
        "sensitive_numeric_identifier",
    }
    assert all(
        "example.com" not in finding.preview
        for finding in report.findings
    )


def test_auditor_detects_canary_without_returning_full_secret() -> None:
    canary = "ENTERPRISE-CANARY-DO-NOT-EXPORT"
    report = PrivacyAuditor().audit_payload(
        {"notes": f"Accidental leak: {canary}"},
        canaries=(canary,),
    )

    assert not report.passed
    assert report.findings[0].finding_type == "canary"
    assert canary not in report.findings[0].preview


def test_require_safe_rejects_unsafe_payload() -> None:
    with pytest.raises(PrivacyAuditError, match="email"):
        PrivacyAuditor().require_safe(
            {"contact_email": "finance@example.com"}
        )


def test_export_writes_only_audited_records(tmp_path: Path) -> None:
    auditor = PrivacyAuditor()
    records = SyntheticVendorGenerator(seed=31).generate_for_cases(
        VENDOR_PAYMENT_CASES.values()
    )
    output_path = tmp_path / "vendors.jsonl"

    result = export_privacy_safe_jsonl(
        records,
        output_path,
        auditor=auditor,
        canaries=("CANARY-NOT-PRESENT",),
    )

    assert result.records_written == len(records)
    assert all(report.passed for report in result.audit_reports)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    payloads = [json.loads(line) for line in lines]
    assert all(payload["is_synthetic"] is True for payload in payloads)
    assert all("provenance" in payload for payload in payloads)


def test_sensitive_numeric_scan_is_context_aware() -> None:
    auditor = PrivacyAuditor()

    safe = auditor.audit_payload({"amount_cents": "4800000"})
    unsafe = auditor.audit_payload({"routing_number": "021000021"})

    assert safe.passed
    assert not unsafe.passed
    assert unsafe.findings[0].finding_type == "sensitive_numeric_identifier"
