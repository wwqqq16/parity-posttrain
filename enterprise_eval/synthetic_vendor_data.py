from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from enterprise_eval.models import VendorPaymentCase
from enterprise_eval.privacy_audit import PrivacyAuditor, PrivacyAuditReport


@dataclass(frozen=True)
class SyntheticDataProvenance:
    generator: str
    generator_version: str
    source: str
    seed: int
    record_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticVendorRecord:
    vendor_id: str
    vendor_name: str
    contact_email: str
    contact_phone: str
    bank_account: str
    routing_number: str
    is_synthetic: bool
    schema_version: str
    provenance: SyntheticDataProvenance

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class PrivacySafeExportResult:
    output_path: Path
    records_written: int
    audit_reports: tuple[PrivacyAuditReport, ...]


class SyntheticVendorGenerator:
    """Deterministic generator for privacy-audited vendor records."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed

    def generate_for_cases(
        self,
        cases: Iterable[VendorPaymentCase],
    ) -> tuple[SyntheticVendorRecord, ...]:
        unique_cases = {
            case.vendor_id: case
            for case in cases
        }
        records = [
            self._generate_record(vendor_id, index)
            for index, vendor_id in enumerate(sorted(unique_cases), start=1)
        ]
        return tuple(records)

    def _generate_record(
        self,
        vendor_id: str,
        record_index: int,
    ) -> SyntheticVendorRecord:
        return SyntheticVendorRecord(
            vendor_id=vendor_id,
            vendor_name=f"<SYNTHETIC_VENDOR_{record_index:04d}>",
            contact_email="<REDACTED_EMAIL>",
            contact_phone="<REDACTED_PHONE>",
            bank_account=self._token("BANK_ACCOUNT", vendor_id),
            routing_number=self._token("ROUTING_NUMBER", vendor_id),
            is_synthetic=True,
            schema_version="1.0",
            provenance=SyntheticDataProvenance(
                generator="SyntheticVendorGenerator",
                generator_version="1.0",
                source="schema_constrained_synthetic",
                seed=self.seed,
                record_index=record_index,
            ),
        )

    def _token(self, field_name: str, vendor_id: str) -> str:
        material = f"{self.seed}:{vendor_id}:{field_name}".encode()
        digest = sha256(material).hexdigest()[:16]
        return f"<TOKENIZED_{field_name}:{digest}>"


def export_privacy_safe_jsonl(
    records: Iterable[SyntheticVendorRecord],
    output_path: Path,
    *,
    auditor: PrivacyAuditor,
    canaries: Iterable[str] = (),
) -> PrivacySafeExportResult:
    record_list = tuple(records)
    reports = tuple(
        auditor.require_safe(record.to_dict(), canaries=canaries)
        for record in record_list
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        json.dumps(record.to_dict(), sort_keys=True)
        for record in record_list
    ]
    output_path.write_text(
        "".join(f"{line}\n" for line in serialized),
        encoding="utf-8",
    )
    return PrivacySafeExportResult(
        output_path=output_path,
        records_written=len(record_list),
        audit_reports=reports,
    )
