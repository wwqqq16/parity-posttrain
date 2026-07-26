from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


class PrivacyAuditError(ValueError):
    """Raised when a payload fails the privacy leakage audit."""


@dataclass(frozen=True)
class PrivacyFinding:
    finding_type: str
    path: str
    preview: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PrivacyAuditReport:
    passed: bool
    findings: tuple[PrivacyFinding, ...]
    scanned_string_values: int
    canaries_checked: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
            "scanned_string_values": self.scanned_string_values,
            "canaries_checked": self.canaries_checked,
        }


class PrivacyAuditor:
    """Deterministic PII and canary leakage scanner for synthetic artifacts."""

    _EMAIL = re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"
    )
    _PHONE = re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    )
    _SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
    _LONG_DIGITS = re.compile(r"(?<!\d)\d{6,17}(?!\d)")
    _SENSITIVE_KEY_FRAGMENTS = (
        "account_number",
        "bank_account",
        "routing_number",
        "tax_id",
        "ssn",
    )
    _SAFE_PLACEHOLDER = re.compile(
        r"^<(?:REDACTED|TOKENIZED|SYNTHETIC)_[A-Z0-9_]+(?::[A-Fa-f0-9]+)?>$"
    )

    def audit_payload(
        self,
        payload: object,
        *,
        canaries: Iterable[str] = (),
    ) -> PrivacyAuditReport:
        normalized_canaries = tuple(
            canary for canary in (value.strip() for value in canaries) if canary
        )
        findings: list[PrivacyFinding] = []
        scanned_string_values = 0

        for path, value in self._walk_strings(payload):
            scanned_string_values += 1
            if self._SAFE_PLACEHOLDER.fullmatch(value):
                continue

            findings.extend(self._regex_findings("email", path, value, self._EMAIL))
            findings.extend(self._regex_findings("phone", path, value, self._PHONE))
            findings.extend(self._regex_findings("ssn", path, value, self._SSN))

            lowered_path = path.lower()
            if any(
                fragment in lowered_path
                for fragment in self._SENSITIVE_KEY_FRAGMENTS
            ):
                findings.extend(
                    self._regex_findings(
                        "sensitive_numeric_identifier",
                        path,
                        value,
                        self._LONG_DIGITS,
                    )
                )

            for canary in normalized_canaries:
                if canary in value:
                    findings.append(
                        PrivacyFinding(
                            finding_type="canary",
                            path=path,
                            preview=self._masked_preview(canary),
                        )
                    )

        return PrivacyAuditReport(
            passed=not findings,
            findings=tuple(findings),
            scanned_string_values=scanned_string_values,
            canaries_checked=len(normalized_canaries),
        )

    def require_safe(
        self,
        payload: object,
        *,
        canaries: Iterable[str] = (),
    ) -> PrivacyAuditReport:
        report = self.audit_payload(payload, canaries=canaries)
        if not report.passed:
            finding_types = ", ".join(
                sorted({finding.finding_type for finding in report.findings})
            )
            raise PrivacyAuditError(
                "Privacy audit failed before export: " + finding_types
            )
        return report

    @classmethod
    def _regex_findings(
        cls,
        finding_type: str,
        path: str,
        value: str,
        pattern: re.Pattern[str],
    ) -> list[PrivacyFinding]:
        return [
            PrivacyFinding(
                finding_type=finding_type,
                path=path,
                preview=cls._masked_preview(match.group(0)),
            )
            for match in pattern.finditer(value)
        ]

    @staticmethod
    def _masked_preview(secret: str) -> str:
        if len(secret) <= 4:
            return "*" * len(secret)
        return f"{secret[:2]}{'*' * min(8, len(secret) - 4)}{secret[-2:]}"

    @classmethod
    def _walk_strings(
        cls,
        value: object,
        *,
        path: str = "$",
    ) -> Iterable[tuple[str, str]]:
        if isinstance(value, str):
            yield path, value
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield from cls._walk_strings(
                    child,
                    path=f"{path}.{key}",
                )
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from cls._walk_strings(
                    child,
                    path=f"{path}[{index}]",
                )
