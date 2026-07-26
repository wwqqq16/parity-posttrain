from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class HumanReviewRecord:
    review_id: str
    run_id: str
    case_id: str
    requested_reason: str
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: str | None = None
    decision_reason: str | None = None
    bank_account_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
