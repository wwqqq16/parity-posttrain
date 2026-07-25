from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class PaymentStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    DISPUTED = "disputed"


class ActionType(StrEnum):
    GET_ORDER = "get_order"
    CHECK_REFUND_POLICY = "check_refund_policy"
    GET_PAYMENT_STATUS = "get_payment_status"
    ISSUE_REFUND = "issue_refund"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    RESPOND = "respond"


@dataclass(frozen=True)
class RefundCase:
    case_id: str
    order_id: str
    user_message: str
    delivered_days_ago: int
    amount_cents: int
    payment_status: PaymentStatus
    customer_claim: str
    order_delivered: bool = True
    refund_window_days: int = 30
    high_value_threshold_cents: int = 100_000

    @property
    def within_refund_window(self) -> bool:
        return self.delivered_days_ago <= self.refund_window_days

    @property
    def high_value(self) -> bool:
        return self.amount_cents >= self.high_value_threshold_cents

    @property
    def claim_conflicts_with_record(self) -> bool:
        claim = self.customer_claim.lower()
        says_not_received = "not received" in claim or "never arrived" in claim
        return says_not_received and self.order_delivered


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    success: bool
    observation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryStep:
    step_index: int
    action: AgentAction
    result: ToolResult


@dataclass
class AgentRun:
    run_id: str
    case_id: str
    initial_observation: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_message: str | None = None
    completed: bool = False
    termination_reason: str | None = None

    def add_step(self, action: AgentAction, result: ToolResult) -> None:
        self.steps.append(
            TrajectoryStep(
                step_index=len(self.steps),
                action=action,
                result=result,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    case_id: str
    expected_outcome: str
    task_success: bool
    policy_violation: bool
    unnecessary_escalation: bool
    missed_escalation: bool
    invalid_tool_calls: int
    recovered_from_tool_failure: bool | None
    failure_step: int | None
    failure_type: str | None
    final_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
