from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class PaymentStatus(StrEnum):
    SETTLED = "settled"
    PENDING = "pending"
    DISPUTED = "disputed"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskType(StrEnum):
    ORDER_STATUS = "order_status"
    REFUND_ELIGIBILITY = "refund_eligibility"
    REFUND_REQUEST = "refund_request"
    CANCEL_REFUND = "cancel_refund"


class ExpectedOutcome(StrEnum):
    INFORM = "inform"
    REFUND = "refund"
    DENY = "deny"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


class Architecture(StrEnum):
    SINGLE = "single"
    PLANNER_CRITIC = "planner-critic"
    ORACLE = "oracle"
    UNSAFE = "unsafe"


class FailureProfile(StrEnum):
    NONE = "none"
    TRANSIENT_TOOL_TIMEOUT = "transient_tool_timeout"
    PERSISTENT_TOOL_TIMEOUT = "persistent_tool_timeout"


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
    user_messages: tuple[str, ...]
    delivered_days_ago: int
    amount_cents: int
    payment_status: PaymentStatus
    customer_claim: str
    task_type: TaskType
    expected_outcome: ExpectedOutcome
    difficulty: Difficulty
    risk_level: RiskLevel
    order_delivered: bool = True
    refund_window_days: int = 30
    high_value_threshold_cents: int = 100_000
    payment_status_failures_before_success: int = 0
    stale_payment_status_once: bool = False
    user_attempts_policy_override: bool = False
    initial_order_id: str | None = None
    corrected_order_id: str | None = None
    initial_task_type: TaskType | None = None
    injected_failures: tuple[str, ...] = ()
    failure_profile: FailureProfile = FailureProfile.NONE
    failure_injection_step: int | None = None
    failure_injection_action: ActionType | None = None
    failure_injection_count: int = 0
    factory_seed: int | None = None
    factory_variant: int | None = None

    @property
    def user_message(self) -> str:
        return self.user_messages[-1]

    @property
    def conversation(self) -> str:
        return "\n".join(
            f"User turn {index + 1}: {message}"
            for index, message in enumerate(self.user_messages)
        )

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

    @property
    def first_order_id(self) -> str:
        return self.initial_order_id or self.order_id

    @property
    def latest_order_id(self) -> str:
        return self.corrected_order_id or self.initial_order_id or self.order_id

    @property
    def first_task_type(self) -> TaskType:
        return self.initial_task_type or self.task_type


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
    architecture: str = Architecture.SINGLE.value
    component_calls: int = 0
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_message: str | None = None
    completed: bool = False
    termination_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
    correct_escalation: bool
    invalid_tool_calls: int
    transient_tool_failures: int
    recovered_from_tool_failure: bool | None
    prompt_injection_resisted: bool | None
    failure_step: int | None
    failure_type: str | None
    final_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
