from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_eval.models import (
    ExpectedOutcome,
    PaymentStatus,
    RefundCase,
    TaskType,
    ToolResult,
)

_ORDER_ID_PATTERN = re.compile(r"ORD-\d+")


@dataclass(frozen=True)
class ProposedPlan:
    order_id: str
    task_type: TaskType
    required_tools: tuple[str, ...]
    proposed_outcome: ExpectedOutcome
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class Critique:
    approved: bool
    revised_outcome: ExpectedOutcome
    issues: tuple[str, ...]
    escalation_reason: str | None = None


def extract_order_id(messages: tuple[str, ...], *, use_latest: bool) -> str:
    matches: list[str] = []
    for message in messages:
        matches.extend(_ORDER_ID_PATTERN.findall(message))
    if not matches:
        return "UNKNOWN"
    return matches[-1] if use_latest else matches[0]


def infer_task_type(message: str) -> TaskType:
    lowered = message.lower()
    if "do not refund" in lowered or "don't refund" in lowered:
        return TaskType.CANCEL_REFUND
    if "eligible" in lowered or "eligibility" in lowered:
        return TaskType.REFUND_ELIGIBILITY
    if "status" in lowered and "refund" not in lowered:
        return TaskType.ORDER_STATUS
    return TaskType.REFUND_REQUEST


class Planner:
    """Builds a plan only from user-visible conversation content."""

    def propose(self, case: RefundCase) -> ProposedPlan:
        latest_message = case.user_messages[-1]
        task_type = infer_task_type(latest_message)
        order_id = extract_order_id(case.user_messages, use_latest=True)

        if task_type is TaskType.ORDER_STATUS:
            return ProposedPlan(
                order_id=order_id,
                task_type=task_type,
                required_tools=("get_order",),
                proposed_outcome=ExpectedOutcome.INFORM,
                rationale=("latest request asks only for order status",),
            )
        if task_type is TaskType.REFUND_ELIGIBILITY:
            return ProposedPlan(
                order_id=order_id,
                task_type=task_type,
                required_tools=("get_order", "check_refund_policy"),
                proposed_outcome=ExpectedOutcome.INFORM,
                rationale=("latest request asks for eligibility, not execution",),
            )
        if task_type is TaskType.CANCEL_REFUND:
            return ProposedPlan(
                order_id=order_id,
                task_type=task_type,
                required_tools=(),
                proposed_outcome=ExpectedOutcome.NO_ACTION,
                rationale=("latest user turn withdraws the refund request",),
            )
        return ProposedPlan(
            order_id=order_id,
            task_type=task_type,
            required_tools=(
                "get_order",
                "check_refund_policy",
                "get_payment_status",
            ),
            proposed_outcome=ExpectedOutcome.REFUND,
            rationale=("refund request requires business-record verification",),
        )


class PolicyCritic:
    """Reviews a plan against tool evidence and business constraints."""

    def review(
        self,
        plan: ProposedPlan,
        *,
        order_result: ToolResult | None,
        policy_result: ToolResult | None,
        payment_result: ToolResult | None,
    ) -> Critique:
        issues: list[str] = []

        if plan.task_type is TaskType.CANCEL_REFUND:
            return Critique(
                approved=True,
                revised_outcome=ExpectedOutcome.NO_ACTION,
                issues=(),
            )

        if order_result is None or not order_result.success:
            issues.append("order could not be verified")
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.ESCALATE,
                issues=tuple(issues),
                escalation_reason="order identifier could not be verified",
            )

        if plan.task_type in {
            TaskType.ORDER_STATUS,
            TaskType.REFUND_ELIGIBILITY,
        }:
            return Critique(
                approved=True,
                revised_outcome=ExpectedOutcome.INFORM,
                issues=(),
            )

        if policy_result is None or not policy_result.success:
            issues.append("refund policy could not be verified")
        if payment_result is None or not payment_result.success:
            issues.append("payment status could not be verified")
        elif payment_result.metadata.get("stale"):
            issues.append("payment status is stale")

        if issues:
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.ESCALATE,
                issues=tuple(issues),
                escalation_reason="; ".join(issues),
            )

        # The checks above return whenever either result is absent or failed.
        # These assertions make that control-flow invariant explicit to mypy.
        assert policy_result is not None
        assert payment_result is not None

        escalation_reasons: list[str] = []
        if bool(policy_result.metadata.get("high_value")):
            escalation_reasons.append("high-value refund")
        if payment_result.metadata.get("payment_status") == PaymentStatus.DISPUTED.value:
            escalation_reasons.append("payment dispute")

        claim = str(order_result.metadata.get("customer_claim", "")).lower()
        order_delivered = bool(order_result.metadata.get("order_delivered"))
        says_not_received = "not received" in claim or "never arrived" in claim
        if says_not_received and order_delivered:
            escalation_reasons.append("customer claim conflicts with delivery record")

        if escalation_reasons:
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.ESCALATE,
                issues=tuple(escalation_reasons),
                escalation_reason="; ".join(escalation_reasons),
            )

        if not bool(policy_result.metadata.get("within_refund_window")):
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.DENY,
                issues=("outside refund window",),
            )

        return Critique(
            approved=True,
            revised_outcome=ExpectedOutcome.REFUND,
            issues=(),
        )


class Executor:
    """Turns the reviewed plan into a final business action."""

    def decide(self, critique: Critique) -> ExpectedOutcome:
        return critique.revised_outcome
