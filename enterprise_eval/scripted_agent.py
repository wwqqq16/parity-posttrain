from __future__ import annotations

from enterprise_eval.coordination import (
    Executor,
    Planner,
    PolicyCritic,
    extract_order_id,
    infer_task_type,
)
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    Architecture,
    ExpectedOutcome,
    PaymentStatus,
    TaskType,
    ToolResult,
)


class SingleAgentBaseline:
    """A one-pass baseline that follows the first intent and first identifier."""

    def run(self, env: RefundEnvironment) -> None:
        env.reset(architecture=Architecture.SINGLE, component_calls=1)
        task_type = infer_task_type(env.case.user_messages[0])
        order_id = extract_order_id(env.case.user_messages, use_latest=False)

        if task_type is TaskType.ORDER_STATUS:
            order = env.step(AgentAction(ActionType.GET_ORDER, {"order_id": order_id}))
            message = order.observation if order.success else "I could not locate the order."
            env.step(AgentAction(ActionType.RESPOND, {"message": message}))
            return

        if task_type is TaskType.REFUND_ELIGIBILITY:
            order = env.step(AgentAction(ActionType.GET_ORDER, {"order_id": order_id}))
            if not order.success:
                env.step(
                    AgentAction(
                        ActionType.RESPOND,
                        {"message": "I could not locate the order."},
                    )
                )
                return
            policy = env.step(
                AgentAction(ActionType.CHECK_REFUND_POLICY, {"order_id": order_id})
            )
            eligible = bool(policy.metadata.get("within_refund_window"))
            env.step(
                AgentAction(
                    ActionType.RESPOND,
                    {"message": f"Refund eligibility within policy window: {eligible}."},
                )
            )
            return

        order = env.step(AgentAction(ActionType.GET_ORDER, {"order_id": order_id}))
        if not order.success:
            env.step(
                AgentAction(
                    ActionType.RESPOND,
                    {"message": "I could not locate the order."},
                )
            )
            return

        policy = env.step(
            AgentAction(ActionType.CHECK_REFUND_POLICY, {"order_id": order_id})
        )
        payment = self._get_payment_status_with_retry(env, order_id)

        if not payment.success:
            self._escalate_and_respond(
                env,
                "payment status could not be verified after retry",
            )
            return

        should_escalate = (
            bool(policy.metadata.get("high_value"))
            or payment.metadata.get("payment_status") == PaymentStatus.DISPUTED.value
            or self._claim_conflicts(order)
        )
        if should_escalate:
            self._escalate_and_respond(
                env,
                self._escalation_reason(policy, payment, order),
            )
            return

        if bool(policy.metadata.get("within_refund_window")) and (
            payment.metadata.get("payment_status") == PaymentStatus.SETTLED.value
        ):
            refund = env.step(
                AgentAction(ActionType.ISSUE_REFUND, {"order_id": order_id})
            )
            message = (
                "Your refund has been issued."
                if refund.success
                else "I could not issue the refund and will not claim that it succeeded."
            )
            env.step(AgentAction(ActionType.RESPOND, {"message": message}))
            return

        env.step(
            AgentAction(
                ActionType.RESPOND,
                {
                    "message": (
                        "This order is not eligible for an automatic refund under "
                        "the current policy."
                    )
                },
            )
        )

    @staticmethod
    def _get_payment_status_with_retry(
        env: RefundEnvironment,
        order_id: str,
    ) -> ToolResult:
        payment = env.step(
            AgentAction(ActionType.GET_PAYMENT_STATUS, {"order_id": order_id})
        )
        if not payment.success and payment.metadata.get("retryable"):
            payment = env.step(
                AgentAction(ActionType.GET_PAYMENT_STATUS, {"order_id": order_id})
            )
        return payment

    @staticmethod
    def _escalate_and_respond(env: RefundEnvironment, reason: str) -> None:
        env.step(
            AgentAction(
                ActionType.REQUEST_HUMAN_REVIEW,
                {"reason": reason},
            )
        )
        env.step(
            AgentAction(
                ActionType.RESPOND,
                {
                    "message": (
                        "I have sent this case for human review before any "
                        "irreversible action is taken."
                    )
                },
            )
        )

    @staticmethod
    def _escalation_reason(
        policy: ToolResult,
        payment: ToolResult,
        order: ToolResult,
    ) -> str:
        reasons: list[str] = []
        if bool(policy.metadata.get("high_value")):
            reasons.append("high-value refund")
        if payment.metadata.get("payment_status") == PaymentStatus.DISPUTED.value:
            reasons.append("payment dispute")
        if SingleAgentBaseline._claim_conflicts(order):
            reasons.append("customer claim conflicts with delivery record")
        return "; ".join(reasons) or "business-risk review required"

    @staticmethod
    def _claim_conflicts(order: ToolResult) -> bool:
        claim = str(order.metadata.get("customer_claim", "")).lower()
        says_not_received = "not received" in claim or "never arrived" in claim
        return says_not_received and bool(order.metadata.get("order_delivered"))


class PlannerCriticAgent:
    """A lightweight planner -> policy critic -> executor architecture."""

    def __init__(self) -> None:
        self.planner = Planner()
        self.critic = PolicyCritic()
        self.executor = Executor()

    def run(self, env: RefundEnvironment) -> None:
        env.reset(architecture=Architecture.PLANNER_CRITIC, component_calls=3)
        plan = self.planner.propose(env.case)

        order_result: ToolResult | None = None
        policy_result: ToolResult | None = None
        payment_result: ToolResult | None = None

        if "get_order" in plan.required_tools:
            order_result = env.step(
                AgentAction(ActionType.GET_ORDER, {"order_id": plan.order_id})
            )
        if "check_refund_policy" in plan.required_tools and (
            order_result is not None and order_result.success
        ):
            policy_result = env.step(
                AgentAction(
                    ActionType.CHECK_REFUND_POLICY,
                    {"order_id": plan.order_id},
                )
            )
        if "get_payment_status" in plan.required_tools and (
            order_result is not None and order_result.success
        ):
            payment_result = self._get_verified_payment_status(env, plan.order_id)

        critique = self.critic.review(
            plan,
            order_result=order_result,
            policy_result=policy_result,
            payment_result=payment_result,
        )
        outcome = self.executor.decide(critique)

        assert env.run is not None
        env.run.metadata["coordination"] = {
            "plan": {
                "order_id": plan.order_id,
                "task_type": plan.task_type.value,
                "required_tools": list(plan.required_tools),
                "proposed_outcome": plan.proposed_outcome.value,
                "rationale": list(plan.rationale),
            },
            "critique": {
                "approved": critique.approved,
                "revised_outcome": critique.revised_outcome.value,
                "issues": list(critique.issues),
                "escalation_reason": critique.escalation_reason,
            },
            "executor_outcome": outcome.value,
        }

        self._execute_outcome(
            env,
            order_id=plan.order_id,
            outcome=outcome,
            critique_reason=critique.escalation_reason,
            order_result=order_result,
            policy_result=policy_result,
            task_type=plan.task_type,
        )

    @staticmethod
    def _get_verified_payment_status(
        env: RefundEnvironment,
        order_id: str,
    ) -> ToolResult:
        payment = env.step(
            AgentAction(ActionType.GET_PAYMENT_STATUS, {"order_id": order_id})
        )
        if not payment.success and payment.metadata.get("retryable"):
            payment = env.step(
                AgentAction(ActionType.GET_PAYMENT_STATUS, {"order_id": order_id})
            )
        if payment.success and payment.metadata.get("stale"):
            payment = env.step(
                AgentAction(ActionType.GET_PAYMENT_STATUS, {"order_id": order_id})
            )
        return payment

    @staticmethod
    def _execute_outcome(
        env: RefundEnvironment,
        *,
        order_id: str,
        outcome: ExpectedOutcome,
        critique_reason: str | None,
        order_result: ToolResult | None,
        policy_result: ToolResult | None,
        task_type: TaskType,
    ) -> None:
        if outcome is ExpectedOutcome.REFUND:
            refund = env.step(
                AgentAction(ActionType.ISSUE_REFUND, {"order_id": order_id})
            )
            message = (
                "Your refund has been issued."
                if refund.success
                else "The refund was blocked, so no success claim was made."
            )
        elif outcome is ExpectedOutcome.ESCALATE:
            env.step(
                AgentAction(
                    ActionType.REQUEST_HUMAN_REVIEW,
                    {"reason": critique_reason or "policy review required"},
                )
            )
            message = "I sent the case for human review before taking action."
        elif outcome is ExpectedOutcome.DENY:
            message = "This order is outside the automatic refund policy."
        elif outcome is ExpectedOutcome.NO_ACTION:
            message = "I will not issue a refund because your latest request withdrew it."
        elif task_type is TaskType.ORDER_STATUS:
            message = (
                order_result.observation
                if order_result is not None
                else "The order status could not be retrieved."
            )
        else:
            eligible = bool(
                policy_result is not None
                and policy_result.metadata.get("within_refund_window")
            )
            message = f"Refund eligibility within policy window: {eligible}."

        env.step(AgentAction(ActionType.RESPOND, {"message": message}))


class SolvabilityOracleAgent:
    """Resolve generated cases with the task factory's two-attempt contract."""

    max_attempts = 2

    def run(self, env: RefundEnvironment) -> None:
        env.reset(architecture=Architecture.ORACLE, component_calls=1)
        order_id = extract_order_id(
            env.case.user_messages,
            use_latest=True,
        )

        order = self._call_with_retry(
            env,
            AgentAction(
                ActionType.GET_ORDER,
                {"order_id": order_id},
            ),
        )
        if not order.success:
            self._escalate(env, "order lookup exhausted the retry budget")
            return

        policy = self._call_with_retry(
            env,
            AgentAction(
                ActionType.CHECK_REFUND_POLICY,
                {"order_id": order_id},
            ),
        )
        if not policy.success:
            self._escalate(env, "policy lookup exhausted the retry budget")
            return

        payment = self._call_with_retry(
            env,
            AgentAction(
                ActionType.GET_PAYMENT_STATUS,
                {"order_id": order_id},
            ),
        )
        if not payment.success:
            self._escalate(env, "payment lookup exhausted the retry budget")
            return

        refund = env.step(
            AgentAction(
                ActionType.ISSUE_REFUND,
                {"order_id": order_id},
            )
        )
        message = (
            "Your refund has been issued."
            if refund.success
            else "The refund was blocked and no success claim was made."
        )
        env.step(
            AgentAction(
                ActionType.RESPOND,
                {"message": message},
            )
        )

    def _call_with_retry(
        self,
        env: RefundEnvironment,
        action: AgentAction,
    ) -> ToolResult:
        result = env.step(action)
        attempts = 1
        while (
            not result.success
            and result.metadata.get("retryable")
            and attempts < self.max_attempts
        ):
            result = env.step(action)
            attempts += 1
        return result

    @staticmethod
    def _escalate(
        env: RefundEnvironment,
        reason: str,
    ) -> None:
        env.step(
            AgentAction(
                ActionType.REQUEST_HUMAN_REVIEW,
                {"reason": reason},
            )
        )
        env.step(
            AgentAction(
                ActionType.RESPOND,
                {
                    "message": (
                        "I sent the case for human review after "
                        "the retry budget was exhausted."
                    )
                },
            )
        )


class UnsafePromptFollowingAgent:
    """An intentionally unsafe baseline used to verify failure detection."""

    def run(self, env: RefundEnvironment) -> None:
        env.reset(architecture=Architecture.UNSAFE, component_calls=1)
        order_id = env.case.order_id
        env.step(AgentAction(ActionType.ISSUE_REFUND, {"order_id": order_id}))
        env.step(
            AgentAction(
                ActionType.RESPOND,
                {"message": "Your refund has been issued as requested."},
            )
        )


# Backward-compatible name used by Milestones 1 and 2.
ScriptedRefundAgent = SingleAgentBaseline
