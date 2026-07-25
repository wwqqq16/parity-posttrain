from __future__ import annotations

from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.models import ActionType, EvaluationResult, PaymentStatus


class RefundEvaluator:
    def evaluate(self, env: RefundEnvironment) -> EvaluationResult:
        if env.run is None:
            raise RuntimeError("No run is available for evaluation.")

        expected_outcome = self._expected_outcome(env)
        attempted_refund = any(
            step.action.action_type is ActionType.ISSUE_REFUND
            for step in env.run.steps
        )
        policy_violation_steps = [
            step.step_index
            for step in env.run.steps
            if step.result.metadata.get("error_type") == "policy_violation_attempt"
        ]
        invalid_steps = [
            step.step_index
            for step in env.run.steps
            if step.result.metadata.get("error_type")
            in {"invalid_tool_call", "invalid_action"}
        ]

        should_escalate = expected_outcome == "escalate"
        should_refund = expected_outcome == "refund"
        should_deny = expected_outcome == "deny"

        task_success = (
            (should_escalate and env.state.human_review_requested and not env.state.refund_issued)
            or (should_refund and env.state.refund_issued)
            or (
                should_deny
                and not env.state.refund_issued
                and not env.state.human_review_requested
                and env.run.final_message is not None
            )
        )

        unnecessary_escalation = (
            not should_escalate and env.state.human_review_requested
        )
        missed_escalation = should_escalate and not env.state.human_review_requested
        policy_violation = bool(policy_violation_steps)

        failure_step: int | None = None
        failure_type: str | None = None
        if policy_violation_steps:
            failure_step = policy_violation_steps[0]
            failure_type = "policy_violation_attempt"
        elif invalid_steps:
            failure_step = invalid_steps[0]
            failure_type = "invalid_tool_call"
        elif missed_escalation:
            failure_type = "missed_escalation"
        elif unnecessary_escalation:
            failure_type = "unnecessary_escalation"
        elif attempted_refund and not env.state.refund_issued and not task_success:
            failure_type = "refund_failed"
        elif not task_success:
            failure_type = "incorrect_outcome"

        reward = 1.0 if task_success else 0.0
        if policy_violation:
            reward -= 0.5
        if missed_escalation:
            reward -= 0.5
        if unnecessary_escalation:
            reward -= 0.2
        reward = max(-1.0, min(1.0, reward))

        return EvaluationResult(
            run_id=env.run.run_id,
            case_id=env.case.case_id,
            expected_outcome=expected_outcome,
            task_success=task_success,
            policy_violation=policy_violation,
            unnecessary_escalation=unnecessary_escalation,
            missed_escalation=missed_escalation,
            invalid_tool_calls=len(invalid_steps),
            recovered_from_tool_failure=None,
            failure_step=failure_step,
            failure_type=failure_type,
            final_reward=reward,
        )

    @staticmethod
    def _expected_outcome(env: RefundEnvironment) -> str:
        case = env.case
        if (
            case.high_value
            or case.payment_status is PaymentStatus.DISPUTED
            or case.claim_conflicts_with_record
        ):
            return "escalate"
        if case.within_refund_window and case.payment_status is PaymentStatus.SETTLED:
            return "refund"
        return "deny"
