from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from enterprise_eval.events import EventPublisher, InMemoryEventPublisher
from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    EvaluationResult,
    ToolResult,
)
from enterprise_eval.vendor_payment_cases import get_vendor_payment_case
from enterprise_eval.vendor_payment_environment import VendorPaymentEnvironment
from enterprise_eval.vendor_payment_evaluator import VendorPaymentEvaluator


@dataclass
class WorkflowRecord:
    env: VendorPaymentEnvironment
    evaluation: EvaluationResult | None = None


class VendorPaymentWorkflowService:
    """Synchronous control-plane service for the vendor-payment vertical slice."""

    def __init__(self, publisher: EventPublisher | None = None) -> None:
        self.publisher = publisher or InMemoryEventPublisher()
        self._runs: dict[str, WorkflowRecord] = {}
        self._review_to_run: dict[str, str] = {}

    def create_workflow(self, case_id: str) -> dict[str, Any]:
        case = get_vendor_payment_case(case_id)
        env = VendorPaymentEnvironment(case)
        initial_observation = env.reset()
        assert env.run is not None
        run_id = env.run.run_id
        self._runs[run_id] = WorkflowRecord(env=env)

        self._emit(
            "workflow.created",
            run_id,
            {
                "case_id": case.case_id,
                "domain": "vendor_payment",
                "initial_observation": initial_observation,
            },
        )

        self._collect_evidence(env)
        payment_action = self._payment_action(env)
        guard = env.inspect_execution_guard(payment_action)

        if guard is None:
            payment = env.step(payment_action)
            self._emit_tool_result(env, payment_action, payment)
            env.step(
                AgentAction(
                    ActionType.RESPOND,
                    {"message": "The validated vendor payment was approved."},
                )
            )
            self._complete_evaluation(run_id)
        else:
            env.run.add_step(payment_action, guard)
            self._emit(
                "guard.action.rejected",
                run_id,
                {
                    "action": payment_action.action_type.value,
                    "reasons": guard.metadata.get("reasons", []),
                    "recommended_action": guard.metadata.get(
                        "recommended_action"
                    ),
                },
            )
            if guard.metadata.get("recommended_action") == "request_human_review":
                review_result = env.step(
                    AgentAction(
                        ActionType.REQUEST_HUMAN_REVIEW,
                        {
                            "reason": (
                                "Changed vendor bank account requires "
                                "independent finance verification."
                            )
                        },
                    )
                )
                self._emit_tool_result(
                    env,
                    AgentAction(
                        ActionType.REQUEST_HUMAN_REVIEW,
                        {
                            "reason": (
                                "Changed vendor bank account requires "
                                "independent finance verification."
                            )
                        },
                    ),
                    review_result,
                )
                assert env.human_review is not None
                self._review_to_run[env.human_review.review_id] = run_id
                self._emit(
                    "review.requested",
                    run_id,
                    {
                        "review_id": env.human_review.review_id,
                        "reason": env.human_review.requested_reason,
                    },
                )

        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        record = self._get_record(run_id)
        env = record.env
        assert env.run is not None
        return {
            "run_id": run_id,
            "case_id": env.case.case_id,
            "status": self._status(record),
            "state": asdict(env.state),
            "human_review": (
                env.human_review.to_dict()
                if env.human_review is not None
                else None
            ),
            "evaluation": (
                record.evaluation.to_dict()
                if record.evaluation is not None
                else None
            ),
            "events": [
                event.to_dict()
                for event in self.publisher.list_events(run_id=run_id)
            ],
        }

    def list_reviews(self) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for record in self._runs.values():
            review = record.env.human_review
            if review is not None:
                reviews.append(review.to_dict())
        return reviews

    def submit_review(
        self,
        review_id: str,
        *,
        decision: ReviewDecision,
        reviewer_id: str,
        reason: str,
        bank_account_verified: bool = False,
    ) -> dict[str, Any]:
        run_id = self._review_to_run.get(review_id)
        if run_id is None:
            raise KeyError(f"Unknown review_id={review_id!r}")

        record = self._get_record(run_id)
        result = record.env.submit_human_review(
            reviewer_id=reviewer_id,
            decision=decision,
            reason=reason,
            bank_account_verified=bank_account_verified,
        )
        if not result.success:
            raise ValueError(result.observation)

        self._emit(
            "review.completed",
            run_id,
            {
                "review_id": review_id,
                "decision": decision.value,
                "reviewer_id": reviewer_id,
                "bank_account_verified": bank_account_verified,
            },
        )
        return self.get_run(run_id)

    def resume_workflow(self, run_id: str) -> dict[str, Any]:
        record = self._get_record(run_id)
        env = record.env
        result = env.resume_after_human_review()
        if not result.success:
            raise ValueError(result.observation)

        self._emit(
            "workflow.resumed",
            run_id,
            {
                "review_status": result.metadata.get("review_status"),
                "recommended_action": result.metadata.get(
                    "recommended_action"
                ),
            },
        )

        if env.state.human_review_approved:
            payment_action = self._payment_action(env)
            guard = env.inspect_execution_guard(payment_action)
            if guard is not None:
                raise RuntimeError(guard.observation)
            payment = env.step(payment_action)
            self._emit_tool_result(env, payment_action, payment)
            env.step(
                AgentAction(
                    ActionType.RESPOND,
                    {
                        "message": (
                            "The independently verified vendor payment "
                            "was approved."
                        )
                    },
                )
            )
        else:
            env.step(
                AgentAction(
                    ActionType.RESPOND,
                    {
                        "message": (
                            "The vendor payment was rejected after "
                            "independent review."
                        )
                    },
                )
            )

        self._complete_evaluation(run_id)
        return self.get_run(run_id)

    def _collect_evidence(self, env: VendorPaymentEnvironment) -> None:
        for action_type in (
            ActionType.GET_INVOICE,
            ActionType.VERIFY_PURCHASE_ORDER,
            ActionType.CHECK_DUPLICATE_INVOICE,
            ActionType.CHECK_BUDGET,
            ActionType.VERIFY_VENDOR_BANK_ACCOUNT,
        ):
            action = AgentAction(
                action_type,
                {"invoice_id": env.case.invoice_id},
            )
            result = env.step(action)
            self._emit_tool_result(env, action, result)

    def _emit_tool_result(
        self,
        env: VendorPaymentEnvironment,
        action: AgentAction,
        result: ToolResult,
    ) -> None:
        assert env.run is not None
        self._emit(
            "tool.execution.completed",
            env.run.run_id,
            {
                "action": action.action_type.value,
                "success": result.success,
                "metadata": result.metadata,
            },
        )

    def _complete_evaluation(self, run_id: str) -> None:
        record = self._get_record(run_id)
        evaluation = VendorPaymentEvaluator().evaluate(record.env)
        record.evaluation = evaluation
        self._emit(
            "evaluation.completed",
            run_id,
            evaluation.to_dict(),
        )

    @staticmethod
    def _payment_action(
        env: VendorPaymentEnvironment,
    ) -> AgentAction:
        return AgentAction(
            ActionType.APPROVE_VENDOR_PAYMENT,
            {"invoice_id": env.case.invoice_id},
        )

    def _emit(
        self,
        event_type: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.publisher.publish(
            event_type,
            run_id=run_id,
            payload=payload,
        )

    def _get_record(self, run_id: str) -> WorkflowRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run_id={run_id!r}") from exc

    @staticmethod
    def _status(record: WorkflowRecord) -> str:
        env = record.env
        if record.evaluation is not None:
            return "completed"
        if env.state.human_review_pending:
            return "review_required"
        if env.state.human_review_completed and not env.state.workflow_resumed:
            return "review_completed"
        return "running"
