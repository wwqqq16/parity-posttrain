from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from enterprise_eval.events import EventPublisher, InMemoryEventPublisher
from enterprise_eval.human_review import ReviewDecision
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    Architecture,
    EvaluationResult,
    ExpectedOutcome,
    PaymentStatus,
    RefundCase,
    TaskType,
    ToolResult,
)
from enterprise_eval.vendor_payment_cases import get_vendor_payment_case
from enterprise_eval.vendor_payment_environment import VendorPaymentEnvironment
from enterprise_eval.vendor_payment_evaluator import VendorPaymentEvaluator

# ---------------------------------------------------------------------------
# Backward-compatible refund-domain coordination interfaces
# ---------------------------------------------------------------------------

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


def extract_order_id(
    messages: tuple[str, ...],
    *,
    use_latest: bool,
) -> str:
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
    """Builds a refund-domain plan from user-visible conversation content."""

    def propose(self, case: RefundCase) -> ProposedPlan:
        latest_message = case.user_messages[-1]
        task_type = infer_task_type(latest_message)
        order_id = extract_order_id(
            case.user_messages,
            use_latest=True,
        )

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
                required_tools=(
                    "get_order",
                    "check_refund_policy",
                ),
                proposed_outcome=ExpectedOutcome.INFORM,
                rationale=(
                    "latest request asks for eligibility, not execution",
                ),
            )

        if task_type is TaskType.CANCEL_REFUND:
            return ProposedPlan(
                order_id=order_id,
                task_type=task_type,
                required_tools=(),
                proposed_outcome=ExpectedOutcome.NO_ACTION,
                rationale=(
                    "latest user turn withdraws the refund request",
                ),
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
            rationale=(
                "refund request requires business-record verification",
            ),
        )


class PolicyCritic:
    """Reviews a refund-domain plan against evidence and policy."""

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
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.ESCALATE,
                issues=("order could not be verified",),
                escalation_reason=(
                    "order identifier could not be verified"
                ),
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

        assert policy_result is not None
        assert payment_result is not None

        escalation_reasons: list[str] = []

        if bool(policy_result.metadata.get("high_value")):
            escalation_reasons.append("high-value refund")

        if (
            payment_result.metadata.get("payment_status")
            == PaymentStatus.DISPUTED.value
        ):
            escalation_reasons.append("payment dispute")

        claim = str(
            order_result.metadata.get("customer_claim", "")
        ).lower()
        order_delivered = bool(
            order_result.metadata.get("order_delivered")
        )
        says_not_received = (
            "not received" in claim
            or "never arrived" in claim
        )

        if says_not_received and order_delivered:
            escalation_reasons.append(
                "customer claim conflicts with delivery record"
            )

        if escalation_reasons:
            return Critique(
                approved=False,
                revised_outcome=ExpectedOutcome.ESCALATE,
                issues=tuple(escalation_reasons),
                escalation_reason="; ".join(escalation_reasons),
            )

        if not bool(
            policy_result.metadata.get("within_refund_window")
        ):
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
    """Turns the reviewed refund-domain plan into an outcome."""

    def decide(self, critique: Critique) -> ExpectedOutcome:
        return critique.revised_outcome


class CritiqueDecision(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


@dataclass(frozen=True)
class CoordinationStep:
    step_id: str
    action: AgentAction
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": {
                "action_type": self.action.action_type.value,
                "arguments": self.action.arguments,
            },
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PlanArtifact:
    plan_id: str
    run_id: str
    case_id: str
    phase: str
    summary: str
    steps: tuple[CoordinationStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "phase": self.phase,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class CritiqueIssue:
    code: str
    severity: str
    message: str
    step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CritiqueArtifact:
    critique_id: str
    run_id: str
    plan_id: str
    decision: CritiqueDecision
    issues: tuple[CritiqueIssue, ...]
    approved_steps: tuple[CoordinationStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "critique_id": self.critique_id,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "decision": self.decision.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "approved_steps": [
                step.to_dict() for step in self.approved_steps
            ],
        }


@dataclass(frozen=True)
class ExecutionProposal:
    proposal_id: str
    run_id: str
    plan_id: str
    critique_id: str
    approved: bool
    steps: tuple[CoordinationStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "critique_id": self.critique_id,
            "approved": self.approved,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class CoordinationSession:
    env: VendorPaymentEnvironment
    plans: list[PlanArtifact] = field(default_factory=list)
    critiques: list[CritiqueArtifact] = field(default_factory=list)
    proposals: list[ExecutionProposal] = field(default_factory=list)
    evaluation: EvaluationResult | None = None


class VendorPaymentPlanner:
    """Produces structured plans without directly dispatching tools."""

    def create_initial_plan(
        self,
        env: VendorPaymentEnvironment,
    ) -> PlanArtifact:
        assert env.run is not None
        invoice_id = env.case.invoice_id
        steps = (
            self._step(
                "collect-invoice",
                ActionType.GET_INVOICE,
                invoice_id,
                "Load the invoice before considering payment.",
            ),
            self._step(
                "verify-po",
                ActionType.VERIFY_PURCHASE_ORDER,
                invoice_id,
                "Verify that the invoice matches the purchase order.",
            ),
            self._step(
                "check-duplicate",
                ActionType.CHECK_DUPLICATE_INVOICE,
                invoice_id,
                "Reject duplicate invoices before payment.",
            ),
            self._step(
                "check-budget",
                ActionType.CHECK_BUDGET,
                invoice_id,
                "Confirm that budget is available.",
            ),
            self._step(
                "verify-bank",
                ActionType.VERIFY_VENDOR_BANK_ACCOUNT,
                invoice_id,
                "Check whether the destination account is verified.",
            ),
            self._step(
                "approve-payment",
                ActionType.APPROVE_VENDOR_PAYMENT,
                invoice_id,
                "Approve after the required evidence has been collected.",
            ),
            CoordinationStep(
                step_id="respond-approved",
                action=AgentAction(
                    ActionType.RESPOND,
                    {"message": "The validated vendor payment was approved."},
                ),
                rationale="Return the final business outcome.",
            ),
        )
        return PlanArtifact(
            plan_id=str(uuid4()),
            run_id=env.run.run_id,
            case_id=env.case.case_id,
            phase="initial",
            summary="Collect finance evidence and approve the payment.",
            steps=steps,
        )

    def create_resume_plan(
        self,
        env: VendorPaymentEnvironment,
    ) -> PlanArtifact:
        assert env.run is not None
        invoice_id = env.case.invoice_id
        steps = (
            self._step(
                "recheck-bank",
                ActionType.VERIFY_VENDOR_BANK_ACCOUNT,
                invoice_id,
                "Refresh bank-account evidence after human review.",
            ),
            self._step(
                "approve-after-review",
                ActionType.APPROVE_VENDOR_PAYMENT,
                invoice_id,
                "Approve only after review completion and workflow resume.",
            ),
            CoordinationStep(
                step_id="respond-after-review",
                action=AgentAction(
                    ActionType.RESPOND,
                    {
                        "message": (
                            "The independently verified vendor payment "
                            "was approved."
                        )
                    },
                ),
                rationale="Return the reviewed business outcome.",
            ),
        )
        return PlanArtifact(
            plan_id=str(uuid4()),
            run_id=env.run.run_id,
            case_id=env.case.case_id,
            phase="resume",
            summary="Resume the reviewed workflow and complete payment.",
            steps=steps,
        )

    @staticmethod
    def _step(
        step_id: str,
        action_type: ActionType,
        invoice_id: str,
        rationale: str,
    ) -> CoordinationStep:
        return CoordinationStep(
            step_id=step_id,
            action=AgentAction(
                action_type,
                {"invoice_id": invoice_id},
            ),
            rationale=rationale,
        )


class VendorPaymentCritic:
    """Reviews planner artifacts and may revise unsafe proposals."""

    def review(
        self,
        env: VendorPaymentEnvironment,
        plan: PlanArtifact,
    ) -> CritiqueArtifact:
        issues: list[CritiqueIssue] = []
        approved_steps = plan.steps
        decision = CritiqueDecision.APPROVE

        if plan.phase == "initial":
            approved_steps, initial_issues = self._review_initial(
                env,
                plan,
            )
            issues.extend(initial_issues)
            if issues:
                decision = CritiqueDecision.REVISE
        elif plan.phase == "resume":
            approved_steps, resume_issues, decision = self._review_resume(
                env,
                plan,
            )
            issues.extend(resume_issues)

        return CritiqueArtifact(
            critique_id=str(uuid4()),
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            decision=decision,
            issues=tuple(issues),
            approved_steps=approved_steps,
        )

    @staticmethod
    def _review_initial(
        env: VendorPaymentEnvironment,
        plan: PlanArtifact,
    ) -> tuple[tuple[CoordinationStep, ...], tuple[CritiqueIssue, ...]]:
        denial_reason = VendorPaymentCritic._static_denial_reason(env)
        if denial_reason is not None:
            issue = CritiqueIssue(
                code=denial_reason,
                severity="high",
                message="The payment must be denied based on known policy evidence.",
                step_id="approve-payment",
            )
            response = CoordinationStep(
                step_id="respond-denied",
                action=AgentAction(
                    ActionType.RESPOND,
                    {
                        "message": (
                            "The vendor payment was denied because policy "
                            "requirements were not satisfied."
                        )
                    },
                ),
                rationale="Return the policy-compliant denial.",
            )
            evidence_steps = tuple(
                step
                for step in plan.steps
                if step.action.action_type
                not in {
                    ActionType.APPROVE_VENDOR_PAYMENT,
                    ActionType.RESPOND,
                }
            )
            return (*evidence_steps, response), (issue,)

        review_required = (
            env.case.bank_account_changed
            and not env.case.bank_account_independently_verified
        )
        if not review_required:
            return plan.steps, ()

        issue = CritiqueIssue(
            code="bank_account_change_requires_human_review",
            severity="high",
            message=(
                "The planner proposed payment before independent review "
                "of the changed bank account."
            ),
            step_id="approve-payment",
        )
        evidence_steps = tuple(
            step
            for step in plan.steps
            if step.action.action_type
            not in {
                ActionType.APPROVE_VENDOR_PAYMENT,
                ActionType.RESPOND,
            }
        )
        review_step = CoordinationStep(
            step_id="request-finance-review",
            action=AgentAction(
                ActionType.REQUEST_HUMAN_REVIEW,
                {
                    "reason": (
                        "Changed vendor bank account requires independent "
                        "finance verification."
                    )
                },
            ),
            rationale=(
                "Pause the workflow and obtain independent verification."
            ),
        )
        return (*evidence_steps, review_step), (issue,)

    @staticmethod
    def _review_resume(
        env: VendorPaymentEnvironment,
        plan: PlanArtifact,
    ) -> tuple[
        tuple[CoordinationStep, ...],
        tuple[CritiqueIssue, ...],
        CritiqueDecision,
    ]:
        if env.state.human_review_rejected:
            issue = CritiqueIssue(
                code="human_review_rejected",
                severity="high",
                message="The reviewed payment must remain blocked.",
                step_id="approve-after-review",
            )
            response = CoordinationStep(
                step_id="respond-review-rejected",
                action=AgentAction(
                    ActionType.RESPOND,
                    {
                        "message": (
                            "The vendor payment was rejected after "
                            "independent review."
                        )
                    },
                ),
                rationale="Return the reviewer-approved rejection.",
            )
            return (response,), (issue,), CritiqueDecision.REVISE

        ready = (
            env.state.human_review_completed
            and env.state.human_review_approved
            and env.state.bank_account_verified_by_human
            and env.state.workflow_resumed
        )
        if not ready:
            issue = CritiqueIssue(
                code="review_state_not_ready",
                severity="high",
                message=(
                    "Payment cannot proceed until review approval, "
                    "verification, and workflow resume are complete."
                ),
                step_id="approve-after-review",
            )
            return (), (issue,), CritiqueDecision.REJECT

        return plan.steps, (), CritiqueDecision.APPROVE

    @staticmethod
    def _static_denial_reason(
        env: VendorPaymentEnvironment,
    ) -> str | None:
        if not env.case.po_matches:
            return "purchase_order_mismatch"
        if env.case.duplicate_invoice:
            return "duplicate_invoice"
        if not env.case.budget_available:
            return "insufficient_budget"
        if not env.case.authorized_approver:
            return "unauthorized_approver"
        return None


class GuardedVendorPaymentExecutor:
    """Executes critic-approved steps while rechecking runtime policy."""

    def execute(
        self,
        session: CoordinationSession,
        proposal: ExecutionProposal,
        publisher: EventPublisher,
    ) -> None:
        env = session.env
        assert env.run is not None

        if not proposal.approved:
            return

        for step in proposal.steps:
            publisher.publish(
                "agent.action.proposed",
                run_id=env.run.run_id,
                payload={
                    "proposal_id": proposal.proposal_id,
                    "step": step.to_dict(),
                },
            )

            guard = env.inspect_execution_guard(step.action)
            if guard is not None:
                env.run.add_step(step.action, guard)
                publisher.publish(
                    "guard.action.rejected",
                    run_id=env.run.run_id,
                    payload={
                        "proposal_id": proposal.proposal_id,
                        "step_id": step.step_id,
                        "action": step.action.action_type.value,
                        "reasons": guard.metadata.get("reasons", []),
                        "recommended_action": guard.metadata.get(
                            "recommended_action"
                        ),
                    },
                )
                break

            result = env.step(step.action)
            self._publish_tool_result(
                publisher,
                env.run.run_id,
                proposal.proposal_id,
                step,
                result,
            )
            if env.state.human_review_pending or env.state.terminated:
                break

    @staticmethod
    def _publish_tool_result(
        publisher: EventPublisher,
        run_id: str,
        proposal_id: str,
        step: CoordinationStep,
        result: ToolResult,
    ) -> None:
        publisher.publish(
            "tool.execution.completed",
            run_id=run_id,
            payload={
                "proposal_id": proposal_id,
                "step_id": step.step_id,
                "action": step.action.action_type.value,
                "success": result.success,
                "metadata": result.metadata,
            },
        )


class MultiAgentVendorPaymentCoordinator:
    """Planner–critic–executor workflow with resumable human review."""

    def __init__(
        self,
        publisher: EventPublisher | None = None,
    ) -> None:
        self.publisher = publisher or InMemoryEventPublisher()
        self.planner = VendorPaymentPlanner()
        self.critic = VendorPaymentCritic()
        self.executor = GuardedVendorPaymentExecutor()
        self._sessions: dict[str, CoordinationSession] = {}
        self._review_to_run: dict[str, str] = {}

    def start(self, case_id: str) -> dict[str, Any]:
        case = get_vendor_payment_case(case_id)
        env = VendorPaymentEnvironment(case)
        env.reset(
            architecture=Architecture.PLANNER_CRITIC,
            component_calls=0,
        )
        assert env.run is not None
        run_id = env.run.run_id
        session = CoordinationSession(env=env)
        self._sessions[run_id] = session

        self.publisher.publish(
            "workflow.created",
            run_id=run_id,
            payload={
                "case_id": case.case_id,
                "architecture": Architecture.PLANNER_CRITIC.value,
                "domain": "vendor_payment",
            },
        )
        self._plan_review_execute(session, phase="initial")
        self._index_review(session)
        self._evaluate_if_complete(session)
        return self.snapshot(run_id)

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

        session = self._get_session(run_id)
        result = session.env.submit_human_review(
            reviewer_id=reviewer_id,
            decision=decision,
            reason=reason,
            bank_account_verified=bank_account_verified,
        )
        if not result.success:
            raise ValueError(result.observation)

        self.publisher.publish(
            "review.completed",
            run_id=run_id,
            payload={
                "review_id": review_id,
                "decision": decision.value,
                "reviewer_id": reviewer_id,
                "bank_account_verified": bank_account_verified,
            },
        )
        return self.snapshot(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        session = self._get_session(run_id)
        result = session.env.resume_after_human_review()
        if not result.success:
            raise ValueError(result.observation)

        self.publisher.publish(
            "workflow.resumed",
            run_id=run_id,
            payload={
                "review_status": result.metadata.get("review_status"),
                "recommended_action": result.metadata.get(
                    "recommended_action"
                ),
            },
        )
        self._plan_review_execute(session, phase="resume")
        self._evaluate_if_complete(session)
        return self.snapshot(run_id)

    def snapshot(self, run_id: str) -> dict[str, Any]:
        session = self._get_session(run_id)
        env = session.env
        assert env.run is not None
        return {
            "run_id": run_id,
            "case_id": env.case.case_id,
            "status": self._status(session),
            "component_calls": env.run.component_calls,
            "state": asdict(env.state),
            "human_review": (
                env.human_review.to_dict()
                if env.human_review is not None
                else None
            ),
            "plans": [plan.to_dict() for plan in session.plans],
            "critiques": [
                critique.to_dict() for critique in session.critiques
            ],
            "proposals": [
                proposal.to_dict() for proposal in session.proposals
            ],
            "evaluation": (
                session.evaluation.to_dict()
                if session.evaluation is not None
                else None
            ),
            "events": [
                event.to_dict()
                for event in self.publisher.list_events(run_id=run_id)
            ],
        }

    def _plan_review_execute(
        self,
        session: CoordinationSession,
        *,
        phase: str,
    ) -> None:
        env = session.env
        assert env.run is not None
        plan = (
            self.planner.create_initial_plan(env)
            if phase == "initial"
            else self.planner.create_resume_plan(env)
        )
        env.run.component_calls += 1
        session.plans.append(plan)
        self.publisher.publish(
            "agent.plan.created",
            run_id=env.run.run_id,
            payload=plan.to_dict(),
        )

        critique = self.critic.review(env, plan)
        env.run.component_calls += 1
        session.critiques.append(critique)
        self.publisher.publish(
            "critic.review.completed",
            run_id=env.run.run_id,
            payload=critique.to_dict(),
        )

        proposal = ExecutionProposal(
            proposal_id=str(uuid4()),
            run_id=env.run.run_id,
            plan_id=plan.plan_id,
            critique_id=critique.critique_id,
            approved=critique.decision is not CritiqueDecision.REJECT,
            steps=critique.approved_steps,
        )
        session.proposals.append(proposal)
        env.run.component_calls += 1
        self.publisher.publish(
            "execution.proposal.created",
            run_id=env.run.run_id,
            payload=proposal.to_dict(),
        )
        self.executor.execute(session, proposal, self.publisher)

    def _index_review(self, session: CoordinationSession) -> None:
        env = session.env
        if env.human_review is None:
            return
        assert env.run is not None
        self._review_to_run[env.human_review.review_id] = env.run.run_id
        self.publisher.publish(
            "review.requested",
            run_id=env.run.run_id,
            payload={
                "review_id": env.human_review.review_id,
                "reason": env.human_review.requested_reason,
            },
        )

    def _evaluate_if_complete(
        self,
        session: CoordinationSession,
    ) -> None:
        env = session.env
        if env.run is None or not env.run.completed:
            return
        evaluation = VendorPaymentEvaluator().evaluate(env)
        session.evaluation = evaluation
        self.publisher.publish(
            "evaluation.completed",
            run_id=env.run.run_id,
            payload=evaluation.to_dict(),
        )

    def _get_session(self, run_id: str) -> CoordinationSession:
        try:
            return self._sessions[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown run_id={run_id!r}") from exc

    @staticmethod
    def _status(session: CoordinationSession) -> str:
        env = session.env
        if session.evaluation is not None:
            return "completed"
        if env.state.human_review_pending:
            return "review_required"
        if env.state.human_review_completed and not env.state.workflow_resumed:
            return "review_completed"
        return "running"
