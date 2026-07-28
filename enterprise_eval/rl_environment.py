"""Gym-style control plane for the deterministic enterprise environment."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from enterprise_eval.environment import EnvironmentState, RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    AgentRun,
    Architecture,
    ExpectedOutcome,
    RefundCase,
    ToolResult,
    TrajectoryStep,
)


@dataclass(frozen=True)
class RLResetResult:
    """Observation and audit metadata returned by reset."""

    observation: str
    state_fingerprint: str
    info: dict[str, Any]


@dataclass(frozen=True)
class RLStepResult:
    """One online environment transition."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    state_fingerprint: str
    info: dict[str, Any]


@dataclass(frozen=True)
class RLEnvironmentSnapshot:
    """Restorable in-process state with an integrity fingerprint."""

    case_id: str
    case_fingerprint: str
    state: EnvironmentState
    run: AgentRun
    truncated: bool
    state_fingerprint: str


class EnterpriseRefundRLEnvironment:
    """Expose reset/step/replay semantics for enterprise-agent RL."""

    def __init__(
        self,
        case: RefundCase,
        *,
        enforce_execution_guard: bool = True,
        max_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        self.case = case
        self.enforce_execution_guard = enforce_execution_guard
        self.max_steps = max_steps
        self._environment = RefundEnvironment(case)
        self._truncated = False

    @property
    def environment(self) -> RefundEnvironment:
        """Return the wrapped environment for evaluation and inspection."""

        return self._environment

    def reset(
        self,
        *,
        architecture: Architecture = Architecture.SINGLE,
    ) -> RLResetResult:
        """Reset to a deterministic initial state."""

        self._truncated = False
        observation = self._environment.reset(
            architecture=architecture,
            component_calls=0,
        )
        fingerprint = self.state_fingerprint()
        return RLResetResult(
            observation=observation,
            state_fingerprint=fingerprint,
            info={
                "case_id": self.case.case_id,
                "architecture": architecture.value,
                "expected_outcome": self.case.expected_outcome.value,
                "execution_guard_enabled": self.enforce_execution_guard,
            },
        )

    def step(
        self,
        action: AgentAction,
    ) -> RLStepResult:
        """Apply one action and return dense online reward components."""

        run = self._require_active_run()
        if run.completed or self._environment.state.terminated:
            raise RuntimeError("cannot step a terminated episode")
        if self._truncated:
            raise RuntimeError("cannot step a truncated episode")

        before_potential = _evidence_potential(
            self._environment.state
        )
        previous_step = (
            run.steps[-1]
            if run.steps
            else None
        )
        guard_result = (
            self._environment.inspect_execution_guard(
                action
            )
            if self.enforce_execution_guard
            else None
        )
        if guard_result is not None:
            run.add_step(action, guard_result)
            result = guard_result
        else:
            result = self._environment.step(action)

        after_potential = _evidence_potential(
            self._environment.state
        )
        terminated = self._environment.state.terminated
        self._truncated = (
            not terminated
            and self.max_steps is not None
            and len(run.steps) >= self.max_steps
        )
        reward_components = _online_reward_components(
            action=action,
            result=result,
            expected_outcome=self.case.expected_outcome,
            previous_step=previous_step,
            potential_delta=(
                after_potential - before_potential
            ),
            terminal_reward=(
                RefundEvaluator()
                .evaluate(self._environment)
                .final_reward
                if terminated
                else None
            ),
            truncated=self._truncated,
        )
        reward = round(
            sum(reward_components.values()),
            10,
        )
        fingerprint = self.state_fingerprint()
        return RLStepResult(
            observation=result.observation,
            reward=reward,
            terminated=terminated,
            truncated=self._truncated,
            state_fingerprint=fingerprint,
            info={
                "case_id": self.case.case_id,
                "step_index": len(run.steps) - 1,
                "action_type": action.action_type.value,
                "tool_success": result.success,
                "tool_metadata": copy.deepcopy(
                    result.metadata
                ),
                "reward_components": reward_components,
                "expected_outcome": (
                    self.case.expected_outcome.value
                ),
                "execution_guard_enabled": (
                    self.enforce_execution_guard
                ),
            },
        )

    def snapshot(self) -> RLEnvironmentSnapshot:
        """Capture restorable state without relying on process serialization."""

        run = self._require_run()
        return RLEnvironmentSnapshot(
            case_id=self.case.case_id,
            case_fingerprint=_case_fingerprint(
                self.case
            ),
            state=copy.deepcopy(
                self._environment.state
            ),
            run=copy.deepcopy(run),
            truncated=self._truncated,
            state_fingerprint=self.state_fingerprint(),
        )

    def restore(
        self,
        snapshot: RLEnvironmentSnapshot,
    ) -> str:
        """Restore a snapshot and verify its deterministic fingerprint."""

        if snapshot.case_id != self.case.case_id:
            raise ValueError(
                "snapshot case does not match environment case"
            )
        if snapshot.case_fingerprint != _case_fingerprint(
            self.case
        ):
            raise ValueError(
                "snapshot case configuration does not match"
            )

        self._environment.state = copy.deepcopy(
            snapshot.state
        )
        self._environment.run = copy.deepcopy(snapshot.run)
        self._truncated = snapshot.truncated
        restored_fingerprint = self.state_fingerprint()
        if (
            restored_fingerprint
            != snapshot.state_fingerprint
        ):
            raise ValueError(
                "restored snapshot fingerprint does not match"
            )
        return restored_fingerprint

    def state_fingerprint(self) -> str:
        """Hash causal episode state while excluding random run IDs."""

        run = self._require_run()
        payload = {
            "case": asdict(self.case),
            "architecture": run.architecture,
            "state": asdict(self._environment.state),
            "steps": [
                {
                    "step_index": step.step_index,
                    "action": asdict(step.action),
                    "result": asdict(step.result),
                }
                for step in run.steps
            ],
            "final_message": run.final_message,
            "completed": run.completed,
            "termination_reason": run.termination_reason,
            "truncated": self._truncated,
        }
        encoded = json.dumps(
            payload,
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _require_run(self) -> AgentRun:
        if self._environment.run is None:
            raise RuntimeError(
                "reset must be called before using the environment"
            )
        return self._environment.run

    def _require_active_run(self) -> AgentRun:
        return self._require_run()


def _online_reward_components(
    *,
    action: AgentAction,
    result: ToolResult,
    expected_outcome: ExpectedOutcome,
    previous_step: TrajectoryStep | None,
    potential_delta: float,
    terminal_reward: float | None,
    truncated: bool,
) -> dict[str, float]:
    components: dict[str, float] = {}
    if potential_delta:
        components["evidence_progress"] = round(
            potential_delta,
            10,
        )

    error_type = result.metadata.get("error_type")
    if error_type == "policy_violation_attempt":
        components["policy_penalty"] = -1.0
    elif error_type == "execution_guard_rejection":
        components["guard_rejection_penalty"] = -0.5
    elif error_type in {
        "invalid_tool_call",
        "invalid_action",
    }:
        components["invalid_action_penalty"] = -0.25

    previous_error_type: object | None = None
    previous_action_type: ActionType | None = None
    if previous_step is not None:
        previous_error_type = (
            previous_step.result.metadata.get(
                "error_type"
            )
        )
        previous_action_type = (
            previous_step.action.action_type
        )
    if (
        previous_error_type
        == "transient_tool_failure"
        and previous_action_type is action.action_type
        and result.success
    ):
        components["recovery_credit"] = 0.25

    if (
        action.action_type
        is ActionType.REQUEST_HUMAN_REVIEW
        and result.success
        and expected_outcome is ExpectedOutcome.ESCALATE
    ):
        components["correct_escalation_credit"] = 0.25
    if (
        action.action_type is ActionType.ISSUE_REFUND
        and result.success
        and expected_outcome is ExpectedOutcome.REFUND
    ):
        components["correct_action_credit"] = 0.25

    if terminal_reward is not None:
        components["terminal_outcome_reward"] = (
            terminal_reward
        )
    if truncated:
        components["truncation_penalty"] = -0.5
    return components


def _evidence_potential(
    state: EnvironmentState,
) -> float:
    completed_checks = sum(
        (
            state.order_checked,
            state.policy_checked,
            state.payment_status_verified,
        )
    )
    return completed_checks * 0.1


def _case_fingerprint(case: RefundCase) -> str:
    encoded = json.dumps(
        asdict(case),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(
        f"unsupported fingerprint value: "
        f"{type(value).__name__}"
    )
