from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from enterprise_eval.artifacts import write_run_artifact
from enterprise_eval.cases import CASES
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.models import Architecture, Difficulty, EvaluationResult, RefundCase
from enterprise_eval.scripted_agent import PlannerCriticAgent, SingleAgentBaseline


class Agent(Protocol):
    def run(self, env: RefundEnvironment) -> None: ...


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    architecture: str
    difficulty: str
    expected: str
    success: bool
    policy_violation: bool
    correct_escalation: bool
    invalid_tool_calls: int
    reward: float
    transient_failures: int
    recovered: bool | None
    injection_resisted: bool | None
    failure_profile: str
    injection_step: int | None
    injection_triggered: bool
    failure_step: int | None
    failure_type: str | None
    tool_steps: int
    component_calls: int
    latency_ms: float
    artifact_path: Path


@dataclass(frozen=True)
class BenchmarkSummary:
    architecture: str
    difficulty: str
    cases: int
    success_rate: float
    policy_violation_rate: float
    correct_escalation_rate: float
    invalid_tool_call_rate: float
    recovery_rate: float | None
    average_steps: float
    average_component_calls: float
    average_latency_ms: float


def build_agent(architecture: Architecture) -> Agent:
    if architecture is Architecture.SINGLE:
        return SingleAgentBaseline()
    if architecture is Architecture.PLANNER_CRITIC:
        return PlannerCriticAgent()
    raise ValueError(f"Unsupported benchmark architecture: {architecture.value}")


def run_case(
    case_id: str,
    output_dir: Path,
    *,
    agent: Agent | None = None,
    architecture: Architecture = Architecture.SINGLE,
) -> BenchmarkRow:
    case = CASES[case_id]
    return run_refund_case(
        case,
        output_dir,
        agent=agent,
        architecture=architecture,
    )


def run_refund_case(
    case: RefundCase,
    output_dir: Path,
    *,
    agent: Agent | None = None,
    architecture: Architecture = Architecture.SINGLE,
) -> BenchmarkRow:
    """Run one provided case, including generated task-factory cases."""

    env = RefundEnvironment(case)
    selected_agent = agent or build_agent(architecture)

    started = perf_counter()
    selected_agent.run(env)
    latency_ms = (perf_counter() - started) * 1_000.0

    evaluation: EvaluationResult = RefundEvaluator().evaluate(env)
    assert env.run is not None
    artifact_path = write_run_artifact(env.run, evaluation, output_dir)
    return BenchmarkRow(
        case_id=case.case_id,
        architecture=env.run.architecture,
        difficulty=case.difficulty.value,
        expected=evaluation.expected_outcome,
        success=evaluation.task_success,
        policy_violation=evaluation.policy_violation,
        correct_escalation=evaluation.correct_escalation,
        invalid_tool_calls=evaluation.invalid_tool_calls,
        reward=evaluation.final_reward,
        transient_failures=evaluation.transient_tool_failures,
        recovered=evaluation.recovered_from_tool_failure,
        injection_resisted=evaluation.prompt_injection_resisted,
        failure_profile=case.failure_profile.value,
        injection_step=case.failure_injection_step,
        injection_triggered=(
            env.state.scheduled_failures_triggered > 0
        ),
        failure_step=evaluation.failure_step,
        failure_type=evaluation.failure_type,
        tool_steps=len(env.run.steps),
        component_calls=env.run.component_calls,
        latency_ms=latency_ms,
        artifact_path=artifact_path,
    )


def run_benchmark(
    output_dir: Path,
    *,
    architecture: Architecture = Architecture.SINGLE,
) -> list[BenchmarkRow]:
    return [
        run_case(
            case_id,
            output_dir / architecture.value,
            architecture=architecture,
        )
        for case_id in sorted(CASES)
    ]


def run_task_suite(
    cases: list[RefundCase] | tuple[RefundCase, ...],
    output_dir: Path,
    *,
    architecture: Architecture = Architecture.SINGLE,
) -> list[BenchmarkRow]:
    """Run an explicit generated task suite."""

    return [
        run_refund_case(
            case,
            output_dir / architecture.value,
            architecture=architecture,
        )
        for case in cases
    ]


def run_ablation(output_dir: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for architecture in (Architecture.SINGLE, Architecture.PLANNER_CRITIC):
        rows.extend(run_benchmark(output_dir, architecture=architecture))
    return rows


def summarize_rows(rows: list[BenchmarkRow]) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    for architecture in (Architecture.SINGLE, Architecture.PLANNER_CRITIC):
        architecture_rows = [
            row for row in rows if row.architecture == architecture.value
        ]
        if not architecture_rows:
            continue
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            selected = [
                row for row in architecture_rows if row.difficulty == difficulty.value
            ]
            if selected:
                summaries.append(_summarize_group(architecture.value, difficulty.value, selected))
        summaries.append(_summarize_group(architecture.value, "all", architecture_rows))
    return summaries


def _summarize_group(
    architecture: str,
    difficulty: str,
    rows: list[BenchmarkRow],
) -> BenchmarkSummary:
    transient_rows = [row for row in rows if row.transient_failures > 0]
    recovery_rate: float | None = None
    if transient_rows:
        recovery_rate = sum(row.recovered is True for row in transient_rows) / len(
            transient_rows
        )

    return BenchmarkSummary(
        architecture=architecture,
        difficulty=difficulty,
        cases=len(rows),
        success_rate=sum(row.success for row in rows) / len(rows),
        policy_violation_rate=sum(row.policy_violation for row in rows) / len(rows),
        correct_escalation_rate=sum(row.correct_escalation for row in rows)
        / len(rows),
        invalid_tool_call_rate=sum(row.invalid_tool_calls > 0 for row in rows)
        / len(rows),
        recovery_rate=recovery_rate,
        average_steps=sum(row.tool_steps for row in rows) / len(rows),
        average_component_calls=sum(row.component_calls for row in rows) / len(rows),
        average_latency_ms=sum(row.latency_ms for row in rows) / len(rows),
    )
