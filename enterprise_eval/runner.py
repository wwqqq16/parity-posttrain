from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enterprise_eval.artifacts import write_run_artifact
from enterprise_eval.cases import CASES
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.models import EvaluationResult
from enterprise_eval.scripted_agent import ScriptedRefundAgent


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    expected: str
    success: bool
    reward: float
    failure_type: str | None
    artifact_path: Path


def run_case(case_id: str, output_dir: Path) -> BenchmarkRow:
    case = CASES[case_id]
    env = RefundEnvironment(case)
    ScriptedRefundAgent().run(env)
    evaluation: EvaluationResult = RefundEvaluator().evaluate(env)
    assert env.run is not None
    artifact_path = write_run_artifact(env.run, evaluation, output_dir)
    return BenchmarkRow(
        case_id=case_id,
        expected=evaluation.expected_outcome,
        success=evaluation.task_success,
        reward=evaluation.final_reward,
        failure_type=evaluation.failure_type,
        artifact_path=artifact_path,
    )


def run_benchmark(output_dir: Path) -> list[BenchmarkRow]:
    return [run_case(case_id, output_dir) for case_id in sorted(CASES)]
