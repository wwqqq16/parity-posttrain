# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.models import Architecture
from enterprise_eval.runner import (
    BenchmarkRow,
    BenchmarkSummary,
    run_ablation,
    run_benchmark,
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-agent and planner-critic refund workflows."
    )
    parser.add_argument(
        "--architecture",
        choices=["single", "planner-critic", "both"],
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/enterprise_benchmark"),
    )
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print one row per case before aggregate metrics.",
    )
    return parser.parse_args()


def print_case_rows(rows: list[BenchmarkRow]) -> None:
    print(
        f"{'ARCH':<16} {'CASE':<33} {'LEVEL':<7} {'OK':<5} "
        f"{'POLICY':<7} {'STEPS':<6} {'CALLS':<6} {'LAT(ms)':<9} FAILURE"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row.architecture:<16} {row.case_id:<33} {row.difficulty:<7} "
            f"{str(row.success):<5} {str(row.policy_violation):<7} "
            f"{row.tool_steps:<6} {row.component_calls:<6} "
            f"{row.latency_ms:<9.3f} {row.failure_type or '-'}"
        )


def _format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def print_summaries(summaries: list[BenchmarkSummary]) -> None:
    print(
        f"{'ARCH':<16} {'LEVEL':<7} {'N':<4} {'SUCCESS':<9} {'POLICY':<9} "
        f"{'ESCALATE':<10} {'INVALID':<9} {'RECOVERY':<9} "
        f"{'STEPS':<7} {'CALLS':<7} {'LAT(ms)':<9}"
    )
    print("-" * 124)
    for summary in summaries:
        print(
            f"{summary.architecture:<16} {summary.difficulty:<7} "
            f"{summary.cases:<4} {summary.success_rate:<9.1%} "
            f"{summary.policy_violation_rate:<9.1%} "
            f"{summary.correct_escalation_rate:<10.1%} "
            f"{summary.invalid_tool_call_rate:<9.1%} "
            f"{_format_rate(summary.recovery_rate):<9} "
            f"{summary.average_steps:<7.2f} "
            f"{summary.average_component_calls:<7.2f} "
            f"{summary.average_latency_ms:<9.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.architecture == "both":
        rows = run_ablation(args.output_dir)
    else:
        architecture = Architecture(args.architecture)
        rows = run_benchmark(args.output_dir, architecture=architecture)

    if args.show_cases:
        print_case_rows(rows)
        print()
    print_summaries(summarize_rows(rows))
    print(
        "\nLatency is local Python control-flow latency, not remote-model serving latency. "
        "Component calls represent architecture overhead."
    )


if __name__ == "__main__":
    main()
