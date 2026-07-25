# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.cases import CASES
from enterprise_eval.runner import BenchmarkRow, run_benchmark, run_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic enterprise refund-agent benchmark."
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        help="Run one case. Omit to run the full benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/enterprise_eval"),
    )
    return parser.parse_args()


def _format_optional(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def print_rows(rows: list[BenchmarkRow]) -> None:
    print(
        f"{'CASE':<29} {'LEVEL':<7} {'EXPECTED':<10} {'SUCCESS':<8} "
        f"{'RETRY':<7} {'INJECT':<7} {'REWARD':<7} FAILURE"
    )
    print("-" * 106)
    for row in rows:
        print(
            f"{row.case_id:<29} {row.difficulty:<7} {row.expected:<10} "
            f"{str(row.success):<8} {_format_optional(row.recovered):<7} "
            f"{_format_optional(row.injection_resisted):<7} "
            f"{row.reward:<7.2f} {row.failure_type or '-'}"
        )
        print(f"  artifact: {row.artifact_path}")


def main() -> None:
    args = parse_args()
    if args.case:
        rows = [run_case(args.case, args.output_dir)]
    else:
        rows = run_benchmark(args.output_dir)
    print_rows(rows)
    successes = sum(row.success for row in rows)
    recovered = sum(row.recovered is True for row in rows)
    transient_cases = sum(row.transient_failures > 0 for row in rows)
    print(f"\nSuccess rate: {successes}/{len(rows)} ({successes / len(rows):.1%})")
    if transient_cases:
        print(f"Recovered transient failures: {recovered}/{transient_cases}")


if __name__ == "__main__":
    main()
