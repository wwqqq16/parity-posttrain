from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.cases import CASES
from enterprise_eval.runner import run_benchmark, run_case


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


def print_rows(rows: list[object]) -> None:
    print(f"{'CASE':<24} {'EXPECTED':<10} {'SUCCESS':<8} {'REWARD':<8} FAILURE")
    print("-" * 72)
    for row in rows:
        print(
            f"{row.case_id:<24} {row.expected:<10} "
            f"{str(row.success):<8} {row.reward:<8.2f} "
            f"{row.failure_type or '-'}"
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
    print(f"\nSuccess rate: {successes}/{len(rows)} ({successes / len(rows):.1%})")


if __name__ == "__main__":
    main()
