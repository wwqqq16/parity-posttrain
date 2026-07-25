# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.runner import BenchmarkRow, run_case
from enterprise_eval.scripted_agent import (
    ScriptedRefundAgent,
    UnsafePromptFollowingAgent,
)


def print_row(label: str, row: BenchmarkRow) -> None:
    print(f"{label}")
    print(f"  success: {row.success}")
    print(f"  expected: {row.expected}")
    print(f"  policy failure: {row.failure_type or '-'}")
    print(f"  failure step: {row.failure_step if row.failure_step is not None else '-'}")
    print(f"  prompt injection resisted: {row.injection_resisted}")
    print(f"  reward: {row.reward:.2f}")
    print(f"  artifact: {row.artifact_path}")


def main() -> None:
    output_dir = Path("artifacts/enterprise_eval/failure_injection")
    safe = run_case(
        "prompt_injection_high_value",
        output_dir,
        agent=ScriptedRefundAgent(),
    )
    unsafe = run_case(
        "prompt_injection_high_value",
        output_dir,
        agent=UnsafePromptFollowingAgent(),
    )

    print("Controlled prompt-injection experiment")
    print("=" * 44)
    print_row("Safe policy-driven agent", safe)
    print()
    print_row("Unsafe instruction-following agent", unsafe)


if __name__ == "__main__":
    main()
