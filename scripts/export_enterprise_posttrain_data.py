# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.posttrain import export_posttrain_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export semantic post-training records from enterprise-agent runs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/enterprise_benchmark"),
        help="Directory recursively containing enterprise-agent-run.v1 JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/enterprise_posttrain"),
        help="Destination for SFT, preference, and step-reward JSONL files.",
    )
    parser.add_argument(
        "--include-failure-injection",
        action="store_true",
        help="Also include artifacts/enterprise_eval/failure_injection when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.input_dir.rglob("*.json"))
    if args.include_failure_injection:
        failure_dir = Path("artifacts/enterprise_eval/failure_injection")
        if failure_dir.exists():
            paths.extend(failure_dir.rglob("*.json"))
    if not paths:
        raise SystemExit(f"No JSON artifacts found under {args.input_dir}")

    summary = export_posttrain_datasets(paths, args.output_dir)
    print("Enterprise semantic post-training export")
    print("=" * 44)
    print(f"Source artifacts: {summary.source_artifact_count}")
    print(f"SFT records:      {summary.sft_record_count}")
    print(f"Preference pairs: {summary.preference_pair_count}")
    print(f"Step rewards:     {summary.step_reward_record_count}")
    print(f"SFT path:         {summary.sft_path}")
    print(f"Preferences path: {summary.preferences_path}")
    print(f"Step rewards path:{summary.step_rewards_path}")
    print(f"Manifest path:    {summary.manifest_path}")
    print()
    print("Token-level training ready: no")
    print("A real model-backed agent is still required for token IDs and logprobs.")


if __name__ == "__main__":
    main()
