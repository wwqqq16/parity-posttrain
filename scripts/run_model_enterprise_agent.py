# ruff: noqa: E402
"""Run one real Hugging Face model through the enterprise environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.artifacts import write_run_artifact
from enterprise_eval.cases import CASES
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.model_agent import (
    GUARD_PROFILES,
    PROMPT_PROFILES,
    ModelBackedRefundAgent,
)
from parity_posttrain.rollout.hf_backend import HuggingFaceRolloutBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="eligible_standard", choices=sorted(CASES))
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--prompt-profile",
        choices=PROMPT_PROFILES,
        default="baseline",
        help="System-prompt condition for controlled ablations.",
    )
    parser.add_argument(
        "--guard-profile",
        choices=GUARD_PROFILES,
        default="none",
        help="Pre-dispatch execution guard for sensitive actions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/enterprise_model"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    backend = HuggingFaceRolloutBackend(args.model_name, device=device)
    agent = ModelBackedRefundAgent(
        backend,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        prompt_profile=args.prompt_profile,
        guard_profile=args.guard_profile,
    )
    env = RefundEnvironment(CASES[args.case])
    agent.run(env)

    assert env.run is not None
    evaluation = RefundEvaluator().evaluate(env)
    artifact_path = write_run_artifact(env.run, evaluation, args.output_dir)

    print("Model-backed enterprise agent")
    print("=" * 44)
    print(f"case:          {args.case}")
    print(f"model:         {args.model_name}")
    print(f"prompt profile: {args.prompt_profile}")
    print(f"guard profile:  {args.guard_profile}")
    print(f"architecture:  {env.run.architecture}")
    print(f"component calls: {env.run.component_calls}")
    print(f"steps:         {len(env.run.steps)}")
    print(f"success:       {evaluation.task_success}")
    print(f"reward:        {evaluation.final_reward:.2f}")
    print(f"failure type:  {evaluation.failure_type or '-'}")
    guard_rejections = env.run.metadata.get("runtime_guard_rejections", [])
    guard_count = len(guard_rejections) if isinstance(guard_rejections, list) else 0
    print(f"guard rejections: {guard_count}")
    print(f"artifact:      {artifact_path}")

    generations = env.run.metadata.get("model_generations", [])
    if isinstance(generations, list):
        for record in generations:
            if not isinstance(record, dict):
                continue
            print()
            print(f"Turn {record.get('turn_index')}")
            print(f"generated: {record.get('generated_text')}")
            print(f"parse error: {record.get('parse_error') or '-'}")


if __name__ == "__main__":
    main()
