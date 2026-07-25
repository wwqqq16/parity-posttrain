# ruff: noqa: E402
"""Run stored and forced parity checks on an enterprise model turn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.model_posttrain import (
    build_enterprise_training_examples,
    get_model_generation,
    load_enterprise_model_artifact,
)
from parity_posttrain.parity.forced_sequence import forced_rollout_logprobs
from parity_posttrain.parity.logprob_parity import (
    build_parity_report,
    rescore_generated_tokens,
)
from parity_posttrain.provenance import (
    build_experiment_provenance,
    set_experiment_seed,
)
from parity_posttrain.rollout.hf_backend import HuggingFaceRolloutBackend


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare stored rollout, fixed-sequence rollout, and trainer "
            "log-probabilities for one enterprise-agent generation."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--turn-index", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        required=True,
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-revision", default=None)
    return parser.parse_args()


def main() -> None:
    """Run parity checks and save one report."""

    args = parse_args()
    set_experiment_seed(args.seed)

    payload = load_enterprise_model_artifact(args.artifact)
    run = cast(dict[str, Any], payload["run"])
    evaluation = cast(dict[str, Any], payload["evaluation"])
    generation = get_model_generation(payload, turn_index=args.turn_index)
    examples = build_enterprise_training_examples(payload)
    example = examples[args.turn_index]

    backend = HuggingFaceRolloutBackend(
        model_name=example.model_name,
        device=torch.device(args.device),
        revision=args.model_revision,
    )
    provenance = build_experiment_provenance(
        source_artifact=args.artifact,
        model_name=backend.model_name,
        requested_model_revision=backend.model_revision,
        resolved_model_revision=backend.resolved_model_revision,
        seed=args.seed,
    )

    pad_token_id = backend.tokenizer.pad_token_id
    if not isinstance(pad_token_id, int):
        raise RuntimeError("tokenizer has no integer pad_token_id")

    forced_rollout = forced_rollout_logprobs(
        model=backend.model,
        device=backend.device,
        pad_token_id=pad_token_id,
        prompt_token_ids=list(example.prompt_token_ids),
        generated_token_ids=list(example.generated_token_ids),
        use_cache=args.use_cache,
    )
    trainer_logprobs = rescore_generated_tokens(
        model=backend.model,
        device=backend.device,
        prompt_token_ids=list(example.prompt_token_ids),
        generated_token_ids=list(example.generated_token_ids),
    )

    token_texts = [
        cast(
            str,
            backend.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            ),
        )
        for token_id in example.generated_token_ids
    ]

    stored_to_trainer = build_parity_report(
        model_name=example.model_name,
        device=str(backend.device),
        dtype=str(backend.dtype),
        token_ids=list(example.generated_token_ids),
        token_texts=token_texts,
        rollout_logprobs=list(example.rollout_logprobs),
        trainer_logprobs=trainer_logprobs,
        tolerance=args.tolerance,
    )
    forced_to_trainer = build_parity_report(
        model_name=example.model_name,
        device=str(backend.device),
        dtype=str(backend.dtype),
        token_ids=list(example.generated_token_ids),
        token_texts=token_texts,
        rollout_logprobs=forced_rollout.token_logprobs,
        trainer_logprobs=trainer_logprobs,
        tolerance=args.tolerance,
    )
    stored_to_forced = build_parity_report(
        model_name=example.model_name,
        device=str(backend.device),
        dtype=str(backend.dtype),
        token_ids=list(example.generated_token_ids),
        token_texts=token_texts,
        rollout_logprobs=list(example.rollout_logprobs),
        trainer_logprobs=forced_rollout.token_logprobs,
        tolerance=args.tolerance,
    )

    result = {
        "schema_version": "enterprise-model-parity.v1",
        "provenance": provenance.to_dict(),
        "condition": {
            "device": str(backend.device),
            "dtype": str(backend.dtype),
            "use_cache": args.use_cache,
            "tolerance": args.tolerance,
        },
        "source": {
            "artifact": str(args.artifact),
            "case_id": run["case_id"],
            "run_id": run["run_id"],
            "turn_index": args.turn_index,
            "task_success": evaluation["task_success"],
            "failure_type": evaluation.get("failure_type"),
            "final_reward": evaluation["final_reward"],
            "model_name": example.model_name,
            "prompt_token_count": len(example.prompt_token_ids),
            "generated_token_count": len(example.generated_token_ids),
            "generated_text": generation["generated_text"],
        },
        "training_example": {
            "task_id": example.task_id,
            "turn_index": example.turn_index,
            "status": example.status,
            "reward": example.reward,
            "input_token_count": len(example.input_ids),
            "trainable_token_count": sum(example.loss_mask),
        },
        "forced_rollout": forced_rollout.to_dict(),
        "trainer_logprobs": trainer_logprobs,
        "parity": {
            "stored_rollout_vs_trainer": stored_to_trainer.to_dict(),
            "forced_rollout_vs_trainer": forced_to_trainer.to_dict(),
            "stored_rollout_vs_forced_rollout": stored_to_forced.to_dict(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Enterprise model parity")
    print("=" * 44)
    print("Case:", run["case_id"])
    print("Run ID:", run["run_id"])
    print("Turn:", args.turn_index)
    print("Model:", example.model_name)
    print("Device:", backend.device)
    print("Dtype:", backend.dtype)
    print("Use cache:", args.use_cache)
    print("Prompt tokens:", len(example.prompt_token_ids))
    print("Generated tokens:", len(example.generated_token_ids))
    _print_report("Stored rollout vs trainer", stored_to_trainer)
    _print_report("Forced rollout vs trainer", forced_to_trainer)
    _print_report("Stored rollout vs forced rollout", stored_to_forced)
    print()
    print("Output:", args.output)


def _print_report(label: str, report: Any) -> None:
    print()
    print(label)
    print("  mean absolute error:", report.mean_absolute_error)
    print("  max absolute error:", report.max_absolute_error)
    print("  P95 absolute error:", report.p95_absolute_error)
    print("  tokens over tolerance:", report.tokens_over_tolerance)
    print("  within tolerance:", report.within_tolerance)


if __name__ == "__main__":
    main()
