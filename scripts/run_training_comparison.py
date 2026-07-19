"""Compare controlled policy updates across normalizations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)
from parity_posttrain.training import (
    collate_training_examples,
    load_training_examples,
    run_training_comparison,
    select_training_examples,
    training_comparison_to_dict,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run controlled policy updates from identical "
            "initial model weights using token, sequence, "
            "and trajectory normalization."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/agent_benchmark.json"
        ),
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=[
            "catalog_004",
            "shopping_004",
        ],
        help=(
            "Task IDs to include. Every generation turn "
            "for each selected task is used."
        ),
    )
    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "mps",
            "cuda",
        ],
        default="cpu",
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=[
            "token",
            "sequence",
            "trajectory",
        ],
        default=[
            "token",
            "sequence",
            "trajectory",
        ],
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help=(
            "Number of optimizer steps to run for each "
            "normalization."
        ),
    )
    parser.add_argument(
        "--trainable-parameter",
        action="append",
        dest="trainable_parameters",
        default=None,
        help=(
            "Exact parameter name to update. Repeat this "
            "argument to train multiple parameters."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/training_comparison.json"
        ),
    )

    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    """Validate and construct the requested device."""

    if (
        name == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is not available"
        )

    if (
        name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return torch.device(name)


def main() -> None:
    """Run the comparison and write its JSON artifact."""

    args = parse_args()

    examples = load_training_examples(
        args.artifact
    )
    selected = select_training_examples(
        examples,
        args.task_ids,
    )

    if not selected:
        raise RuntimeError(
            "no training examples were selected"
        )

    model_name = selected[0].model_name
    device = resolve_device(args.device)

    backend = HuggingFaceRolloutBackend(
        model_name=model_name,
        device=device,
    )

    pad_token_id = backend.tokenizer.pad_token_id

    if not isinstance(pad_token_id, int):
        raise RuntimeError(
            "tokenizer has no integer pad_token_id"
        )

    batch = collate_training_examples(
        selected,
        pad_token_id=pad_token_id,
        device=backend.device,
    )

    trainable_parameters = (
        args.trainable_parameters
        if args.trainable_parameters is not None
        else ["model.norm.weight"]
    )

    summary = run_training_comparison(
        model=backend.model,
        batch=batch,
        source_artifact=str(args.artifact),
        model_name=model_name,
        trainable_parameter_names=(
            trainable_parameters
        ),
        normalizations=args.normalizations,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        max_gradient_norm=(
            args.max_gradient_norm
        ),
    )

    payload = training_comparison_to_dict(
        summary
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Model:", summary.model_name)
    print("Device:", summary.device)
    print("Dtype:", summary.dtype)
    print(
        "Trainable parameters:",
        summary.trainable_parameter_count,
    )
    print(
        "Trainable parameter names:",
        list(summary.trainable_parameter_names),
    )
    print(
        "Optimizer steps:",
        len(summary.rows[0].resolved_steps),
    )

    print()
    print("Selected trajectories:")

    for task in summary.tasks:
        print(
            f"- {task.task_id}:",
            f"reward={task.reward}",
            f"turns={task.turn_count}",
            f"tokens={task.generated_token_count}",
        )

    print()
    print("Normalization results:")

    for row in summary.rows:
        print()
        print(
            "Normalization:",
            row.normalization,
        )
        print("Step history:")

        for step in row.resolved_steps:
            print(
                f"  Step {step.step_index}:",
                f"loss={step.loss:.12e}",
                (
                    "gradient_norm="
                    f"{step.gradient_norm:.12e}"
                ),
                f"mean_ratio={step.mean_ratio:.12e}",
                (
                    "approximate_kl="
                    f"{step.approximate_kl:.12e}"
                ),
                (
                    "clip_fraction="
                    f"{step.clip_fraction:.12e}"
                ),
                (
                    "parameter_delta="
                    f"{step.parameter_delta:.12e}"
                ),
            )

        print("Final loss:", row.loss)
        print(
            "Final gradient norm:",
            row.gradient_norm,
        )
        print(
            "Final mean ratio:",
            row.mean_ratio,
        )
        print(
            "Final approximate KL:",
            row.approximate_kl,
        )
        print(
            "Final clip fraction:",
            row.clip_fraction,
        )
        print(
            "Final parameter delta:",
            row.parameter_delta,
        )
        print(
            "Mean absolute logprob shift:",
            row.mean_absolute_logprob_shift,
        )
        print(
            "Maximum absolute logprob shift:",
            row.max_absolute_logprob_shift,
        )

        for shift in row.task_shifts:
            print(
                f"  {shift.task_id}:",
                (
                    "mean_shift="
                    f"{shift.mean_logprob_shift:.12e}"
                ),
                (
                    "mean_abs_shift="
                    f"{shift.mean_absolute_logprob_shift:.12e}"
                ),
                (
                    "max_abs_shift="
                    f"{shift.max_absolute_logprob_shift:.12e}"
                ),
            )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
