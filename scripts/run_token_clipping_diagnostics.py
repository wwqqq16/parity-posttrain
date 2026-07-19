"""Run token-level PPO clipping diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from parity_posttrain.provenance import (
    build_experiment_provenance,
    set_experiment_seed,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)
from parity_posttrain.training import (
    PolicyNormalization,
    TokenClippingDiagnostic,
    collate_training_examples,
    load_training_examples,
    prepare_trainable_parameters,
    run_clipped_policy_training,
    select_training_examples,
)

_DEFAULT_TASK_IDS = [
    "catalog_004",
    "shopping_004",
]


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train on fixed rollout logprobs and report "
            "token-level PPO clipping."
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
        default=_DEFAULT_TASK_IDS,
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
        "--trainable-parameters",
        nargs="+",
        default=[
            "model.layers.23.self_attn.o_proj.weight"
        ],
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "token",
            "sequence",
            "trajectory",
        ],
        default="trajectory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.003,
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
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--model-revision",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/token_clipping_diagnostics.json"
        ),
    )

    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    """Validate and construct a requested device."""

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


def diagnostic_to_dict(
    diagnostic: TokenClippingDiagnostic,
    *,
    backend: HuggingFaceRolloutBackend,
) -> dict[str, object]:
    """Serialize one token diagnostic."""

    return {
        "flat_token_index": (
            diagnostic.flat_token_index
        ),
        "batch_row": diagnostic.batch_row,
        "task_id": diagnostic.task_id,
        "turn_index": diagnostic.turn_index,
        "generated_token_index": (
            diagnostic.generated_token_index
        ),
        "sequence_position": (
            diagnostic.sequence_position
        ),
        "token_id": diagnostic.token_id,
        "token_text": backend.tokenizer.decode(
            [diagnostic.token_id],
            skip_special_tokens=False,
        ),
        "advantage": diagnostic.advantage,
        "old_logprob": diagnostic.old_logprob,
        "current_logprob": (
            diagnostic.current_logprob
        ),
        "ratio": diagnostic.ratio,
        "out_of_range": diagnostic.out_of_range,
        "active_clipped": (
            diagnostic.active_clipped
        ),
        "clip_direction": (
            diagnostic.clip_direction
        ),
    }


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run training and write clipping diagnostics."""

    args = parse_args(argv)
    set_experiment_seed(args.seed)
    task_ids = tuple(args.task_ids)
    examples = load_training_examples(
        args.artifact
    )
    selected = select_training_examples(
        examples,
        task_ids,
    )
    model_name = selected[0].model_name
    device = resolve_device(args.device)

    backend = HuggingFaceRolloutBackend(
        model_name=model_name,
        device=device,
        revision=args.model_revision,
    )
    provenance = build_experiment_provenance(
        source_artifact=args.artifact,
        model_name=backend.model_name,
        model_revision=backend.model_revision,
        seed=args.seed,
    )
    pad_token_id = (
        backend.tokenizer.pad_token_id
    )

    if not isinstance(pad_token_id, int):
        raise ValueError(
            "tokenizer has no integer pad_token_id"
        )

    batch = collate_training_examples(
        selected,
        pad_token_id=pad_token_id,
        device=backend.device,
    )
    normalization = cast(
        PolicyNormalization,
        args.normalization,
    )
    parameter_selection = (
        prepare_trainable_parameters(
            backend.model,
            args.trainable_parameters,
        )
    )
    step_payloads: list[
        dict[str, object]
    ] = []

    def observer(
        step_index: int,
        diagnostics: tuple[
            TokenClippingDiagnostic,
            ...,
        ],
    ) -> None:
        ratios = tuple(
            diagnostic.ratio
            for diagnostic in diagnostics
        )
        out_of_range = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.out_of_range
        )
        active = tuple(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.active_clipped
        )

        step_payloads.append(
            {
                "step_index": step_index,
                "token_count": len(diagnostics),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "out_of_range_count": len(
                    out_of_range
                ),
                "active_clip_count": len(active),
                "out_of_range_tokens": [
                    diagnostic_to_dict(
                        diagnostic,
                        backend=backend,
                    )
                    for diagnostic in out_of_range
                ],
            }
        )

    try:
        optimizer = torch.optim.SGD(
            parameter_selection.parameters,
            lr=args.learning_rate,
        )
        loop_result = run_clipped_policy_training(
            model=backend.model,
            optimizer=optimizer,
            batch=batch,
            steps=args.steps,
            clip_epsilon=args.clip_epsilon,
            max_gradient_norm=(
                args.max_gradient_norm
            ),
            normalization=normalization,
            observer=observer,
        )
    finally:
        parameter_selection.restore_requires_grad()

    payload = {
        "schema_version": 2,
        "source_artifact": str(args.artifact),
        "provenance": provenance.to_dict(),
        "model_name": model_name,
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "task_ids": list(task_ids),
        "training": {
            "normalization": normalization,
            "optimizer_steps": args.steps,
            "learning_rate": args.learning_rate,
            "clip_epsilon": args.clip_epsilon,
            "max_gradient_norm": (
                args.max_gradient_norm
            ),
            "trainable_parameter_names": list(
                args.trainable_parameters
            ),
            "parameter_deltas": list(
                loop_result.parameter_deltas
            ),
            "final_parameter_delta": (
                loop_result.final_parameter_delta
            ),
        },
        "steps": step_payloads,
    }

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

    print("=" * 72)
    print("TOKEN CLIPPING DIAGNOSTICS")
    print("=" * 72)
    print("Model:", model_name)
    print("Tasks:", ", ".join(task_ids))
    print("Trainable tokens:", batch.trainable_token_count)
    print()

    for step in step_payloads:
        print(
            f"Step {step['step_index']}:",
            f"ratio_min={step['ratio_min']:.12f}",
            f"ratio_max={step['ratio_max']:.12f}",
            (
                "out_of_range="
                f"{step['out_of_range_count']}"
            ),
            (
                "active="
                f"{step['active_clip_count']}"
            ),
        )

        tokens = step["out_of_range_tokens"]

        if not isinstance(tokens, list):
            raise RuntimeError(
                "diagnostic token payload is invalid"
            )

        for token in tokens:
            if not isinstance(token, dict):
                raise RuntimeError(
                    "diagnostic token payload is invalid"
                )

            print(
                " ",
                f"task={token['task_id']}",
                f"turn={token['turn_index']}",
                (
                    "generated_index="
                    f"{token['generated_token_index']}"
                ),
                f"token={token['token_text']!r}",
                f"ratio={token['ratio']:.12f}",
                f"direction={token['clip_direction']}",
                (
                    "active="
                    f"{token['active_clipped']}"
                ),
            )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
