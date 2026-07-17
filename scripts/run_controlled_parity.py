"""Run fixed-sequence parity under one device/cache condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from parity_posttrain.parity.forced_sequence import (
    forced_rollout_logprobs,
)
from parity_posttrain.parity.logprob_parity import (
    build_parity_report,
    rescore_generated_tokens,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure parity on a fixed token sequence under one "
            "device and cache condition."
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
        "--task-id",
        default="basket_001",
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu"],
        required=True,
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def find_generation(
    payload: dict[str, Any],
    *,
    task_id: str,
    turn_index: int,
) -> dict[str, Any]:
    """Find one stored generation in a benchmark artifact."""

    tasks = cast(
        list[dict[str, Any]],
        payload["tasks"],
    )

    for task_result in tasks:
        record = cast(
            dict[str, Any],
            task_result["benchmark_record"],
        )

        if record["task_id"] != task_id:
            continue

        run = cast(
            dict[str, Any],
            task_result["run"],
        )
        generations = cast(
            list[dict[str, Any]],
            run["generations"],
        )

        if turn_index < 0 or turn_index >= len(generations):
            raise ValueError(
                "turn index is out of range"
            )

        return generations[turn_index]

    raise ValueError(
        f"task not found in artifact: {task_id}"
    )


def main() -> None:
    """Run one controlled condition and save its report."""

    args = parse_args()

    payload = cast(
        dict[str, Any],
        json.loads(args.artifact.read_text()),
    )
    stored_generation = find_generation(
        payload,
        task_id=args.task_id,
        turn_index=args.turn_index,
    )

    model_name = cast(
        str,
        stored_generation["model_name"],
    )
    prompt_token_ids = cast(
        list[int],
        stored_generation["prompt_token_ids"],
    )
    generated_token_ids = cast(
        list[int],
        stored_generation["generated_token_ids"],
    )

    backend = HuggingFaceRolloutBackend(
        model_name=model_name,
        device=torch.device(args.device),
    )

    pad_token_id = backend.tokenizer.pad_token_id

    if not isinstance(pad_token_id, int):
        raise RuntimeError(
            "tokenizer has no integer pad_token_id"
        )

    rollout = forced_rollout_logprobs(
        model=backend.model,
        device=backend.device,
        pad_token_id=pad_token_id,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        use_cache=args.use_cache,
    )

    trainer_logprobs = rescore_generated_tokens(
        model=backend.model,
        device=backend.device,
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
    )

    token_texts = [
        cast(
            str,
            backend.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            ),
        )
        for token_id in generated_token_ids
    ]

    report = build_parity_report(
        model_name=model_name,
        device=str(backend.device),
        dtype=str(backend.dtype),
        token_ids=generated_token_ids,
        token_texts=token_texts,
        rollout_logprobs=rollout.token_logprobs,
        trainer_logprobs=trainer_logprobs,
        tolerance=args.tolerance,
    )

    result = {
        "condition": {
            "device": str(backend.device),
            "dtype": str(backend.dtype),
            "use_cache": args.use_cache,
        },
        "source": {
            "artifact": str(args.artifact),
            "task_id": args.task_id,
            "turn_index": args.turn_index,
            "prompt_token_count": len(prompt_token_ids),
            "generated_token_count": len(
                generated_token_ids
            ),
            "generated_text": stored_generation[
                "generated_text"
            ],
        },
        "forced_rollout": rollout.to_dict(),
        "trainer_logprobs": trainer_logprobs,
        "parity": report.to_dict(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Task:", args.task_id)
    print("Turn:", args.turn_index)
    print("Device:", backend.device)
    print("Dtype:", backend.dtype)
    print("Use cache:", args.use_cache)
    print("Prompt tokens:", len(prompt_token_ids))
    print("Generated tokens:", len(generated_token_ids))
    print("Rollout latency ms:", rollout.latency_ms)
    print(
        "Mean absolute error:",
        report.mean_absolute_error,
    )
    print(
        "Max absolute error:",
        report.max_absolute_error,
    )
    print(
        "P95 absolute error:",
        report.p95_absolute_error,
    )
    print(
        "Tokens over tolerance:",
        report.tokens_over_tolerance,
    )
    print("Within tolerance:", report.within_tolerance)

    print()
    print("Worst tokens:")

    worst = sorted(
        report.token_records,
        key=lambda token: token.absolute_error,
        reverse=True,
    )[:10]

    for token in worst:
        print(
            f"{token.index:02d}",
            f"token={token.token_text!r}",
            f"rollout={token.rollout_logprob:.8f}",
            f"trainer={token.trainer_logprob:.8f}",
            f"error={token.absolute_error:.8f}",
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
