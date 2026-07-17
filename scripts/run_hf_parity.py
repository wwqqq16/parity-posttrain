"""Compare rollout and trainer logprobs for one HF generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch

from parity_posttrain.parity.logprob_parity import (
    build_parity_report,
    rescore_generated_tokens,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
    synchronize_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run HF rollout/trainer logprob parity."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hf_parity.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate, rescore, compare, and save one report."""

    args = parse_args()

    device = (
        None
        if args.device == "auto"
        else torch.device(args.device)
    )

    backend = HuggingFaceRolloutBackend(
        model_name=args.model,
        device=device,
    )

    generation = backend.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. "
                    "Return only the final answer."
                ),
            },
            {
                "role": "user",
                "content": "What is 17 multiplied by 6?",
            },
        ],
        max_new_tokens=args.max_new_tokens,
    )

    synchronize_device(backend.device)
    rescore_started = time.perf_counter()

    trainer_logprobs = rescore_generated_tokens(
        model=backend.model,
        device=backend.device,
        prompt_token_ids=generation.prompt_token_ids,
        generated_token_ids=generation.generated_token_ids,
    )

    synchronize_device(backend.device)
    rescore_latency_ms = (
        time.perf_counter() - rescore_started
    ) * 1000

    token_texts = [
        cast(
            str,
            backend.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            ),
        )
        for token_id in generation.generated_token_ids
    ]

    report = build_parity_report(
        model_name=generation.model_name,
        device=generation.device,
        dtype=str(backend.dtype),
        token_ids=generation.generated_token_ids,
        token_texts=token_texts,
        rollout_logprobs=(
            generation.generated_token_logprobs
        ),
        trainer_logprobs=trainer_logprobs,
        tolerance=args.tolerance,
    )

    payload = {
        "generation": generation.to_dict(),
        "trainer_logprobs": trainer_logprobs,
        "rescore_latency_ms": round(
            rescore_latency_ms,
            3,
        ),
        "parity": report.to_dict(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Model:", report.model_name)
    print("Device:", report.device)
    print("Dtype:", report.dtype)
    print("Generated text:", repr(generation.generated_text))
    print("Token count:", report.token_count)
    print("Tolerance:", report.tolerance)
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
    print("Rescore latency ms:", round(rescore_latency_ms, 3))
    print()

    for record in report.token_records:
        print(
            f"{record.index:02d}",
            f"id={record.token_id}",
            f"token={record.token_text!r}",
            f"rollout={record.rollout_logprob:.8f}",
            f"trainer={record.trainer_logprob:.8f}",
            f"error={record.absolute_error:.8f}",
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
