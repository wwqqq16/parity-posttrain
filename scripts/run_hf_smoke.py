"""Run one real Hugging Face rollout and save token metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a Hugging Face rollout smoke test."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model identifier.",
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
        "--output",
        type=Path,
        default=Path("artifacts/hf_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate one response and save its metadata."""

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

    result = backend.generate(
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("Model:", result.model_name)
    print("Device:", result.device)
    print("Generated text:", repr(result.generated_text))
    print(
        "Prompt tokens:",
        len(result.prompt_token_ids),
    )
    print(
        "Generated tokens:",
        len(result.generated_token_ids),
    )
    print(
        "Generated logprobs:",
        len(result.generated_token_logprobs),
    )
    print("Latency ms:", result.latency_ms)
    print("Tokens/s:", result.tokens_per_second)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
