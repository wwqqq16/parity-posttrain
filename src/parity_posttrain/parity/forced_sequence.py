"""Controlled parity experiments on a fixed continuation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from transformers import LogitsProcessorList

from parity_posttrain.rollout.hf_backend import (
    gather_generated_logprobs,
    synchronize_device,
)


@dataclass(slots=True)
class ForcedRolloutResult:
    """Raw rollout logprobs for one forced continuation."""

    use_cache: bool
    token_logprobs: list[float]
    latency_ms: float

    def validate(self) -> None:
        """Validate the controlled rollout result."""

        if not self.token_logprobs:
            raise ValueError(
                "token_logprobs must not be empty"
            )

        if self.latency_ms <= 0:
            raise ValueError(
                "latency_ms must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        self.validate()
        return asdict(self)


class ForcedContinuationProcessor:
    """Force generate() to follow a predefined token sequence."""

    def __init__(
        self,
        *,
        prompt_length: int,
        token_ids: list[int],
    ) -> None:
        """Configure the continuation to force."""

        if prompt_length <= 0:
            raise ValueError(
                "prompt_length must be positive"
            )

        if not token_ids:
            raise ValueError(
                "token_ids must not be empty"
            )

        if any(token_id < 0 for token_id in token_ids):
            raise ValueError(
                "token IDs must not be negative"
            )

        self.prompt_length = prompt_length
        self.token_ids = token_ids.copy()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Allow only the predefined token for the current step."""

        step = int(input_ids.shape[1]) - self.prompt_length

        if step < 0 or step >= len(self.token_ids):
            raise ValueError(
                "forced continuation step is out of range"
            )

        target_token_id = self.token_ids[step]

        if target_token_id >= scores.shape[-1]:
            raise ValueError(
                "forced token ID exceeds vocabulary size"
            )

        forced_scores = torch.full_like(
            scores,
            float("-inf"),
        )
        forced_scores[:, target_token_id] = 0.0

        return cast(
            torch.FloatTensor,
            forced_scores,
        )


def forced_rollout_logprobs(
    *,
    model: Any,
    device: torch.device,
    pad_token_id: int,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    use_cache: bool,
) -> ForcedRolloutResult:
    """Run generate() along a fixed token sequence.

    The logits processor forces the same continuation in every
    experimental condition, while output logits remain the model's raw
    logits used for parity measurement.
    """

    if not prompt_token_ids:
        raise ValueError(
            "prompt_token_ids must not be empty"
        )

    if not generated_token_ids:
        raise ValueError(
            "generated_token_ids must not be empty"
        )

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    processor = ForcedContinuationProcessor(
        prompt_length=len(prompt_token_ids),
        token_ids=generated_token_ids,
    )

    synchronize_device(device)
    started = time.perf_counter()

    output: Any = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=len(generated_token_ids),
        min_new_tokens=len(generated_token_ids),
        do_sample=False,
        use_cache=use_cache,
        logits_processor=LogitsProcessorList(
            [processor]
        ),
        return_dict_in_generate=True,
        output_logits=True,
        pad_token_id=pad_token_id,
    )

    synchronize_device(device)
    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    generated_ids = cast(
        torch.Tensor,
        output.sequences[:, len(prompt_token_ids) :],
    )

    expected_ids = torch.tensor(
        [generated_token_ids],
        dtype=torch.long,
        device=device,
    )

    if not torch.equal(generated_ids, expected_ids):
        raise RuntimeError(
            "forced generation did not reproduce target tokens"
        )

    if output.logits is None:
        raise RuntimeError(
            "generate() did not return raw logits"
        )

    logprobs = gather_generated_logprobs(
        output.logits,
        generated_ids,
    )

    result = ForcedRolloutResult(
        use_cache=use_cache,
        token_logprobs=cast(
            list[float],
            logprobs[0].detach().cpu().tolist(),
        ),
        latency_ms=round(latency_ms, 3),
    )
    result.validate()

    return result
