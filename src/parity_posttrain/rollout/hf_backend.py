"""Hugging Face rollout backend with token-level generation metadata."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class GenerationResult:
    """One generated assistant response and its rollout metadata."""

    model_name: str
    device: str
    prompt_text: str
    generated_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_token_logprobs: list[float]
    latency_ms: float
    tokens_per_second: float

    def validate(self) -> None:
        """Validate token-level generation metadata."""

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")

        if not self.generated_token_ids:
            raise ValueError("generation must contain at least one token")

        if len(self.generated_token_ids) != len(
            self.generated_token_logprobs
        ):
            raise ValueError(
                "generated token IDs and logprobs must have equal lengths"
            )

        if self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")

        if self.tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable generation record."""

        self.validate()
        return asdict(self)


def select_device() -> torch.device:
    """Select the best available local inference device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    """Select an inference dtype for the given device."""

    if device.type in {"mps", "cuda"}:
        return torch.float16

    return torch.float32


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator operations to finish."""

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def gather_generated_logprobs(
    scores: Sequence[torch.Tensor],
    generated_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract normalized logprobs for the generated token IDs.

    Args:
        scores:
            One score tensor for each generation step. Each tensor has
            shape ``[batch_size, vocabulary_size]``.
        generated_token_ids:
            Generated token IDs with shape
            ``[batch_size, generation_length]``.

    Returns:
        Token logprobs with shape
        ``[batch_size, generation_length]``.
    """

    if not scores:
        raise ValueError("generation scores must not be empty")

    stacked_scores = torch.stack(tuple(scores), dim=1)

    if stacked_scores.shape[:2] != generated_token_ids.shape:
        raise ValueError(
            "score steps and generated token IDs must have matching shapes"
        )

    log_probs = torch.log_softmax(
        stacked_scores.float(),
        dim=-1,
    )

    return torch.gather(
        log_probs,
        dim=-1,
        index=generated_token_ids.unsqueeze(-1),
    ).squeeze(-1)


class HuggingFaceRolloutBackend:
    """Generate assistant responses with a Transformers model."""

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        *,
        revision: str | None = None,
    ) -> None:
        """Load the tokenizer and causal language model."""

        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if (
            revision is not None
            and not revision.strip()
        ):
            raise ValueError(
                "revision must not be empty"
            )

        self.model_name = model_name
        self.model_revision = revision
        self.device = device or select_device()
        self.dtype = select_dtype(self.device)

        if revision is None:
            self.tokenizer: Any = (
                AutoTokenizer.from_pretrained(
                    model_name
                )
            )
            loaded_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=self.dtype,
                ),
            )
        else:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                )
            )
            loaded_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=self.dtype,
                    revision=revision,
                ),
            )

        loaded_model.to(self.device)
        loaded_model.eval()
        self.model = loaded_model

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        """Generate one deterministic assistant response."""

        if not messages:
            raise ValueError("messages must not be empty")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        prompt_text = cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

        encoded = cast(
            dict[str, torch.Tensor],
            self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ),
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        synchronize_device(self.device)
        started = time.perf_counter()

        output: Any = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        synchronize_device(self.device)
        latency_seconds = time.perf_counter() - started

        generated_ids = output.sequences[:, prompt_length:]

        if output.logits is None:
            raise RuntimeError(
                "model.generate did not return generation logits"
            )

        generated_logprobs = gather_generated_logprobs(
            output.logits,
            generated_ids,
        )

        generated_text = cast(
            str,
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            ),
        ).strip()

        token_count = int(generated_ids.shape[1])
        tokens_per_second = token_count / latency_seconds

        prompt_token_ids = cast(
            list[int],
            input_ids[0].detach().cpu().tolist(),
        )
        generated_token_ids = cast(
            list[int],
            generated_ids[0].detach().cpu().tolist(),
        )
        token_logprobs = cast(
            list[float],
            generated_logprobs[0].detach().cpu().tolist(),
        )

        result = GenerationResult(
            model_name=self.model_name,
            device=str(self.device),
            prompt_text=prompt_text,
            generated_text=generated_text,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_token_logprobs=token_logprobs,
            latency_ms=round(latency_seconds * 1000, 3),
            tokens_per_second=round(tokens_per_second, 3),
        )
        result.validate()

        return result
