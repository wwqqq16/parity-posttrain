"""Rollout-versus-trainer token logprob parity analysis."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch


@dataclass(slots=True)
class TokenParityRecord:
    """Parity metrics for one generated token."""

    index: int
    token_id: int
    token_text: str
    rollout_logprob: float
    trainer_logprob: float
    absolute_error: float


@dataclass(slots=True)
class ParityReport:
    """Aggregate token-level rollout/trainer parity report."""

    model_name: str
    device: str
    dtype: str
    token_count: int
    tolerance: float
    mean_absolute_error: float
    max_absolute_error: float
    p95_absolute_error: float
    tokens_over_tolerance: int
    within_tolerance: bool
    token_records: list[TokenParityRecord]

    def validate(self) -> None:
        """Validate the parity report."""

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

        if self.token_count <= 0:
            raise ValueError("token_count must be positive")

        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if len(self.token_records) != self.token_count:
            raise ValueError(
                "token record count must match token_count"
            )

        if self.mean_absolute_error < 0:
            raise ValueError(
                "mean_absolute_error must not be negative"
            )

        if self.max_absolute_error < 0:
            raise ValueError(
                "max_absolute_error must not be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parity report."""

        self.validate()
        return asdict(self)


def gather_teacher_forced_logprobs(
    logits: torch.Tensor,
    prompt_length: int,
    generated_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather generated-token logprobs from full-sequence logits.

    For a causal language model, token at sequence position ``t`` is
    predicted by logits at position ``t - 1``.
    """

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, sequence, vocabulary]"
        )

    if generated_token_ids.ndim != 2:
        raise ValueError(
            "generated_token_ids must have shape [batch, length]"
        )

    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    if logits.shape[0] != generated_token_ids.shape[0]:
        raise ValueError(
            "logits and generated token IDs must share batch size"
        )

    generation_length = generated_token_ids.shape[1]

    if generation_length <= 0:
        raise ValueError(
            "generated_token_ids must not be empty"
        )

    required_sequence_length = prompt_length + generation_length

    if logits.shape[1] < required_sequence_length:
        raise ValueError(
            "logit sequence is too short for prompt and generation"
        )

    prediction_start = prompt_length - 1
    prediction_end = prediction_start + generation_length

    prediction_logits = logits[
        :,
        prediction_start:prediction_end,
        :,
    ]

    if prediction_logits.shape[:2] != generated_token_ids.shape:
        raise ValueError(
            "teacher-forced logits and generated IDs are misaligned"
        )

    log_probs = torch.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )

    return torch.gather(
        log_probs,
        dim=-1,
        index=generated_token_ids.unsqueeze(-1),
    ).squeeze(-1)


def rescore_generated_tokens(
    *,
    model: Any,
    device: torch.device,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
) -> list[float]:
    """Teacher-force a generated sequence through the trainer model."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")

    if not generated_token_ids:
        raise ValueError(
            "generated_token_ids must not be empty"
        )

    full_token_ids = prompt_token_ids + generated_token_ids

    input_ids = torch.tensor(
        [full_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output: Any = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = cast(torch.Tensor, output.logits)

    token_ids = torch.tensor(
        [generated_token_ids],
        dtype=torch.long,
        device=device,
    )

    logprobs = gather_teacher_forced_logprobs(
        logits=logits,
        prompt_length=len(prompt_token_ids),
        generated_token_ids=token_ids,
    )

    return cast(
        list[float],
        logprobs[0].detach().cpu().tolist(),
    )


def build_parity_report(
    *,
    model_name: str,
    device: str,
    dtype: str,
    token_ids: list[int],
    token_texts: list[str],
    rollout_logprobs: list[float],
    trainer_logprobs: list[float],
    tolerance: float = 1e-3,
) -> ParityReport:
    """Compare rollout and trainer logprobs token by token."""

    token_count = len(token_ids)

    lengths = {
        token_count,
        len(token_texts),
        len(rollout_logprobs),
        len(trainer_logprobs),
    }

    if len(lengths) != 1:
        raise ValueError(
            "token IDs, texts, and logprob lists must align"
        )

    if token_count == 0:
        raise ValueError("parity comparison must not be empty")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    errors = [
        abs(rollout - trainer)
        for rollout, trainer in zip(
            rollout_logprobs,
            trainer_logprobs,
            strict=True,
        )
    ]

    sorted_errors = sorted(errors)
    p95_index = math.ceil(0.95 * token_count) - 1

    records = [
        TokenParityRecord(
            index=index,
            token_id=token_id,
            token_text=token_text,
            rollout_logprob=rollout_logprob,
            trainer_logprob=trainer_logprob,
            absolute_error=error,
        )
        for (
            index,
            token_id,
            token_text,
            rollout_logprob,
            trainer_logprob,
            error,
        ) in zip(
            range(token_count),
            token_ids,
            token_texts,
            rollout_logprobs,
            trainer_logprobs,
            errors,
            strict=True,
        )
    ]

    max_error = max(errors)

    report = ParityReport(
        model_name=model_name,
        device=device,
        dtype=dtype,
        token_count=token_count,
        tolerance=tolerance,
        mean_absolute_error=sum(errors) / token_count,
        max_absolute_error=max_error,
        p95_absolute_error=sorted_errors[p95_index],
        tokens_over_tolerance=sum(
            error > tolerance for error in errors
        ),
        within_tolerance=max_error <= tolerance,
        token_records=records,
    )
    report.validate()

    return report
