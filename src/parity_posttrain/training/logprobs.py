"""Trainer-side logprob computation for padded batches."""

from __future__ import annotations

from typing import Any, cast

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)


def gather_masked_causal_logprobs(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.FloatTensor:
    """Gather causal token logprobs selected by a loss mask.

    Token at sequence position ``t`` is predicted by logits
    at position ``t - 1``. The returned tensor has the same
    shape as ``input_ids``. Unselected positions contain zero.
    """

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape "
            "[batch, sequence, vocabulary]"
        )

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence]"
        )

    if loss_mask.ndim != 2:
        raise ValueError(
            "loss_mask must have shape [batch, sequence]"
        )

    if input_ids.shape != loss_mask.shape:
        raise ValueError(
            "input_ids and loss_mask must have equal shapes"
        )

    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            "logits and input_ids must share batch and "
            "sequence dimensions"
        )

    if input_ids.shape[1] < 2:
        raise ValueError(
            "sequence length must be at least two"
        )

    if loss_mask.dtype != torch.bool:
        raise ValueError("loss_mask must have boolean dtype")

    if bool(torch.any(loss_mask[:, 0]).item()):
        raise ValueError(
            "loss_mask cannot select the first token"
        )

    selected_token_ids = input_ids[loss_mask]

    if selected_token_ids.numel() == 0:
        raise ValueError(
            "loss_mask must select at least one token"
        )

    vocabulary_size = logits.shape[-1]

    if bool(
        torch.any(selected_token_ids < 0).item()
    ) or bool(
        torch.any(
            selected_token_ids >= vocabulary_size
        ).item()
    ):
        raise ValueError(
            "selected token IDs must be inside the "
            "logit vocabulary"
        )

    prediction_logits = logits[:, :-1, :]
    target_token_ids = input_ids[:, 1:]
    target_mask = loss_mask[:, 1:]

    safe_target_token_ids = torch.where(
        target_mask,
        target_token_ids,
        torch.zeros_like(target_token_ids),
    )

    log_probs = torch.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )
    gathered_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_target_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    masked_logprobs = torch.where(
        target_mask,
        gathered_logprobs,
        torch.zeros_like(gathered_logprobs),
    )
    leading_zeros = torch.zeros(
        (input_ids.shape[0], 1),
        dtype=masked_logprobs.dtype,
        device=masked_logprobs.device,
    )

    result = torch.cat(
        (leading_zeros, masked_logprobs),
        dim=1,
    )

    return cast(torch.FloatTensor, result)


def rescore_training_batch(
    *,
    model: Any,
    batch: TrajectoryTrainingBatch,
) -> torch.FloatTensor:
    """Compute differentiable trainer logprobs for a batch."""

    batch.validate()

    output: Any = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        return_dict=True,
    )

    if not hasattr(output, "logits"):
        raise ValueError(
            "model output must provide logits"
        )

    logits = cast(torch.Tensor, output.logits)

    return gather_masked_causal_logprobs(
        logits=logits,
        input_ids=batch.input_ids,
        loss_mask=batch.loss_mask,
    )
