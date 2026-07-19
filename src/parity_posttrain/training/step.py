"""Single-step trajectory policy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
)
from parity_posttrain.training.logprobs import (
    rescore_training_batch,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
    clipped_policy_loss,
)


@dataclass(frozen=True)
class TrainingStepResult:
    """Detached metrics from one optimizer step."""

    loss: float
    mean_ratio: float
    approximate_kl: float
    clip_fraction: float
    trainable_token_count: int
    gradient_norm: float
    normalization: PolicyNormalization


def run_clipped_policy_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrajectoryTrainingBatch,
    clip_epsilon: float = 0.2,
    max_gradient_norm: float = 1.0,
    normalization: PolicyNormalization = "token",
) -> TrainingStepResult:
    """Run one differentiable clipped policy update."""

    if (
        not isinstance(max_gradient_norm, (int, float))
        or isinstance(max_gradient_norm, bool)
        or not math.isfinite(max_gradient_norm)
        or max_gradient_norm <= 0.0
    ):
        raise ValueError(
            "max_gradient_norm must be finite and positive"
        )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise ValueError(
            "model must contain trainable parameters"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    trainer_logprobs = rescore_training_batch(
        model=model,
        batch=batch,
    )
    objective = clipped_policy_loss(
        trainer_logprobs=trainer_logprobs,
        batch=batch,
        clip_epsilon=clip_epsilon,
        normalization=normalization,
    )

    if not bool(
        torch.isfinite(objective.loss).item()
    ):
        raise ValueError("policy loss must be finite")

    torch.autograd.backward(objective.loss)

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float(max_gradient_norm),
        error_if_nonfinite=True,
    )

    optimizer.step()

    return TrainingStepResult(
        loss=float(objective.loss.detach().item()),
        mean_ratio=objective.mean_ratio,
        approximate_kl=objective.approximate_kl,
        clip_fraction=objective.clip_fraction,
        trainable_token_count=(
            objective.trainable_token_count
        ),
        gradient_norm=float(
            gradient_norm.detach().item()
        ),
        normalization=normalization,
    )
