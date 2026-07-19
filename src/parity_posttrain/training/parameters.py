"""Utilities for selecting trainable model parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainableParameterSelection:
    """A temporary model trainability configuration."""

    names: tuple[str, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    all_parameters: tuple[torch.nn.Parameter, ...]
    original_requires_grad: tuple[bool, ...]

    @property
    def parameter_count(self) -> int:
        """Return the selected scalar parameter count."""

        return sum(
            parameter.numel()
            for parameter in self.parameters
        )

    def restore_requires_grad(self) -> None:
        """Restore the model's original gradient flags."""

        if (
            len(self.all_parameters)
            != len(self.original_requires_grad)
        ):
            raise ValueError(
                "parameter and requires_grad state "
                "lengths must match"
            )

        for parameter, requires_grad in zip(
            self.all_parameters,
            self.original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)


def prepare_trainable_parameters(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> TrainableParameterSelection:
    """Resolve named parameters and freeze all others."""

    requested = tuple(parameter_names)

    if not requested:
        raise ValueError(
            "trainable_parameter_names must not be empty"
        )

    if any(
        not isinstance(name, str)
        or not name.strip()
        for name in requested
    ):
        raise ValueError(
            "trainable parameter names must not be empty"
        )

    if len(set(requested)) != len(requested):
        raise ValueError(
            "trainable parameter names must be unique"
        )

    available = dict(model.named_parameters())
    missing = tuple(
        name
        for name in requested
        if name not in available
    )

    if missing:
        raise ValueError(
            "unknown trainable parameters: "
            + ", ".join(missing)
        )

    selected_parameters = tuple(
        available[name]
        for name in requested
    )
    all_parameters = tuple(model.parameters())
    original_requires_grad = tuple(
        parameter.requires_grad
        for parameter in all_parameters
    )

    try:
        for parameter in all_parameters:
            parameter.requires_grad_(False)

        for parameter in selected_parameters:
            parameter.requires_grad_(True)
    except Exception:
        for parameter, requires_grad in zip(
            all_parameters,
            original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)

        raise

    return TrainableParameterSelection(
        names=requested,
        parameters=selected_parameters,
        all_parameters=all_parameters,
        original_requires_grad=(
            original_requires_grad
        ),
    )
