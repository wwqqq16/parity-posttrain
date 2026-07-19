"""Tests for trainable model parameter selection."""

from __future__ import annotations

import pytest
import torch

from parity_posttrain.training import (
    prepare_trainable_parameters,
)


class TinyModel(torch.nn.Module):
    """Model with two independently named parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(
            2,
            2,
            bias=False,
        )
        self.second = torch.nn.Linear(
            2,
            2,
            bias=False,
        )


def test_prepares_only_requested_parameters() -> None:
    model = TinyModel()
    model.first.weight.requires_grad_(True)
    model.second.weight.requires_grad_(False)

    selection = prepare_trainable_parameters(
        model,
        ("second.weight",),
    )

    assert selection.names == ("second.weight",)
    assert selection.parameters == (
        model.second.weight,
    )
    assert selection.parameter_count == 4
    assert model.first.weight.requires_grad is False
    assert model.second.weight.requires_grad is True

    selection.restore_requires_grad()

    assert model.first.weight.requires_grad is True
    assert model.second.weight.requires_grad is False


def test_preserves_requested_parameter_order() -> None:
    model = TinyModel()

    selection = prepare_trainable_parameters(
        model,
        (
            "second.weight",
            "first.weight",
        ),
    )

    try:
        assert selection.names == (
            "second.weight",
            "first.weight",
        )
        assert selection.parameters == (
            model.second.weight,
            model.first.weight,
        )
    finally:
        selection.restore_requires_grad()


@pytest.mark.parametrize(
    ("parameter_names", "message"),
    [
        (
            (),
            "must not be empty",
        ),
        (
            ("",),
            "names must not be empty",
        ),
        (
            ("first.weight", "first.weight"),
            "names must be unique",
        ),
    ],
)
def test_rejects_invalid_parameter_names(
    parameter_names: tuple[str, ...],
    message: str,
) -> None:
    model = TinyModel()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        prepare_trainable_parameters(
            model,
            parameter_names,
        )


def test_rejects_unknown_parameter_name() -> None:
    model = TinyModel()

    with pytest.raises(
        ValueError,
        match="unknown trainable parameters",
    ):
        prepare_trainable_parameters(
            model,
            ("missing.weight",),
        )
