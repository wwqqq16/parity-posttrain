"""Tests for generated trajectory fingerprints."""

from __future__ import annotations

import pytest

from parity_posttrain.training import (
    fingerprint_generated_token_ids,
)


def test_fingerprint_is_deterministic() -> None:
    first = fingerprint_generated_token_ids(
        (
            (1, 2, 3),
            (4, 5),
        )
    )
    second = fingerprint_generated_token_ids(
        (
            [1, 2, 3],
            [4, 5],
        )
    )

    assert first == second
    assert len(first) == 64


def test_fingerprint_preserves_generation_boundaries() -> None:
    first = fingerprint_generated_token_ids(
        (
            (1, 2),
            (3,),
        )
    )
    second = fingerprint_generated_token_ids(
        (
            (1,),
            (2, 3),
        )
    )

    assert first != second


def test_fingerprint_preserves_token_order() -> None:
    first = fingerprint_generated_token_ids(
        ((1, 2, 3),)
    )
    second = fingerprint_generated_token_ids(
        ((1, 3, 2),)
    )

    assert first != second


@pytest.mark.parametrize(
    "generations",
    [
        (),
        ((),),
        ((1, -1),),
        ((1, True),),
    ],
)
def test_rejects_invalid_generations(
    generations: tuple[tuple[object, ...], ...],
) -> None:
    with pytest.raises(ValueError):
        fingerprint_generated_token_ids(
            generations  # type: ignore[arg-type]
        )
