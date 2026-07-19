"""Tests for the training-comparison CLI."""

from __future__ import annotations

from scripts.run_training_comparison import parse_args


def test_cli_defaults_to_one_optimizer_step() -> None:
    args = parse_args([])

    assert args.steps == 1


def test_cli_parses_optimizer_step_count() -> None:
    args = parse_args(
        [
            "--steps",
            "4",
        ]
    )

    assert args.steps == 4
