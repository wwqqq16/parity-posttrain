"""Tests for the controlled-parity CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_controlled_parity.py"
    )

    spec = importlib.util.spec_from_file_location(
        "run_controlled_parity",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load run_controlled_parity.py"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_parse_args_accepts_provenance_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_controlled_parity.py",
            "--artifact",
            "artifacts/agent_benchmark.json",
            "--task-id",
            "basket_001",
            "--turn-index",
            "0",
            "--device",
            "cpu",
            "--use-cache",
            "--tolerance",
            "0.001",
            "--seed",
            "17",
            "--model-revision",
            "revision-123",
            "--output",
            "artifacts/result.json",
        ],
    )

    args = module.parse_args()

    assert args.seed == 17
    assert args.model_revision == "revision-123"
    assert args.device == "cpu"
    assert args.use_cache is True


def test_parse_args_uses_provenance_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_controlled_parity.py",
            "--artifact",
            "input.json",
            "--task-id",
            "basket_001",
            "--device",
            "cpu",
            "--no-use-cache",
            "--output",
            "output.json",
        ],
    )

    args = module.parse_args()

    assert args.seed == 0
    assert args.model_revision is None
    assert args.use_cache is False
