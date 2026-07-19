"""Tests for the training-comparison CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_training_comparison as cli


def test_cli_defaults_to_one_optimizer_step() -> None:
    args = cli.parse_args([])

    assert args.steps == 1
    assert args.seed == 0
    assert args.model_revision is None


def test_cli_parses_optimizer_step_count() -> None:
    args = cli.parse_args(
        [
            "--steps",
            "4",
        ]
    )

    assert args.steps == 4


def test_main_writes_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "benchmark.json"
    output = tmp_path / "comparison.json"
    artifact.write_text(
        "{}",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    examples = (
        SimpleNamespace(
            model_name="tiny-model",
        ),
    )

    class FakeTokenizer:
        """Minimal tokenizer for CLI testing."""

        pad_token_id = 0

    class FakeBackend:
        """Minimal Hugging Face backend."""

        def __init__(
            self,
            *,
            model_name: str,
            device: torch.device,
            revision: str | None = None,
        ) -> None:
            calls["backend_model_name"] = model_name
            calls["backend_device"] = device
            calls["revision"] = revision
            self.model_name = model_name
            self.model_revision = revision
            self.device = device
            self.dtype = torch.float32
            self.model = torch.nn.Linear(
                1,
                1,
                bias=False,
            )
            self.tokenizer = FakeTokenizer()

    class FakeProvenance:
        """Minimal provenance value."""

        def to_dict(self) -> dict[str, object]:
            return {
                "git_commit": "a" * 40,
                "source_artifact_sha256": "b" * 64,
                "model_name": "tiny-model",
                "model_revision": "revision-123",
                "seed": 17,
            }

    fake_summary = SimpleNamespace(
        model_name="tiny-model",
        device="cpu",
        dtype="torch.float32",
        trainable_parameter_count=1,
        trainable_parameter_names=(
            "model.norm.weight",
        ),
        optimizer_steps=2,
        tasks=(),
        rows=(),
    )

    monkeypatch.setattr(
        cli,
        "load_training_examples",
        lambda _: examples,
    )
    monkeypatch.setattr(
        cli,
        "select_training_examples",
        lambda selected, task_ids: selected,
    )
    monkeypatch.setattr(
        cli,
        "collate_training_examples",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli,
        "HuggingFaceRolloutBackend",
        FakeBackend,
    )
    monkeypatch.setattr(
        cli,
        "set_experiment_seed",
        lambda seed: calls.__setitem__(
            "seed",
            seed,
        ),
    )

    def fake_build_provenance(
        **kwargs: object,
    ) -> FakeProvenance:
        calls["provenance_kwargs"] = kwargs
        return FakeProvenance()

    monkeypatch.setattr(
        cli,
        "build_experiment_provenance",
        fake_build_provenance,
    )

    def fake_run_comparison(
        **kwargs: object,
    ) -> object:
        calls["run_kwargs"] = kwargs
        return fake_summary

    monkeypatch.setattr(
        cli,
        "run_training_comparison",
        fake_run_comparison,
    )
    monkeypatch.setattr(
        cli,
        "training_comparison_to_dict",
        lambda _: {
            "schema_version": 2,
            "source": {
                "model_name": "tiny-model",
            },
            "training": {
                "steps": 2,
            },
            "results": [],
        },
    )

    cli.main(
        [
            "--artifact",
            str(artifact),
            "--task-ids",
            "catalog_004",
            "shopping_004",
            "--steps",
            "2",
            "--seed",
            "17",
            "--model-revision",
            "revision-123",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 3
    assert payload["provenance"]["seed"] == 17
    assert payload["provenance"][
        "model_revision"
    ] == "revision-123"
    assert payload["source"]["model_name"] == (
        "tiny-model"
    )

    assert calls["seed"] == 17
    assert calls["revision"] == "revision-123"

    provenance_kwargs = calls["provenance_kwargs"]

    assert isinstance(provenance_kwargs, dict)
    assert provenance_kwargs[
        "source_artifact"
    ] == artifact
    assert provenance_kwargs["model_name"] == (
        "tiny-model"
    )
    assert provenance_kwargs[
        "model_revision"
    ] == "revision-123"
    assert provenance_kwargs["seed"] == 17

    run_kwargs = calls["run_kwargs"]

    assert isinstance(run_kwargs, dict)
    assert run_kwargs["steps"] == 2
    assert run_kwargs["model_name"] == "tiny-model"
    assert run_kwargs["source_artifact"] == str(
        artifact
    )
