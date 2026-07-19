"""Tests for the token-clipping diagnostic CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_token_clipping_diagnostics as cli
from parity_posttrain.training import (
    TokenClippingDiagnostic,
)


def test_parse_args_defaults() -> None:
    args = cli.parse_args([])

    assert args.artifact == Path(
        "artifacts/agent_benchmark.json"
    )
    assert args.task_ids == [
        "catalog_004",
        "shopping_004",
    ]
    assert args.device == "cpu"
    assert args.normalization == "trajectory"
    assert args.steps == 10
    assert args.learning_rate == pytest.approx(
        0.003
    )
    assert args.seed == 0
    assert args.model_revision is None


def test_main_writes_token_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "benchmark.json"
    output = tmp_path / "diagnostics.json"
    artifact.write_text(
        "{}",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeTokenizer:
        """Minimal tokenizer for CLI tests."""

        pad_token_id = 0

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
        ) -> str:
            assert skip_special_tokens is False
            return f"token-{token_ids[0]}"

    class FakeBackend:
        """Minimal model backend for CLI tests."""

        def __init__(
            self,
            *,
            model_name: str,
            device: torch.device,
            revision: str | None = None,
        ) -> None:
            calls["model_name"] = model_name
            calls["device"] = device
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

    class FakeSelection:
        """Minimal trainable parameter selection."""

        def __init__(
            self,
            model: torch.nn.Module,
        ) -> None:
            self.parameters = tuple(
                model.parameters()
            )
            self.restored = False

        def restore_requires_grad(self) -> None:
            self.restored = True
            calls["restored"] = True

    examples = (
        SimpleNamespace(
            model_name="tiny-model",
        ),
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
        lambda *args, **kwargs: SimpleNamespace(
            trainable_token_count=1,
        ),
    )
    monkeypatch.setattr(
        cli,
        "HuggingFaceRolloutBackend",
        FakeBackend,
    )
    monkeypatch.setattr(
        cli,
        "prepare_trainable_parameters",
        lambda model, names: FakeSelection(model),
    )
    monkeypatch.setattr(
        cli,
        "set_experiment_seed",
        lambda seed: calls.__setitem__(
            "seed",
            seed,
        ),
    )

    class FakeProvenance:
        """Minimal provenance object for CLI tests."""

        def to_dict(self) -> dict[str, object]:
            return {
                "git_commit": "a" * 40,
                "source_artifact_sha256": "b" * 64,
                "model_name": "tiny-model",
                "model_revision": "test-revision",
                "seed": 17,
            }

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

    diagnostic = TokenClippingDiagnostic(
        flat_token_index=0,
        batch_row=0,
        task_id="catalog_004",
        turn_index=1,
        generated_token_index=11,
        sequence_position=634,
        token_id=25951,
        advantage=0.5,
        old_logprob=-1.99,
        current_logprob=-1.80,
        ratio=1.21,
        out_of_range=True,
        active_clipped=True,
        clip_direction="positive_high",
    )

    def fake_training(**kwargs: object) -> object:
        observer = kwargs["observer"]

        assert callable(observer)
        observer(1, (diagnostic,))
        calls["training_kwargs"] = kwargs

        return SimpleNamespace(
            parameter_deltas=(0.001,),
            final_parameter_delta=0.001,
        )

    monkeypatch.setattr(
        cli,
        "run_clipped_policy_training",
        fake_training,
    )

    cli.main(
        [
            "--artifact",
            str(artifact),
            "--task-ids",
            "catalog_004",
            "--trainable-parameters",
            "weight",
            "--steps",
            "1",
            "--seed",
            "17",
            "--model-revision",
            "test-revision",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 2
    assert payload["provenance"]["seed"] == 17
    assert payload["provenance"][
        "model_revision"
    ] == "test-revision"
    assert payload["model_name"] == "tiny-model"
    assert payload["task_ids"] == [
        "catalog_004",
    ]
    assert payload["steps"][0][
        "active_clip_count"
    ] == 1

    token = payload["steps"][0][
        "out_of_range_tokens"
    ][0]

    assert token["task_id"] == "catalog_004"
    assert token["token_id"] == 25951
    assert token["token_text"] == "token-25951"
    assert token["active_clipped"] is True
    assert token["clip_direction"] == "positive_high"
    assert calls["restored"] is True
    assert calls["seed"] == 17
    assert calls["revision"] == "test-revision"

    provenance_kwargs = calls["provenance_kwargs"]

    assert isinstance(provenance_kwargs, dict)
    assert provenance_kwargs["source_artifact"] == artifact
    assert provenance_kwargs["model_name"] == "tiny-model"
    assert provenance_kwargs[
        "model_revision"
    ] == "test-revision"
    assert provenance_kwargs["seed"] == 17
