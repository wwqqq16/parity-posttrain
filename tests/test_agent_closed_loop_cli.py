"""Tests for the agent closed-loop command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_agent_closed_loop as cli


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
    assert args.steps == 1
    assert args.learning_rate == pytest.approx(
        0.05
    )
    assert args.seed == 0
    assert args.model_revision is None


def test_select_tasks_preserves_requested_order() -> None:
    tasks = cli.select_tasks(
        (
            "shopping_004",
            "catalog_004",
        )
    )

    assert [
        task.task_id
        for task in tasks
    ] == [
        "shopping_004",
        "catalog_004",
    ]


def test_select_tasks_rejects_missing_task() -> None:
    with pytest.raises(
        ValueError,
        match="tasks were not found",
    ):
        cli.select_tasks(("missing_task",))


def test_main_writes_closed_loop_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "closed_loop.json"
    artifact = tmp_path / "benchmark.json"
    artifact.write_text(
        "{}",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeBackend:
        """Minimal backend used by the CLI test."""

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

    fake_summary = SimpleNamespace(
        model_name="tiny-model",
        normalization="trajectory",
        optimizer_steps=1,
        before=SimpleNamespace(
            task_count=2,
            total_reward=1.0,
            answer_accuracy=0.5,
        ),
        after=SimpleNamespace(
            total_reward=2.0,
            answer_accuracy=1.0,
        ),
        reward_delta=1.0,
        training_steps=(),
        tasks=(),
    )

    monkeypatch.setattr(
        cli,
        "resolve_model_name",
        lambda **_: "tiny-model",
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

    class FakeProvenance:
        """Minimal provenance object for CLI tests."""

        def to_dict(self) -> dict[str, object]:
            return {
                "git_commit": "a" * 40,
                "source_artifact_sha256": "b" * 64,
                "model_name": "tiny-model",
                "model_revision": "revision-123",
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

    def fake_run(**kwargs: object) -> object:
        calls["run_kwargs"] = kwargs
        return fake_summary

    monkeypatch.setattr(
        cli,
        "run_agent_closed_loop_experiment",
        fake_run,
    )
    monkeypatch.setattr(
        cli,
        "closed_loop_summary_to_dict",
        lambda _: {
            "schema_version": 1,
            "model_name": "tiny-model",
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

    assert payload["schema_version"] == 4
    assert payload["provenance"]["seed"] == 17
    assert payload["provenance"][
        "model_revision"
    ] == "revision-123"
    assert payload["model_name"] == "tiny-model"
    assert payload["experiment"]["task_ids"] == [
        "catalog_004",
        "shopping_004",
    ]
    assert payload["experiment"]["device"] == "cpu"

    run_kwargs = calls["run_kwargs"]

    assert isinstance(run_kwargs, dict)
    assert run_kwargs["steps"] == 2
    assert [
        task.task_id
        for task in run_kwargs["tasks"]
    ] == [
        "catalog_004",
        "shopping_004",
    ]

    assert calls["seed"] == 17
    assert calls["revision"] == "revision-123"

    provenance_kwargs = calls["provenance_kwargs"]

    assert isinstance(provenance_kwargs, dict)
    assert provenance_kwargs["source_artifact"] == artifact
    assert provenance_kwargs["model_name"] == "tiny-model"
    assert provenance_kwargs[
        "model_revision"
    ] == "revision-123"
    assert provenance_kwargs["seed"] == 17
