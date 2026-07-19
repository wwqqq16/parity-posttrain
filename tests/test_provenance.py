"""Tests for experiment provenance metadata."""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path

import pytest
import torch

import parity_posttrain.provenance as provenance_module
from parity_posttrain.provenance import (
    ExperimentProvenance,
    build_experiment_provenance,
    resolve_git_commit,
    set_experiment_seed,
    sha256_file,
)


def test_sha256_file_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    content = b'{"result": 1}\n'
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(
        content
    ).hexdigest()


def test_sha256_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="source artifact is not a file",
    ):
        sha256_file(
            tmp_path / "missing.json"
        )


def test_resolve_git_commit_returns_normalized_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "A" * 40

    def fake_run(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del args

        assert kwargs["cwd"] == tmp_path
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        assert kwargs["text"] is True

        return subprocess.CompletedProcess(
            args=[
                "git",
                "rev-parse",
                "HEAD",
            ],
            returncode=0,
            stdout=expected + "\n",
            stderr="",
        )

    monkeypatch.setattr(
        provenance_module.subprocess,
        "run",
        fake_run,
    )

    assert resolve_git_commit(
        tmp_path
    ) == expected.lower()


def test_resolve_git_commit_returns_none_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del args, kwargs

        return subprocess.CompletedProcess(
            args=[
                "git",
                "rev-parse",
                "HEAD",
            ],
            returncode=128,
            stdout="",
            stderr="not a repository",
        )

    monkeypatch.setattr(
        provenance_module.subprocess,
        "run",
        fake_run,
    )

    assert resolve_git_commit(tmp_path) is None


def test_set_experiment_seed_is_repeatable() -> None:
    set_experiment_seed(17)

    first_python = random.random()
    first_torch = torch.rand(3)

    set_experiment_seed(17)

    assert random.random() == first_python
    assert torch.equal(
        torch.rand(3),
        first_torch,
    )


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        True,
        1.5,
    ],
)
def test_set_experiment_seed_rejects_invalid_seed(
    seed: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="seed must be a non-negative integer",
    ):
        set_experiment_seed(
            seed,  # type: ignore[arg-type]
        )


def test_build_experiment_provenance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        '{"schema_version": 1}\n',
        encoding="utf-8",
    )

    result = build_experiment_provenance(
        source_artifact=artifact,
        model_name="test-model",
        requested_model_revision=None,
        resolved_model_revision="c" * 40,
        seed=0,
        repository_path=tmp_path,
    )
    payload = result.to_dict()

    assert result.git_commit is None
    assert result.source_artifact_sha256 == (
        sha256_file(artifact)
    )
    assert result.model_name == "test-model"
    assert result.requested_model_revision is None
    assert result.model_revision is None
    assert result.resolved_model_revision == (
        "c" * 40
    )
    assert result.seed == 0
    assert result.python_version
    assert result.platform

    assert payload["model_name"] == "test-model"
    assert payload[
        "requested_model_revision"
    ] is None
    assert payload[
        "resolved_model_revision"
    ] == "c" * 40
    assert payload["seed"] == 0
    assert payload["source_artifact_sha256"] == (
        result.source_artifact_sha256
    )


def test_provenance_rejects_invalid_digest() -> None:
    result = ExperimentProvenance(
        git_commit=None,
        source_artifact_sha256="invalid",
        python_version="3.12.0",
        platform="test-platform",
        pytorch_version="2.0.0",
        transformers_version="5.0.0",
        model_name="test-model",
        requested_model_revision=None,
        resolved_model_revision=None,
        seed=0,
    )

    with pytest.raises(
        ValueError,
        match="64-character hexadecimal digest",
    ):
        result.validate()


def test_provenance_rejects_invalid_resolved_revision(
) -> None:
    result = ExperimentProvenance(
        git_commit=None,
        source_artifact_sha256="a" * 64,
        python_version="3.12.0",
        platform="test-platform",
        pytorch_version="2.0.0",
        transformers_version="5.0.0",
        model_name="test-model",
        requested_model_revision="main",
        resolved_model_revision="not-a-commit",
        seed=0,
    )

    with pytest.raises(
        ValueError,
        match="hexadecimal commit ID",
    ):
        result.validate()
