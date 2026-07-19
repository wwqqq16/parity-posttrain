"""Reproducibility metadata for experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform as platform_module
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_GIT_COMMIT_PATTERN = re.compile(
    r"^[0-9a-f]{40,64}$"
)
_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


@dataclass(frozen=True)
class ExperimentProvenance:
    """Environment and source metadata for one experiment."""

    git_commit: str | None
    source_artifact_sha256: str
    python_version: str
    platform: str
    pytorch_version: str | None
    transformers_version: str | None
    model_name: str
    model_revision: str | None
    seed: int

    def validate(self) -> None:
        """Validate provenance fields."""

        if (
            self.git_commit is not None
            and _GIT_COMMIT_PATTERN.fullmatch(
                self.git_commit
            )
            is None
        ):
            raise ValueError(
                "git_commit must be a hexadecimal "
                "Git object ID"
            )

        if _SHA256_PATTERN.fullmatch(
            self.source_artifact_sha256
        ) is None:
            raise ValueError(
                "source_artifact_sha256 must be a "
                "64-character hexadecimal digest"
            )

        if not self.python_version:
            raise ValueError(
                "python_version must not be empty"
            )

        if not self.platform:
            raise ValueError(
                "platform must not be empty"
            )

        if (
            self.pytorch_version is not None
            and not self.pytorch_version
        ):
            raise ValueError(
                "pytorch_version must not be empty"
            )

        if (
            self.transformers_version is not None
            and not self.transformers_version
        ):
            raise ValueError(
                "transformers_version must not be empty"
            )

        if not self.model_name:
            raise ValueError(
                "model_name must not be empty"
            )

        if (
            self.model_revision is not None
            and not self.model_revision
        ):
            raise ValueError(
                "model_revision must not be empty"
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize provenance metadata."""

        self.validate()

        return {
            "git_commit": self.git_commit,
            "source_artifact_sha256": (
                self.source_artifact_sha256
            ),
            "python_version": self.python_version,
            "platform": self.platform,
            "pytorch_version": self.pytorch_version,
            "transformers_version": (
                self.transformers_version
            ),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "seed": self.seed,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    if not path.is_file():
        raise ValueError(
            f"source artifact is not a file: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_git_commit(
    start_path: Path | None = None,
) -> str | None:
    """Resolve the current Git commit when available."""

    working_directory = (
        Path.cwd()
        if start_path is None
        else start_path
    )

    if working_directory.is_file():
        working_directory = working_directory.parent

    try:
        completed = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=working_directory,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    commit = completed.stdout.strip().lower()

    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        return None

    return commit


def installed_package_version(
    distribution_name: str,
) -> str | None:
    """Return an installed distribution version."""

    if not distribution_name:
        raise ValueError(
            "distribution_name must not be empty"
        )

    try:
        return importlib.metadata.version(
            distribution_name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def set_experiment_seed(seed: int) -> None:
    """Seed Python and PyTorch random generators."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required to set the "
            "experiment seed"
        ) from error

    random.seed(seed)
    torch.manual_seed(seed)


def build_experiment_provenance(
    *,
    source_artifact: Path,
    model_name: str,
    seed: int,
    model_revision: str | None = None,
    repository_path: Path | None = None,
) -> ExperimentProvenance:
    """Build validated experiment provenance."""

    provenance = ExperimentProvenance(
        git_commit=resolve_git_commit(
            repository_path
        ),
        source_artifact_sha256=sha256_file(
            source_artifact
        ),
        python_version=(
            platform_module.python_version()
        ),
        platform=platform_module.platform(),
        pytorch_version=installed_package_version(
            "torch"
        ),
        transformers_version=(
            installed_package_version(
                "transformers"
            )
        ),
        model_name=model_name,
        model_revision=model_revision,
        seed=seed,
    )
    provenance.validate()

    return provenance
