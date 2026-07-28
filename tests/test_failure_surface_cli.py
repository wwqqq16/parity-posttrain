"""Tests for the controlled enterprise failure-surface CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_failure_surface.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_failure_surface",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load run_failure_surface.py"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_validated_failure_surface(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    output = tmp_path / "summary.json"

    exit_code = module.main(
        [
            "--architecture",
            "single",
            "--variants-per-cell",
            "1",
            "--output-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert (
        payload["schema_version"]
        == "enterprise-failure-surface.v1"
    )
    assert payload["validation"] == {
        "case_count": 21,
        "coverage_cell_count": 21,
        "cases_per_cell": 1,
        "oracle_success_rate": 1.0,
    }
    assert payload["run_count"] == 21
    assert len(payload["cells"]) == 21
