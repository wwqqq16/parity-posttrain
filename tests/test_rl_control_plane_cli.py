"""Tests for the deterministic RL control-plane demo."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def load_script_module() -> ModuleType:
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_rl_control_plane_demo.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_rl_control_plane_demo",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load run_rl_control_plane_demo.py"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_writes_replay_and_guard_artifact(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    output = tmp_path / "demo.json"

    exit_code = module.main(
        [
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
        == "enterprise-rl-control-plane-demo.v1"
    )
    assert payload["replay_identical"] is True
    assert payload["episode"]["total_return"] == 1.55
    assert (
        payload["unsafe_action_probe"][
            "transition"
        ]["reward"]
        == -0.5
    )
    assert (
        payload["unsafe_action_probe"][
            "refund_issued"
        ]
        is False
    )
