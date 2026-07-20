"""Tests for controlled-parity matrix execution."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from parity_posttrain.parity.matrix_runner import (
    run_controlled_parity_matrix,
)


def output_path_from_command(
    command: Sequence[str],
) -> Path:
    output_index = command.index("--output")

    return Path(command[output_index + 1])


def test_runner_executes_four_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True

        resolved_command = tuple(command)
        commands.append(resolved_command)

        output_path = output_path_from_command(
            resolved_command
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            "{}",
            encoding="utf-8",
        )

        return subprocess.CompletedProcess(
            args=resolved_command,
            returncode=0,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    runs = run_controlled_parity_matrix(
        artifact=Path("input.json"),
        task_id="basket_001",
        turn_index=0,
        tolerance=1e-3,
        output_directory=tmp_path / "matrix",
        seed=17,
        model_revision="revision-test",
        python_executable="python-test",
    )

    assert [
        run.condition.slug
        for run in runs
    ] == [
        "cpu_cache",
        "cpu_no_cache",
        "mps_cache",
        "mps_no_cache",
    ]

    assert len(commands) == 4

    assert all(
        command[:2]
        == (
            "python-test",
            "scripts/run_controlled_parity.py",
        )
        for command in commands
    )
    assert all(
        command[
            command.index("--seed") + 1
        ] == "17"
        for command in commands
    )
    assert all(
        command[
            command.index("--model-revision") + 1
        ] == "revision-test"
        for command in commands
    )

    assert "--use-cache" in commands[0]
    assert "--no-use-cache" in commands[1]
    assert "--use-cache" in commands[2]
    assert "--no-use-cache" in commands[3]

    assert all(
        run.output_path.is_file()
        for run in runs
    )


def test_runner_can_skip_mps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True

        resolved_command = tuple(command)
        commands.append(resolved_command)

        output_path = output_path_from_command(
            resolved_command
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            "{}",
            encoding="utf-8",
        )

        return subprocess.CompletedProcess(
            args=resolved_command,
            returncode=0,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    runs = run_controlled_parity_matrix(
        artifact=Path("input.json"),
        task_id="basket_001",
        turn_index=0,
        tolerance=1e-3,
        output_directory=tmp_path / "matrix",
        include_mps=False,
    )

    assert [
        run.condition.slug
        for run in runs
    ] == [
        "cpu_cache",
        "cpu_no_cache",
    ]

    assert len(commands) == 2
    assert all(
        "mps" not in command
        for command in commands
    )
    assert all(
        command[
            command.index("--seed") + 1
        ] == "0"
        for command in commands
    )
    assert all(
        "--model-revision" not in command
        for command in commands
    )


def test_runner_requires_subprocess_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "completed without writing"
        ),
    ):
        run_controlled_parity_matrix(
            artifact=Path("input.json"),
            task_id="basket_001",
            turn_index=0,
            tolerance=1e-3,
            output_directory=tmp_path / "matrix",
        )


def test_runner_rejects_empty_python_executable(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "python_executable must not be empty"
        ),
    ):
        run_controlled_parity_matrix(
            artifact=Path("input.json"),
            task_id="basket_001",
            turn_index=0,
            tolerance=1e-3,
            output_directory=tmp_path / "matrix",
            python_executable=" ",
        )
