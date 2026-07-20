"""Execute controlled-parity matrix conditions."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from parity_posttrain.parity.matrix import (
    ControlledParityMatrixCondition,
    build_controlled_parity_matrix,
)


@dataclass(frozen=True)
class ControlledParityMatrixRun:
    """Describe one completed matrix subprocess."""

    condition: ControlledParityMatrixCondition
    command: tuple[str, ...]
    output_path: Path


def run_controlled_parity_matrix(
    *,
    artifact: Path,
    task_id: str,
    turn_index: int,
    tolerance: float,
    output_directory: Path,
    include_mps: bool = True,
    script_path: Path = Path(
        "scripts/run_controlled_parity.py"
    ),
    python_executable: str = sys.executable,
) -> tuple[ControlledParityMatrixRun, ...]:
    """Run every planned controlled-parity condition."""

    if not python_executable.strip():
        raise ValueError(
            "python_executable must not be empty"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_runs: list[
        ControlledParityMatrixRun
    ] = []

    for condition in build_controlled_parity_matrix(
        include_mps=include_mps
    ):
        output_path = condition.output_path(
            output_directory
        )

        arguments = condition.cli_arguments(
            artifact=artifact,
            task_id=task_id,
            turn_index=turn_index,
            tolerance=tolerance,
            output_directory=output_directory,
        )

        command = (
            python_executable,
            str(script_path),
            *arguments,
        )

        subprocess.run(
            command,
            check=True,
        )

        if not output_path.is_file():
            raise FileNotFoundError(
                "controlled-parity subprocess "
                "completed without writing "
                f"{output_path}"
            )

        completed_runs.append(
            ControlledParityMatrixRun(
                condition=condition,
                command=command,
                output_path=output_path,
            )
        )

    return tuple(completed_runs)
