"""Build deterministic controlled-parity experiment matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ControlledParityDevice = Literal["cpu", "mps"]


@dataclass(frozen=True)
class ControlledParityMatrixCondition:
    """Describe one device/cache parity condition."""

    device: ControlledParityDevice
    use_cache: bool

    def validate(self) -> None:
        """Validate one matrix condition."""

        if self.device not in {"cpu", "mps"}:
            raise ValueError(
                "device must be either 'cpu' or 'mps'"
            )

        if not isinstance(self.use_cache, bool):
            raise ValueError(
                "use_cache must be a boolean"
            )

    @property
    def slug(self) -> str:
        """Return a stable filename-safe condition name."""

        cache_name = (
            "cache"
            if self.use_cache
            else "no_cache"
        )

        return f"{self.device}_{cache_name}"

    def output_path(
        self,
        output_directory: Path,
    ) -> Path:
        """Return the artifact path for this condition."""

        self.validate()

        return (
            output_directory
            / f"controlled_{self.slug}.json"
        )

    def cli_arguments(
        self,
        *,
        artifact: Path,
        task_id: str,
        turn_index: int,
        tolerance: float,
        output_directory: Path,
    ) -> list[str]:
        """Build arguments for run_controlled_parity.py."""

        self.validate()

        if not task_id.strip():
            raise ValueError(
                "task_id must not be empty"
            )

        if turn_index < 0:
            raise ValueError(
                "turn_index must be non-negative"
            )

        if tolerance <= 0:
            raise ValueError(
                "tolerance must be positive"
            )

        cache_argument = (
            "--use-cache"
            if self.use_cache
            else "--no-use-cache"
        )

        return [
            "--artifact",
            str(artifact),
            "--task-id",
            task_id,
            "--turn-index",
            str(turn_index),
            "--device",
            self.device,
            cache_argument,
            "--tolerance",
            format(tolerance, ".12g"),
            "--output",
            str(
                self.output_path(
                    output_directory
                )
            ),
        ]


def build_controlled_parity_matrix(
    *,
    include_mps: bool = True,
) -> tuple[
    ControlledParityMatrixCondition,
    ...,
]:
    """Build the standard device/cache experiment matrix."""

    conditions = [
        ControlledParityMatrixCondition(
            device="cpu",
            use_cache=True,
        ),
        ControlledParityMatrixCondition(
            device="cpu",
            use_cache=False,
        ),
    ]

    if include_mps:
        conditions.extend(
            [
                ControlledParityMatrixCondition(
                    device="mps",
                    use_cache=True,
                ),
                ControlledParityMatrixCondition(
                    device="mps",
                    use_cache=False,
                ),
            ]
        )

    for condition in conditions:
        condition.validate()

    return tuple(conditions)
