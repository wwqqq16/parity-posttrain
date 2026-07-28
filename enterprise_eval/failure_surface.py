"""Aggregate controlled task-factory runs into a failure surface."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from enterprise_eval.runner import BenchmarkRow


@dataclass(frozen=True)
class FailureSurfaceCell:
    """Metrics for one controlled failure-surface cell."""

    architecture: str
    difficulty: str
    failure_profile: str
    injection_step: int | None
    cases: int
    injection_exposure_rate: float | None
    success_rate: float
    policy_violation_rate: float
    recovery_rate: float | None
    mean_failure_step: float | None
    observed_failure_types: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_failure_surface(
    rows: list[BenchmarkRow],
) -> list[FailureSurfaceCell]:
    """Group benchmark rows across every controlled factory axis."""

    groups: dict[
        tuple[str, str, str, int | None],
        list[BenchmarkRow],
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.architecture,
                row.difficulty,
                row.failure_profile,
                row.injection_step,
            )
        ].append(row)

    cells = [
        _summarize_cell(key, selected)
        for key, selected in groups.items()
    ]
    return sorted(
        cells,
        key=lambda cell: (
            cell.architecture,
            _difficulty_order(cell.difficulty),
            cell.failure_profile,
            (
                -1
                if cell.injection_step is None
                else cell.injection_step
            ),
        ),
    )


def failure_surface_markdown(
    cells: list[FailureSurfaceCell],
) -> str:
    """Render a compact, interview-friendly Markdown result table."""

    lines = [
        "| Architecture | Difficulty | Failure type | Injection step | "
        "N | Exposed | Success | Recovery | Policy violations |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        step = (
            "—"
            if cell.injection_step is None
            else str(cell.injection_step)
        )
        lines.append(
            "| "
            f"{cell.architecture} | "
            f"{cell.difficulty} | "
            f"{cell.failure_profile} | "
            f"{step} | "
            f"{cell.cases} | "
            f"{_format_rate(cell.injection_exposure_rate)} | "
            f"{_format_rate(cell.success_rate)} | "
            f"{_format_rate(cell.recovery_rate)} | "
            f"{_format_rate(cell.policy_violation_rate)} |"
        )
    return "\n".join(lines)


def _summarize_cell(
    key: tuple[str, str, str, int | None],
    rows: list[BenchmarkRow],
) -> FailureSurfaceCell:
    architecture, difficulty, failure_profile, injection_step = key
    exposure_rate: float | None = None
    if failure_profile != "none":
        exposure_rate = (
            sum(row.injection_triggered for row in rows)
            / len(rows)
        )

    recovery_rows = [
        row for row in rows if row.recovered is not None
    ]
    recovery_rate: float | None = None
    if recovery_rows:
        recovery_rate = (
            sum(row.recovered is True for row in recovery_rows)
            / len(recovery_rows)
        )

    failure_steps = [
        row.failure_step
        for row in rows
        if row.failure_step is not None
    ]
    mean_failure_step: float | None = None
    if failure_steps:
        mean_failure_step = sum(failure_steps) / len(failure_steps)

    observed_failures = Counter(
        row.failure_type or "none"
        for row in rows
    )
    return FailureSurfaceCell(
        architecture=architecture,
        difficulty=difficulty,
        failure_profile=failure_profile,
        injection_step=injection_step,
        cases=len(rows),
        injection_exposure_rate=exposure_rate,
        success_rate=sum(row.success for row in rows) / len(rows),
        policy_violation_rate=(
            sum(row.policy_violation for row in rows)
            / len(rows)
        ),
        recovery_rate=recovery_rate,
        mean_failure_step=mean_failure_step,
        observed_failure_types=dict(
            sorted(observed_failures.items())
        ),
    )


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _difficulty_order(difficulty: str) -> int:
    return {
        "easy": 0,
        "medium": 1,
        "hard": 2,
    }.get(difficulty, 99)
