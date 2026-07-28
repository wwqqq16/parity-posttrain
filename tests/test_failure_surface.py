from __future__ import annotations

from pathlib import Path

from enterprise_eval.failure_surface import (
    FailureSurfaceCell,
    build_failure_surface,
    failure_surface_markdown,
)
from enterprise_eval.models import Architecture
from enterprise_eval.runner import run_task_suite
from enterprise_eval.task_factory import (
    TaskFactoryConfig,
    generate_failure_surface_cases,
)


def test_failure_surface_exposes_position_dependent_recovery(
    tmp_path: Path,
) -> None:
    cases = generate_failure_surface_cases(
        TaskFactoryConfig(variants_per_cell=1)
    )
    rows = run_task_suite(
        cases,
        tmp_path,
        architecture=Architecture.PLANNER_CRITIC,
    )
    cells = build_failure_surface(rows)

    assert len(cells) == 21
    early_timeout = _get_cell(
        cells,
        difficulty="easy",
        failure_profile="transient_tool_timeout",
        injection_step=0,
    )
    payment_timeout = _get_cell(
        cells,
        difficulty="easy",
        failure_profile="transient_tool_timeout",
        injection_step=2,
    )

    assert early_timeout.injection_exposure_rate == 1.0
    assert early_timeout.success_rate == 0.0
    assert early_timeout.recovery_rate == 0.0
    assert payment_timeout.injection_exposure_rate == 1.0
    assert payment_timeout.success_rate == 1.0
    assert payment_timeout.recovery_rate == 1.0


def test_failure_surface_records_unreached_injections(
    tmp_path: Path,
) -> None:
    cases = generate_failure_surface_cases(
        TaskFactoryConfig(variants_per_cell=1)
    )
    rows = run_task_suite(
        cases,
        tmp_path,
        architecture=Architecture.SINGLE,
    )
    cells = build_failure_surface(rows)
    late_hard_timeout = _get_cell(
        cells,
        difficulty="hard",
        failure_profile="transient_tool_timeout",
        injection_step=2,
    )

    assert late_hard_timeout.injection_exposure_rate == 0.0
    assert late_hard_timeout.success_rate == 0.0
    assert late_hard_timeout.recovery_rate is None

    markdown = failure_surface_markdown(cells)
    assert "Injection step" in markdown
    assert "transient_tool_timeout" in markdown


def _get_cell(
    cells: list[FailureSurfaceCell],
    *,
    difficulty: str,
    failure_profile: str,
    injection_step: int,
) -> FailureSurfaceCell:
    return next(
        cell
        for cell in cells
        if cell.difficulty == difficulty
        and cell.failure_profile == failure_profile
        and cell.injection_step == injection_step
    )
