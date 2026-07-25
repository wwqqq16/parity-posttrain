from __future__ import annotations

from pathlib import Path

from enterprise_eval.cases import CASES
from enterprise_eval.models import Architecture
from enterprise_eval.runner import run_benchmark, summarize_rows


def test_benchmark_has_fourteen_cases_across_three_levels(tmp_path: Path) -> None:
    assert len(CASES) == 14
    levels = {case.difficulty.value for case in CASES.values()}
    assert levels == {"easy", "medium", "hard"}

    rows = run_benchmark(tmp_path, architecture=Architecture.PLANNER_CRITIC)
    assert len(rows) == 14
    assert all(row.success for row in rows)


def test_single_agent_is_strong_on_easy_cases(tmp_path: Path) -> None:
    rows = run_benchmark(tmp_path, architecture=Architecture.SINGLE)
    easy = [row for row in rows if row.difficulty == "easy"]
    assert easy
    assert all(row.success for row in easy)


def test_planner_critic_handles_corrected_order_id(tmp_path: Path) -> None:
    single = run_benchmark(tmp_path / "single", architecture=Architecture.SINGLE)
    multi = run_benchmark(
        tmp_path / "multi",
        architecture=Architecture.PLANNER_CRITIC,
    )
    single_row = next(row for row in single if row.case_id == "wrong_order_then_corrected")
    multi_row = next(row for row in multi if row.case_id == "wrong_order_then_corrected")

    assert not single_row.success
    assert single_row.failure_type == "invalid_tool_call"
    assert multi_row.success


def test_planner_critic_rechecks_stale_payment_status(tmp_path: Path) -> None:
    single = run_benchmark(tmp_path / "single", architecture=Architecture.SINGLE)
    multi = run_benchmark(
        tmp_path / "multi",
        architecture=Architecture.PLANNER_CRITIC,
    )
    single_row = next(row for row in single if row.case_id == "stale_payment_status")
    multi_row = next(row for row in multi if row.case_id == "stale_payment_status")

    assert not single_row.success
    assert single_row.policy_violation
    assert multi_row.success


def test_planner_critic_respects_latest_user_intent(tmp_path: Path) -> None:
    single = run_benchmark(tmp_path / "single", architecture=Architecture.SINGLE)
    multi = run_benchmark(
        tmp_path / "multi",
        architecture=Architecture.PLANNER_CRITIC,
    )
    single_row = next(row for row in single if row.case_id == "user_withdraws_refund")
    multi_row = next(row for row in multi if row.case_id == "user_withdraws_refund")

    assert not single_row.success
    assert single_row.policy_violation
    assert multi_row.success


def test_summary_exposes_coordination_overhead(tmp_path: Path) -> None:
    rows = []
    rows.extend(run_benchmark(tmp_path / "single", architecture=Architecture.SINGLE))
    rows.extend(
        run_benchmark(
            tmp_path / "multi",
            architecture=Architecture.PLANNER_CRITIC,
        )
    )
    summaries = summarize_rows(rows)
    single_all = next(
        summary
        for summary in summaries
        if summary.architecture == "single" and summary.difficulty == "all"
    )
    multi_all = next(
        summary
        for summary in summaries
        if summary.architecture == "planner-critic" and summary.difficulty == "all"
    )

    assert single_all.average_component_calls == 1.0
    assert multi_all.average_component_calls == 3.0
    assert multi_all.success_rate > single_all.success_rate
