"""Tests for closed-loop benchmark artifact loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parity_posttrain.training import (
    extract_closed_loop_snapshots,
    load_closed_loop_snapshots,
)


def make_record(
    task_id: str,
    *,
    category: str,
    status: str,
    reward: float,
    answer_correct: bool,
    generation_count: int = 1,
    generated_token_count: int = 10,
) -> dict[str, object]:
    """Create one benchmark record."""

    return {
        "task_id": task_id,
        "category": category,
        "status": status,
        "reward": reward,
        "answer_correct": answer_correct,
        "generation_count": generation_count,
        "generated_token_count": (
            generated_token_count
        ),
    }


def make_payload() -> dict[str, object]:
    """Create a two-task benchmark payload."""

    return {
        "tasks": [
            {
                "benchmark_record": make_record(
                    "catalog",
                    category="catalog",
                    status="completed",
                    reward=1.0,
                    answer_correct=True,
                    generation_count=2,
                    generated_token_count=42,
                )
            },
            {
                "benchmark_record": make_record(
                    "shopping",
                    category="shopping",
                    status="protocol_error",
                    reward=0.0,
                    answer_correct=False,
                    generation_count=1,
                    generated_token_count=47,
                )
            },
        ]
    }


def test_extracts_snapshots_in_artifact_order() -> None:
    snapshots = extract_closed_loop_snapshots(
        make_payload()
    )

    assert [
        snapshot.task_id
        for snapshot in snapshots
    ] == [
        "catalog",
        "shopping",
    ]
    assert snapshots[0].reward == 1.0
    assert snapshots[0].answer_correct is True
    assert snapshots[0].generation_count == 2
    assert snapshots[0].generated_token_count == 42
    assert snapshots[1].status == "protocol_error"


def test_selects_snapshots_in_requested_order() -> None:
    snapshots = extract_closed_loop_snapshots(
        make_payload(),
        task_ids=(
            "shopping",
            "catalog",
        ),
    )

    assert [
        snapshot.task_id
        for snapshot in snapshots
    ] == [
        "shopping",
        "catalog",
    ]


def test_loads_snapshots_from_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(make_payload()),
        encoding="utf-8",
    )

    snapshots = load_closed_loop_snapshots(
        path,
        task_ids=("catalog",),
    )

    assert len(snapshots) == 1
    assert snapshots[0].task_id == "catalog"


def test_rejects_missing_task_id() -> None:
    with pytest.raises(
        ValueError,
        match="not found in artifact",
    ):
        extract_closed_loop_snapshots(
            make_payload(),
            task_ids=("missing",),
        )


def test_rejects_duplicate_requested_task_ids() -> None:
    with pytest.raises(
        ValueError,
        match="task_ids must be unique",
    ):
        extract_closed_loop_snapshots(
            make_payload(),
            task_ids=(
                "catalog",
                "catalog",
            ),
        )


def test_rejects_duplicate_artifact_task_ids() -> None:
    record = make_record(
        "duplicate",
        category="catalog",
        status="completed",
        reward=1.0,
        answer_correct=True,
    )
    payload = {
        "tasks": [
            {"benchmark_record": record},
            {"benchmark_record": record},
        ]
    }

    with pytest.raises(
        ValueError,
        match="benchmark task IDs must be unique",
    ):
        extract_closed_loop_snapshots(payload)
