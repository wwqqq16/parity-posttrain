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
    generation_count: int,
    generated_token_count: int,
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


def make_task_result(
    task_id: str,
    *,
    category: str,
    status: str,
    reward: float,
    answer_correct: bool,
    token_sequences: tuple[
        tuple[int, ...],
        ...,
    ],
) -> dict[str, object]:
    """Create one benchmark task result."""

    return {
        "benchmark_record": make_record(
            task_id,
            category=category,
            status=status,
            reward=reward,
            answer_correct=answer_correct,
            generation_count=len(token_sequences),
            generated_token_count=sum(
                len(token_ids)
                for token_ids in token_sequences
            ),
        ),
        "run": {
            "generations": [
                {
                    "generated_token_ids": list(
                        token_ids
                    )
                }
                for token_ids in token_sequences
            ]
        },
    }


def make_payload() -> dict[str, object]:
    """Create a two-task benchmark payload."""

    return {
        "tasks": [
            make_task_result(
                "catalog",
                category="catalog",
                status="completed",
                reward=1.0,
                answer_correct=True,
                token_sequences=(
                    tuple(range(20)),
                    tuple(range(20, 42)),
                ),
            ),
            make_task_result(
                "shopping",
                category="shopping",
                status="protocol_error",
                reward=0.0,
                answer_correct=False,
                token_sequences=(
                    tuple(range(47)),
                ),
            ),
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
    assert len(
        snapshots[0].trajectory_fingerprint
    ) == 64
    assert (
        snapshots[0].trajectory_fingerprint
        != snapshots[1].trajectory_fingerprint
    )


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
    task_result = make_task_result(
        "duplicate",
        category="catalog",
        status="completed",
        reward=1.0,
        answer_correct=True,
        token_sequences=((1,),),
    )
    payload = {
        "tasks": [
            task_result,
            task_result,
        ]
    }

    with pytest.raises(
        ValueError,
        match="benchmark task IDs must be unique",
    ):
        extract_closed_loop_snapshots(payload)


def test_rejects_record_token_count_mismatch() -> None:
    payload = make_payload()
    tasks = payload["tasks"]

    assert isinstance(tasks, list)
    first = tasks[0]

    assert isinstance(first, dict)
    record = first["benchmark_record"]

    assert isinstance(record, dict)
    record["generated_token_count"] = 999

    with pytest.raises(
        ValueError,
        match="generated_token_count does not match",
    ):
        extract_closed_loop_snapshots(payload)
