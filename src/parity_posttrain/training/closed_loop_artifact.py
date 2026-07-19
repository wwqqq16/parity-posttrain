"""Load closed-loop task snapshots from benchmark artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from parity_posttrain.training.closed_loop import (
    ClosedLoopTaskSnapshot,
)


def _require_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    """Return a mapping or raise a descriptive error."""

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field} must be a mapping"
        )

    return value


def _require_list(
    value: object,
    *,
    field: str,
) -> list[object]:
    """Return a list or raise a descriptive error."""

    if not isinstance(value, list):
        raise ValueError(
            f"{field} must be a list"
        )

    return value


def _require_str(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> str:
    """Read a non-empty string field."""

    value = mapping.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field}.{key} must be a non-empty string"
        )

    return value


def _require_bool(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> bool:
    """Read a boolean field."""

    value = mapping.get(key)

    if not isinstance(value, bool):
        raise ValueError(
            f"{field}.{key} must be boolean"
        )

    return value


def _require_int(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> int:
    """Read a non-negative integer field."""

    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field}.{key} must be a "
            "non-negative integer"
        )

    return value


def _require_float(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> float:
    """Read a numeric field as a float."""

    value = mapping.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{field}.{key} must be numeric"
        )

    return float(value)


def _snapshot_from_record(
    record: Mapping[str, object],
    *,
    field: str,
) -> ClosedLoopTaskSnapshot:
    """Construct one snapshot from a benchmark record."""

    snapshot = ClosedLoopTaskSnapshot(
        task_id=_require_str(
            record,
            "task_id",
            field=field,
        ),
        category=_require_str(
            record,
            "category",
            field=field,
        ),
        status=_require_str(
            record,
            "status",
            field=field,
        ),
        reward=_require_float(
            record,
            "reward",
            field=field,
        ),
        answer_correct=_require_bool(
            record,
            "answer_correct",
            field=field,
        ),
        generation_count=_require_int(
            record,
            "generation_count",
            field=field,
        ),
        generated_token_count=_require_int(
            record,
            "generated_token_count",
            field=field,
        ),
    )
    snapshot.validate()

    return snapshot


def extract_closed_loop_snapshots(
    payload: Mapping[str, object],
    *,
    task_ids: Sequence[str] | None = None,
) -> tuple[ClosedLoopTaskSnapshot, ...]:
    """Extract task snapshots from a benchmark payload."""

    task_values = _require_list(
        payload.get("tasks"),
        field="tasks",
    )

    if not task_values:
        raise ValueError(
            "tasks must not be empty"
        )

    snapshots_by_id: dict[
        str,
        ClosedLoopTaskSnapshot,
    ] = {}
    artifact_order: list[str] = []

    for task_index, task_value in enumerate(task_values):
        task_field = f"tasks[{task_index}]"
        task_result = _require_mapping(
            task_value,
            field=task_field,
        )
        record_field = (
            f"{task_field}.benchmark_record"
        )
        record = _require_mapping(
            task_result.get("benchmark_record"),
            field=record_field,
        )
        snapshot = _snapshot_from_record(
            record,
            field=record_field,
        )

        if snapshot.task_id in snapshots_by_id:
            raise ValueError(
                "benchmark task IDs must be unique"
            )

        snapshots_by_id[snapshot.task_id] = snapshot
        artifact_order.append(snapshot.task_id)

    if task_ids is None:
        selected_ids = tuple(artifact_order)
    else:
        selected_ids = tuple(task_ids)

        if not selected_ids:
            raise ValueError(
                "task_ids must not be empty"
            )

        if any(
            not isinstance(task_id, str)
            or not task_id
            for task_id in selected_ids
        ):
            raise ValueError(
                "task_ids must contain non-empty strings"
            )

        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(
                "task_ids must be unique"
            )

        missing_ids = tuple(
            task_id
            for task_id in selected_ids
            if task_id not in snapshots_by_id
        )

        if missing_ids:
            raise ValueError(
                "task IDs not found in artifact: "
                + ", ".join(missing_ids)
            )

    return tuple(
        snapshots_by_id[task_id]
        for task_id in selected_ids
    )


def load_closed_loop_snapshots(
    path: Path,
    *,
    task_ids: Sequence[str] | None = None,
) -> tuple[ClosedLoopTaskSnapshot, ...]:
    """Load task snapshots from a benchmark JSON file."""

    raw = json.loads(
        path.read_text(encoding="utf-8")
    )
    payload = _require_mapping(
        raw,
        field=str(path),
    )

    return extract_closed_loop_snapshots(
        payload,
        task_ids=task_ids,
    )
