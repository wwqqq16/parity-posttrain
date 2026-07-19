"""Load closed-loop task snapshots from benchmark artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from parity_posttrain.training.closed_loop import (
    ClosedLoopTaskSnapshot,
)
from parity_posttrain.training.trajectory_fingerprint import (
    fingerprint_generated_token_ids,
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


def _extract_generated_token_sequences(
    task_result: Mapping[str, object],
    *,
    field: str,
) -> tuple[tuple[int, ...], ...]:
    """Read generated token IDs while preserving turns."""

    run_field = f"{field}.run"
    run = _require_mapping(
        task_result.get("run"),
        field=run_field,
    )
    generations_field = f"{run_field}.generations"
    generations = _require_list(
        run.get("generations"),
        field=generations_field,
    )

    if not generations:
        raise ValueError(
            f"{generations_field} must not be empty"
        )

    token_sequences: list[tuple[int, ...]] = []

    for generation_index, generation_value in enumerate(
        generations
    ):
        generation_field = (
            f"{generations_field}[{generation_index}]"
        )
        generation = _require_mapping(
            generation_value,
            field=generation_field,
        )
        token_values = _require_list(
            generation.get("generated_token_ids"),
            field=(
                f"{generation_field}."
                "generated_token_ids"
            ),
        )

        token_ids: list[int] = []

        for token_index, token_value in enumerate(
            token_values
        ):
            if (
                isinstance(token_value, bool)
                or not isinstance(token_value, int)
                or token_value < 0
            ):
                raise ValueError(
                    f"{generation_field}."
                    "generated_token_ids"
                    f"[{token_index}] must be a "
                    "non-negative integer"
                )

            token_ids.append(token_value)

        if not token_ids:
            raise ValueError(
                f"{generation_field}."
                "generated_token_ids must not be empty"
            )

        token_sequences.append(tuple(token_ids))

    return tuple(token_sequences)


def _snapshot_from_record(
    record: Mapping[str, object],
    *,
    field: str,
    trajectory_fingerprint: str,
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
        trajectory_fingerprint=(
            trajectory_fingerprint
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
        token_sequences = (
            _extract_generated_token_sequences(
                task_result,
                field=task_field,
            )
        )
        snapshot = _snapshot_from_record(
            record,
            field=record_field,
            trajectory_fingerprint=(
                fingerprint_generated_token_ids(
                    token_sequences
                )
            ),
        )

        if (
            len(token_sequences)
            != snapshot.generation_count
        ):
            raise ValueError(
                f"{task_field} generation_count does not "
                "match run.generations"
            )

        actual_token_count = sum(
            len(token_ids)
            for token_ids in token_sequences
        )

        if (
            actual_token_count
            != snapshot.generated_token_count
        ):
            raise ValueError(
                f"{task_field} generated_token_count does "
                "not match run.generations"
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
