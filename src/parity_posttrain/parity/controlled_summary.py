"""Summarize fixed-sequence controlled parity experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class ControlledParityRow:
    """Metrics from one device, dtype, and cache condition."""

    task_id: str
    turn_index: int
    model_name: str
    device: str
    dtype: str
    use_cache: bool
    prompt_token_count: int
    generated_token_count: int
    token_count: int
    tolerance: float
    latency_ms: float
    mean_absolute_error: float
    max_absolute_error: float
    p95_absolute_error: float
    tokens_over_tolerance: int
    within_tolerance: bool

    def validate(self) -> None:
        """Validate one controlled parity row."""

        if not self.task_id:
            raise ValueError("task_id must not be empty")

        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if not self.device:
            raise ValueError("device must not be empty")

        if not self.dtype:
            raise ValueError("dtype must not be empty")

        if self.prompt_token_count <= 0:
            raise ValueError(
                "prompt_token_count must be positive"
            )

        if self.generated_token_count <= 0:
            raise ValueError(
                "generated_token_count must be positive"
            )

        if self.token_count <= 0:
            raise ValueError("token_count must be positive")

        if self.generated_token_count != self.token_count:
            raise ValueError(
                "generated_token_count and token_count "
                "must match"
            )

        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        error_values = (
            self.mean_absolute_error,
            self.max_absolute_error,
            self.p95_absolute_error,
        )

        if any(value < 0 for value in error_values):
            raise ValueError(
                "absolute-error metrics must be non-negative"
            )

        if not 0 <= self.tokens_over_tolerance <= self.token_count:
            raise ValueError(
                "tokens_over_tolerance must be between "
                "zero and token_count"
            )


@dataclass(frozen=True)
class ControlledParityComparison:
    """Cache versus no-cache comparison for one backend."""

    device: str
    dtype: str
    cached: ControlledParityRow
    uncached: ControlledParityRow
    no_cache_latency_factor: float
    max_error_reduction_factor: float | None
    mean_error_reduction_factor: float | None


@dataclass(frozen=True)
class ControlledParitySummary:
    """Summary of a controlled parity condition matrix."""

    rows: tuple[ControlledParityRow, ...]
    comparisons: tuple[ControlledParityComparison, ...]


def _require_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")

    return cast(Mapping[str, object], value)


def _require_str(
    mapping: Mapping[str, object],
    key: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")

    return value


def _require_bool(
    mapping: Mapping[str, object],
    key: str,
) -> bool:
    value = mapping.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")

    return value


def _require_int(
    mapping: Mapping[str, object],
    key: str,
) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")

    return value


def _require_float(
    mapping: Mapping[str, object],
    key: str,
) -> float:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(f"{key} must be numeric")

    return float(value)


def _reduction_factor(
    cached_error: float,
    uncached_error: float,
) -> float | None:
    if uncached_error == 0:
        return None

    return cached_error / uncached_error


def parse_controlled_parity_payload(
    payload: Mapping[str, object],
) -> ControlledParityRow:
    """Parse one controlled experiment JSON payload."""

    condition = _require_mapping(
        payload.get("condition"),
        field="condition",
    )
    forced_rollout = _require_mapping(
        payload.get("forced_rollout"),
        field="forced_rollout",
    )
    parity = _require_mapping(
        payload.get("parity"),
        field="parity",
    )
    source = _require_mapping(
        payload.get("source"),
        field="source",
    )

    device = _require_str(condition, "device")
    dtype = _require_str(condition, "dtype")
    use_cache = _require_bool(condition, "use_cache")

    if _require_bool(
        forced_rollout,
        "use_cache",
    ) != use_cache:
        raise ValueError(
            "condition and forced_rollout disagree "
            "about use_cache"
        )

    if _require_str(parity, "device") != device:
        raise ValueError(
            "condition and parity disagree about device"
        )

    if _require_str(parity, "dtype") != dtype:
        raise ValueError(
            "condition and parity disagree about dtype"
        )

    row = ControlledParityRow(
        task_id=_require_str(source, "task_id"),
        turn_index=_require_int(source, "turn_index"),
        model_name=_require_str(parity, "model_name"),
        device=device,
        dtype=dtype,
        use_cache=use_cache,
        prompt_token_count=_require_int(
            source,
            "prompt_token_count",
        ),
        generated_token_count=_require_int(
            source,
            "generated_token_count",
        ),
        token_count=_require_int(parity, "token_count"),
        tolerance=_require_float(parity, "tolerance"),
        latency_ms=_require_float(
            forced_rollout,
            "latency_ms",
        ),
        mean_absolute_error=_require_float(
            parity,
            "mean_absolute_error",
        ),
        max_absolute_error=_require_float(
            parity,
            "max_absolute_error",
        ),
        p95_absolute_error=_require_float(
            parity,
            "p95_absolute_error",
        ),
        tokens_over_tolerance=_require_int(
            parity,
            "tokens_over_tolerance",
        ),
        within_tolerance=_require_bool(
            parity,
            "within_tolerance",
        ),
    )
    row.validate()

    return row


def _source_identity(
    row: ControlledParityRow,
) -> tuple[str, int, str, int, int, float]:
    return (
        row.task_id,
        row.turn_index,
        row.model_name,
        row.prompt_token_count,
        row.generated_token_count,
        row.tolerance,
    )


def build_controlled_parity_summary(
    payloads: Iterable[Mapping[str, object]],
) -> ControlledParitySummary:
    """Build deterministic cache comparisons from payloads."""

    rows = tuple(
        parse_controlled_parity_payload(payload)
        for payload in payloads
    )

    if not rows:
        raise ValueError(
            "controlled parity payloads must not be empty"
        )

    reference_identity = _source_identity(rows[0])

    for row in rows[1:]:
        if _source_identity(row) != reference_identity:
            raise ValueError(
                "all conditions must use the same source "
                "sequence, model, and tolerance"
            )

    grouped: dict[
        tuple[str, str],
        dict[bool, ControlledParityRow],
    ] = {}

    for row in rows:
        key = (row.device, row.dtype)
        conditions = grouped.setdefault(key, {})

        if row.use_cache in conditions:
            raise ValueError(
                "duplicate controlled parity condition for "
                f"{row.device}/{row.dtype}/"
                f"use_cache={row.use_cache}"
            )

        conditions[row.use_cache] = row

    comparisons: list[ControlledParityComparison] = []

    for (device, dtype), conditions in sorted(
        grouped.items()
    ):
        if set(conditions) != {False, True}:
            raise ValueError(
                "each device and dtype must include both "
                "cache modes"
            )

        cached = conditions[True]
        uncached = conditions[False]

        comparisons.append(
            ControlledParityComparison(
                device=device,
                dtype=dtype,
                cached=cached,
                uncached=uncached,
                no_cache_latency_factor=(
                    uncached.latency_ms / cached.latency_ms
                    if cached.latency_ms > 0
                    else 0.0
                ),
                max_error_reduction_factor=(
                    _reduction_factor(
                        cached.max_absolute_error,
                        uncached.max_absolute_error,
                    )
                ),
                mean_error_reduction_factor=(
                    _reduction_factor(
                        cached.mean_absolute_error,
                        uncached.mean_absolute_error,
                    )
                ),
            )
        )

    ordered_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.device,
                row.dtype,
                0 if row.use_cache else 1,
            ),
        )
    )

    return ControlledParitySummary(
        rows=ordered_rows,
        comparisons=tuple(comparisons),
    )


def controlled_parity_summary_to_dict(
    summary: ControlledParitySummary,
) -> dict[str, object]:
    """Convert a controlled parity summary to JSON-safe data."""

    rows = [
        {
            "task_id": row.task_id,
            "turn_index": row.turn_index,
            "model_name": row.model_name,
            "device": row.device,
            "dtype": row.dtype,
            "use_cache": row.use_cache,
            "prompt_token_count": row.prompt_token_count,
            "generated_token_count": (
                row.generated_token_count
            ),
            "token_count": row.token_count,
            "tolerance": row.tolerance,
            "latency_ms": row.latency_ms,
            "mean_absolute_error": (
                row.mean_absolute_error
            ),
            "max_absolute_error": row.max_absolute_error,
            "p95_absolute_error": row.p95_absolute_error,
            "tokens_over_tolerance": (
                row.tokens_over_tolerance
            ),
            "within_tolerance": row.within_tolerance,
        }
        for row in summary.rows
    ]

    comparisons = [
        {
            "device": comparison.device,
            "dtype": comparison.dtype,
            "cached_max_absolute_error": (
                comparison.cached.max_absolute_error
            ),
            "uncached_max_absolute_error": (
                comparison.uncached.max_absolute_error
            ),
            "cached_mean_absolute_error": (
                comparison.cached.mean_absolute_error
            ),
            "uncached_mean_absolute_error": (
                comparison.uncached.mean_absolute_error
            ),
            "cached_latency_ms": (
                comparison.cached.latency_ms
            ),
            "uncached_latency_ms": (
                comparison.uncached.latency_ms
            ),
            "no_cache_latency_factor": (
                comparison.no_cache_latency_factor
            ),
            "max_error_reduction_factor": (
                comparison.max_error_reduction_factor
            ),
            "mean_error_reduction_factor": (
                comparison.mean_error_reduction_factor
            ),
        }
        for comparison in summary.comparisons
    ]

    return {
        "source": {
            "task_id": summary.rows[0].task_id,
            "turn_index": summary.rows[0].turn_index,
            "model_name": summary.rows[0].model_name,
            "prompt_token_count": (
                summary.rows[0].prompt_token_count
            ),
            "generated_token_count": (
                summary.rows[0].generated_token_count
            ),
            "tolerance": summary.rows[0].tolerance,
        },
        "rows": rows,
        "comparisons": comparisons,
    }
