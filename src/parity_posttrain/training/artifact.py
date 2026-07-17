"""Extract training examples from agent benchmark artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from parity_posttrain.training.example import (
    TrajectoryTrainingExample,
    build_trajectory_training_example,
)


def _require_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    """Require a mapping with string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")

    if not all(isinstance(key, str) for key in value):
        raise ValueError(
            f"{field} must contain only string keys"
        )

    return cast(Mapping[str, object], value)


def _require_list(
    value: object,
    *,
    field: str,
) -> list[object]:
    """Require a JSON-style list."""

    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")

    return cast(list[object], value)


def _require_str(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> str:
    """Read a required non-empty string."""

    value = mapping.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field}.{key} must be a non-empty string"
        )

    return value


def _require_int(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> int:
    """Read a required integer."""

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field}.{key} must be an integer"
        )

    return value


def _require_float(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> float:
    """Read a required numeric value."""

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{field}.{key} must be numeric"
        )

    return float(value)


def _require_token_ids(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> list[int]:
    """Read a list of non-negative token IDs."""

    values = _require_list(
        mapping.get(key),
        field=f"{field}.{key}",
    )
    token_ids: list[int] = []

    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{field}.{key}[{index}] must be a "
                "non-negative integer"
            )

        token_ids.append(value)

    return token_ids


def _require_logprobs(
    mapping: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> list[float]:
    """Read a list of numeric token log-probabilities."""

    values = _require_list(
        mapping.get(key),
        field=f"{field}.{key}",
    )
    logprobs: list[float] = []

    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{field}.{key}[{index}] must be numeric"
            )

        logprobs.append(float(value))

    return logprobs


def extract_training_examples(
    payload: Mapping[str, object],
) -> tuple[TrajectoryTrainingExample, ...]:
    """Extract one training example per rollout generation."""

    tasks = _require_list(
        payload.get("tasks"),
        field="tasks",
    )

    if not tasks:
        raise ValueError("tasks must not be empty")

    examples: list[TrajectoryTrainingExample] = []

    for task_index, task_value in enumerate(tasks):
        task_field = f"tasks[{task_index}]"
        task_result = _require_mapping(
            task_value,
            field=task_field,
        )
        record = _require_mapping(
            task_result.get("benchmark_record"),
            field=f"{task_field}.benchmark_record",
        )
        run = _require_mapping(
            task_result.get("run"),
            field=f"{task_field}.run",
        )

        task_id = _require_str(
            record,
            "task_id",
            field=f"{task_field}.benchmark_record",
        )
        status = _require_str(
            record,
            "status",
            field=f"{task_field}.benchmark_record",
        )
        reward = _require_float(
            record,
            "reward",
            field=f"{task_field}.benchmark_record",
        )
        expected_generation_count = _require_int(
            record,
            "generation_count",
            field=f"{task_field}.benchmark_record",
        )
        expected_token_count = _require_int(
            record,
            "generated_token_count",
            field=f"{task_field}.benchmark_record",
        )

        generations = _require_list(
            run.get("generations"),
            field=f"{task_field}.run.generations",
        )

        if len(generations) != expected_generation_count:
            raise ValueError(
                f"{task_field} generation_count does not "
                "match run.generations"
            )

        actual_token_count = 0

        for turn_index, generation_value in enumerate(
            generations
        ):
            generation_field = (
                f"{task_field}.run.generations[{turn_index}]"
            )
            generation = _require_mapping(
                generation_value,
                field=generation_field,
            )

            generated_token_ids = _require_token_ids(
                generation,
                "generated_token_ids",
                field=generation_field,
            )

            example = build_trajectory_training_example(
                task_id=task_id,
                turn_index=turn_index,
                status=status,
                model_name=_require_str(
                    generation,
                    "model_name",
                    field=generation_field,
                ),
                reward=reward,
                prompt_token_ids=_require_token_ids(
                    generation,
                    "prompt_token_ids",
                    field=generation_field,
                ),
                generated_token_ids=generated_token_ids,
                rollout_logprobs=_require_logprobs(
                    generation,
                    "generated_token_logprobs",
                    field=generation_field,
                ),
            )
            examples.append(example)
            actual_token_count += len(generated_token_ids)

        if actual_token_count != expected_token_count:
            raise ValueError(
                f"{task_field} generated_token_count does "
                "not match its generations"
            )

    return tuple(examples)


def load_training_examples(
    path: Path,
) -> tuple[TrajectoryTrainingExample, ...]:
    """Load training examples from a benchmark JSON file."""

    raw = json.loads(
        path.read_text(encoding="utf-8")
    )
    payload = _require_mapping(
        raw,
        field=str(path),
    )

    return extract_training_examples(payload)
