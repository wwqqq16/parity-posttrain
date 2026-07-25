"""Convert model-backed enterprise artifacts into training examples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from parity_posttrain.training import (
    TrajectoryTrainingExample,
    build_trajectory_training_example,
)


def load_enterprise_model_artifact(path: Path) -> dict[str, Any]:
    """Load and minimally validate one enterprise model artifact."""

    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != "enterprise-agent-run.v1":
        raise ValueError("unsupported enterprise artifact schema")
    if not isinstance(payload.get("run"), dict):
        raise ValueError("enterprise artifact is missing run")
    if not isinstance(payload.get("evaluation"), dict):
        raise ValueError("enterprise artifact is missing evaluation")
    return payload


def get_model_generations(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return stored model generations in artifact order."""

    run = _require_mapping(payload, "run")
    metadata = _require_mapping(run, "metadata")
    raw_generations = metadata.get("model_generations")
    if not isinstance(raw_generations, list) or not raw_generations:
        raise ValueError("artifact contains no model_generations")

    generations: list[dict[str, Any]] = []
    for index, raw_generation in enumerate(raw_generations):
        if not isinstance(raw_generation, dict):
            raise ValueError(f"model generation {index} must be an object")
        generations.append(cast(dict[str, Any], raw_generation))
    return tuple(generations)


def get_model_generation(
    payload: dict[str, Any],
    *,
    turn_index: int,
) -> dict[str, Any]:
    """Return one stored model generation by turn index."""

    generations = get_model_generations(payload)
    if turn_index < 0 or turn_index >= len(generations):
        raise ValueError("turn index is out of range")
    return generations[turn_index]


def build_enterprise_training_examples(
    payload: dict[str, Any],
) -> tuple[TrajectoryTrainingExample, ...]:
    """Build validated token-level examples from a model-backed run."""

    run = _require_mapping(payload, "run")
    evaluation = _require_mapping(payload, "evaluation")

    case_id = _require_non_empty_string(run, "case_id")
    run_id = _require_non_empty_string(run, "run_id")
    task_id = f"{case_id}:{run_id}"
    reward = _require_finite_number(evaluation, "final_reward")
    status = _training_status(evaluation)

    examples: list[TrajectoryTrainingExample] = []
    for expected_turn_index, generation in enumerate(get_model_generations(payload)):
        stored_turn_index = generation.get("turn_index")
        if not isinstance(stored_turn_index, int):
            raise ValueError("model generation turn_index must be an integer")
        if stored_turn_index != expected_turn_index:
            raise ValueError("model generation turn indices must be contiguous")

        generated_token_ids = _require_int_list(
            generation,
            "generated_token_ids",
        )
        rollout_logprobs = _require_float_list(
            generation,
            "generated_token_logprobs",
        )
        if len(generated_token_ids) != len(rollout_logprobs):
            raise ValueError(
                "generated_token_ids and generated_token_logprobs "
                "must have equal length"
            )

        examples.append(
            build_trajectory_training_example(
                task_id=task_id,
                turn_index=stored_turn_index,
                status=status,
                model_name=_require_non_empty_string(generation, "model_name"),
                reward=reward,
                prompt_token_ids=_require_int_list(
                    generation,
                    "prompt_token_ids",
                ),
                generated_token_ids=generated_token_ids,
                rollout_logprobs=rollout_logprobs,
            )
        )

    return tuple(examples)


def build_enterprise_training_examples_from_artifact(
    path: Path,
) -> tuple[TrajectoryTrainingExample, ...]:
    """Load an artifact and build all model-generation examples."""

    return build_enterprise_training_examples(load_enterprise_model_artifact(path))


def _training_status(evaluation: dict[str, Any]) -> str:
    task_success = evaluation.get("task_success")
    if not isinstance(task_success, bool):
        raise ValueError("evaluation task_success must be a boolean")
    if task_success:
        return "completed"

    failure_type = evaluation.get("failure_type")
    if isinstance(failure_type, str) and failure_type.strip():
        return failure_type
    return "failed"


def _require_mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_non_empty_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_finite_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _require_int_list(payload: dict[str, Any], field: str) -> list[int]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain only integers")
    return cast(list[int], value)


def _require_float_list(payload: dict[str, Any], field: str) -> list[float]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")

    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field} must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain only finite numbers")
        normalized.append(number)
    return normalized
