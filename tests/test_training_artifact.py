"""Tests for extracting training examples from artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from parity_posttrain.training.artifact import (
    extract_training_examples,
    load_training_examples,
)


def make_generation(
    *,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_token_logprobs: list[float],
) -> dict[str, object]:
    """Create one benchmark generation."""

    return {
        "device": "cpu",
        "generated_text": "test",
        "generated_token_ids": generated_token_ids,
        "generated_token_logprobs": (
            generated_token_logprobs
        ),
        "latency_ms": 1.0,
        "model_name": "test-model",
        "prompt_text": "prompt",
        "prompt_token_ids": prompt_token_ids,
        "tokens_per_second": 1.0,
    }


def make_payload() -> dict[str, object]:
    """Create a representative benchmark artifact."""

    generations = [
        make_generation(
            prompt_token_ids=[10, 11],
            generated_token_ids=[20, 21],
            generated_token_logprobs=[-0.1, -0.2],
        ),
        make_generation(
            prompt_token_ids=[10, 11, 20, 21],
            generated_token_ids=[30],
            generated_token_logprobs=[-0.3],
        ),
    ]

    return {
        "model": "test-model",
        "summary": {},
        "task_ids": ["calculator_001"],
        "tolerance": 1e-3,
        "tasks": [
            {
                "benchmark_record": {
                    "task_id": "calculator_001",
                    "status": "completed",
                    "reward": 1.0,
                    "generation_count": 2,
                    "generated_token_count": 3,
                },
                "evaluation": {},
                "parity": {},
                "run": {
                    "error": None,
                    "generations": generations,
                    "status": "completed",
                    "trajectory": {},
                },
                "task": {},
            }
        ],
    }


def first_record(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return the first benchmark record."""

    tasks = cast(list[object], payload["tasks"])
    task = cast(dict[str, object], tasks[0])

    return cast(
        dict[str, object],
        task["benchmark_record"],
    )


def first_generation(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return the first rollout generation."""

    tasks = cast(list[object], payload["tasks"])
    task = cast(dict[str, object], tasks[0])
    run = cast(dict[str, object], task["run"])
    generations = cast(list[object], run["generations"])

    return cast(
        dict[str, object],
        generations[0],
    )


def test_extract_training_examples() -> None:
    examples = extract_training_examples(
        make_payload()
    )

    assert len(examples) == 2

    first, second = examples

    assert first.task_id == "calculator_001"
    assert first.turn_index == 0
    assert first.status == "completed"
    assert first.model_name == "test-model"
    assert first.reward == 1.0
    assert first.prompt_token_ids == (10, 11)
    assert first.generated_token_ids == (20, 21)
    assert first.rollout_logprobs == (-0.1, -0.2)

    assert second.turn_index == 1
    assert second.prompt_token_ids == (
        10,
        11,
        20,
        21,
    )
    assert second.generated_token_ids == (30,)
    assert second.rollout_logprobs == (-0.3,)


def test_load_training_examples(
    tmp_path: Path,
) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(make_payload()),
        encoding="utf-8",
    )

    examples = load_training_examples(path)

    assert len(examples) == 2
    assert examples[0].task_id == "calculator_001"


def test_rejects_generation_count_mismatch() -> None:
    payload = make_payload()
    first_record(payload)["generation_count"] = 1

    with pytest.raises(
        ValueError,
        match="generation_count does not match",
    ):
        extract_training_examples(payload)


def test_rejects_generated_token_count_mismatch() -> None:
    payload = make_payload()
    first_record(payload)["generated_token_count"] = 99

    with pytest.raises(
        ValueError,
        match="generated_token_count does not match",
    ):
        extract_training_examples(payload)


def test_rejects_missing_rollout_logprobs() -> None:
    payload = make_payload()
    del first_generation(payload)[
        "generated_token_logprobs"
    ]

    with pytest.raises(
        ValueError,
        match="generated_token_logprobs must be a list",
    ):
        extract_training_examples(payload)
