"""Tests for token-level enterprise model artifact conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_eval.model_posttrain import (
    build_enterprise_training_examples,
    build_enterprise_training_examples_from_artifact,
    get_model_generation,
    get_model_generations,
    load_enterprise_model_artifact,
)


def make_payload(*, success: bool = True) -> dict[str, object]:
    return {
        "schema_version": "enterprise-agent-run.v1",
        "run": {
            "run_id": "run-1",
            "case_id": "eligible_standard",
            "metadata": {
                "model_generations": [
                    {
                        "turn_index": 0,
                        "model_name": "test-model",
                        "prompt_token_ids": [1, 2],
                        "generated_token_ids": [3, 4],
                        "generated_token_logprobs": [-0.1, -0.2],
                        "generated_text": '{"action":"get_order"}',
                    },
                    {
                        "turn_index": 1,
                        "model_name": "test-model",
                        "prompt_token_ids": [1, 2, 3, 4],
                        "generated_token_ids": [5],
                        "generated_token_logprobs": [-0.3],
                        "generated_text": '{"action":"respond"}',
                    },
                ]
            },
        },
        "evaluation": {
            "task_success": success,
            "failure_type": None if success else "policy_violation_attempt",
            "final_reward": 1.0 if success else -0.5,
        },
    }


def test_builds_one_training_example_per_generation() -> None:
    examples = build_enterprise_training_examples(make_payload())

    assert len(examples) == 2
    assert examples[0].task_id == "eligible_standard:run-1"
    assert examples[0].turn_index == 0
    assert examples[0].status == "completed"
    assert examples[0].reward == 1.0
    assert examples[0].prompt_token_ids == (1, 2)
    assert examples[0].generated_token_ids == (3, 4)
    assert examples[0].rollout_logprobs == (-0.1, -0.2)
    assert examples[1].turn_index == 1


def test_failed_run_uses_failure_type_and_reward() -> None:
    examples = build_enterprise_training_examples(make_payload(success=False))

    assert all(example.status == "policy_violation_attempt" for example in examples)
    assert all(example.reward == -0.5 for example in examples)


def test_selects_generation_by_turn() -> None:
    generation = get_model_generation(make_payload(), turn_index=1)
    assert generation["generated_token_ids"] == [5]


def test_rejects_out_of_range_turn() -> None:
    with pytest.raises(ValueError, match="out of range"):
        get_model_generation(make_payload(), turn_index=2)


def test_rejects_missing_generations() -> None:
    payload = make_payload()
    payload["run"]["metadata"]["model_generations"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="no model_generations"):
        get_model_generations(payload)  # type: ignore[arg-type]


def test_rejects_token_logprob_length_mismatch() -> None:
    payload = make_payload()
    generations = payload["run"]["metadata"]["model_generations"]  # type: ignore[index]
    generations[0]["generated_token_logprobs"] = [-0.1]  # type: ignore[index]

    with pytest.raises(ValueError, match="equal length"):
        build_enterprise_training_examples(payload)  # type: ignore[arg-type]


def test_loads_artifact_and_builds_examples(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(make_payload()), encoding="utf-8")

    payload = load_enterprise_model_artifact(path)
    examples = build_enterprise_training_examples_from_artifact(path)

    assert payload["run"]["case_id"] == "eligible_standard"  # type: ignore[index]
    assert len(examples) == 2


def test_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = make_payload()
    payload["schema_version"] = "unknown"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_enterprise_model_artifact(path)
