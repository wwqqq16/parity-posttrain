from __future__ import annotations

import json
from pathlib import Path

import pytest

from enterprise_eval.models import Architecture
from enterprise_eval.posttrain import (
    build_preference_pairs,
    build_semantic_messages,
    build_sft_records,
    build_step_reward_records,
    export_posttrain_datasets,
    load_run_artifact,
    load_run_artifacts,
)
from enterprise_eval.runner import run_benchmark


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _run_ablation(tmp_path: Path) -> tuple[Path, ...]:
    rows = []
    rows.extend(
        run_benchmark(tmp_path / "runs", architecture=Architecture.SINGLE)
    )
    rows.extend(
        run_benchmark(
            tmp_path / "runs",
            architecture=Architecture.PLANNER_CRITIC,
        )
    )
    return tuple(row.artifact_path for row in rows)


def test_builds_semantic_sft_records_without_fake_tokens(tmp_path: Path) -> None:
    paths = _run_ablation(tmp_path)
    artifacts = load_run_artifacts(paths)
    records = build_sft_records(artifacts)

    assert len(records) == 25
    assert all(record["schema_version"] == "enterprise-sft.v1" for record in records)
    assert all("prompt_token_ids" not in record for record in records)
    assert all("rollout_logprobs" not in record for record in records)
    assert all(record["messages"] for record in records)


def test_semantic_messages_preserve_tool_calls_and_results(tmp_path: Path) -> None:
    paths = _run_ablation(tmp_path)
    artifact = load_run_artifact(
        next(path for path in paths if "eligible_standard" in path.name)
    )
    messages = build_semantic_messages(artifact)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(message.get("tool_call") for message in messages)
    assert any(message["role"] == "tool" for message in messages)
    assert messages[-1]["role"] == "assistant"


def test_builds_three_failure_based_preference_pairs(tmp_path: Path) -> None:
    paths = _run_ablation(tmp_path)
    pairs = build_preference_pairs(load_run_artifacts(paths))

    assert {pair["case_id"] for pair in pairs} == {
        "stale_payment_status",
        "user_withdraws_refund",
        "wrong_order_then_corrected",
    }
    assert all(pair["chosen"]["reward"] > pair["rejected"]["reward"] for pair in pairs)
    assert all(pair["chosen"]["architecture"] == "planner-critic" for pair in pairs)
    assert all(pair["rejected"]["architecture"] == "single" for pair in pairs)


def test_step_rewards_include_policy_penalty_and_recovery_credit(
    tmp_path: Path,
) -> None:
    paths = _run_ablation(tmp_path)
    records = build_step_reward_records(load_run_artifacts(paths))

    stale_single = [
        record
        for record in records
        if record["case_id"] == "stale_payment_status"
        and record["architecture"] == "single"
    ]
    assert any(
        record["reward_components"].get("policy_penalty") == -1.0
        for record in stale_single
    )

    recovered = [
        record
        for record in records
        if record["case_id"] == "payment_timeout_recoverable"
    ]
    assert any(
        record["reward_components"].get("recovery_credit") == 0.25
        for record in recovered
    )


def test_exports_jsonl_and_manifest(tmp_path: Path) -> None:
    paths = _run_ablation(tmp_path)
    summary = export_posttrain_datasets(paths, tmp_path / "posttrain")

    assert summary.source_artifact_count == 28
    assert summary.sft_record_count == 25
    assert summary.preference_pair_count == 3
    assert summary.step_reward_record_count > 0
    assert len(_load_jsonl(summary.sft_path)) == 25
    assert len(_load_jsonl(summary.preferences_path)) == 3
    assert len(_load_jsonl(summary.step_rewards_path)) == summary.step_reward_record_count

    manifest = json.loads(summary.manifest_path.read_text())
    assert manifest["token_level_training_ready"] is False
    assert manifest["preference_pair_count"] == 3


def test_rejects_unsupported_artifact_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "unknown.v1",
                "run": {},
                "evaluation": {},
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_run_artifact(path)
