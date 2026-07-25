from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ENTERPRISE_RUN_SCHEMA = "enterprise-agent-run.v1"
SFT_SCHEMA = "enterprise-sft.v1"
PREFERENCE_SCHEMA = "enterprise-preference.v1"
STEP_REWARD_SCHEMA = "enterprise-step-reward.v1"
MANIFEST_SCHEMA = "enterprise-posttrain-manifest.v1"

SYSTEM_PROMPT = (
    "Resolve the latest enterprise workflow request safely. Use business tools "
    "before taking irreversible actions, follow policy constraints, and request "
    "human review when the evidence is stale, conflicting, or high risk."
)


@dataclass(frozen=True)
class LoadedRunArtifact:
    path: Path
    run: Mapping[str, object]
    evaluation: Mapping[str, object]

    @property
    def run_id(self) -> str:
        return _require_str(self.run, "run_id", field="run")

    @property
    def case_id(self) -> str:
        return _require_str(self.run, "case_id", field="run")

    @property
    def architecture(self) -> str:
        return _require_str(self.run, "architecture", field="run")

    @property
    def task_success(self) -> bool:
        return _require_bool(self.evaluation, "task_success", field="evaluation")

    @property
    def policy_violation(self) -> bool:
        return _require_bool(self.evaluation, "policy_violation", field="evaluation")

    @property
    def final_reward(self) -> float:
        return _require_float(self.evaluation, "final_reward", field="evaluation")


@dataclass(frozen=True)
class ExportSummary:
    source_artifact_count: int
    sft_record_count: int
    preference_pair_count: int
    step_reward_record_count: int
    sft_path: Path
    preferences_path: Path
    step_rewards_path: Path
    manifest_path: Path


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must contain only string keys")
    return cast(Mapping[str, object], value)


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return cast(list[object], value)


def _require_str(mapping: Mapping[str, object], key: str, *, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}.{key} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _require_bool(mapping: Mapping[str, object], key: str, *, field: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{field}.{key} must be a boolean")
    return value


def _require_float(mapping: Mapping[str, object], key: str, *, field: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}.{key} must be numeric")
    return float(value)


def load_run_artifact(path: Path) -> LoadedRunArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw, field=str(path))
    schema_version = _require_str(payload, "schema_version", field=str(path))
    if schema_version != ENTERPRISE_RUN_SCHEMA:
        raise ValueError(
            f"{path} has unsupported schema_version={schema_version!r}"
        )
    run = _require_mapping(payload.get("run"), field=f"{path}.run")
    evaluation = _require_mapping(
        payload.get("evaluation"), field=f"{path}.evaluation"
    )
    artifact = LoadedRunArtifact(path=path, run=run, evaluation=evaluation)
    if artifact.case_id != _require_str(
        evaluation, "case_id", field=f"{path}.evaluation"
    ):
        raise ValueError(f"{path} run/evaluation case_id mismatch")
    if artifact.run_id != _require_str(
        evaluation, "run_id", field=f"{path}.evaluation"
    ):
        raise ValueError(f"{path} run/evaluation run_id mismatch")
    return artifact


def load_run_artifacts(paths: Iterable[Path]) -> tuple[LoadedRunArtifact, ...]:
    loaded = [load_run_artifact(path) for path in sorted(set(paths))]
    if not loaded:
        raise ValueError("at least one enterprise run artifact is required")
    return tuple(loaded)


def build_semantic_messages(artifact: LoadedRunArtifact) -> list[dict[str, Any]]:
    initial_observation = _require_str(
        artifact.run, "initial_observation", field="run"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_observation},
    ]

    steps = _require_list(artifact.run.get("steps"), field="run.steps")
    responded = False
    for step_index, step_value in enumerate(steps):
        step = _require_mapping(step_value, field=f"run.steps[{step_index}]")
        action = _require_mapping(
            step.get("action"), field=f"run.steps[{step_index}].action"
        )
        result = _require_mapping(
            step.get("result"), field=f"run.steps[{step_index}].result"
        )
        action_type = _require_str(
            action, "action_type", field=f"run.steps[{step_index}].action"
        )
        arguments = _require_mapping(
            action.get("arguments"),
            field=f"run.steps[{step_index}].action.arguments",
        )
        observation = _require_str(
            result, "observation", field=f"run.steps[{step_index}].result"
        )
        success = _require_bool(
            result, "success", field=f"run.steps[{step_index}].result"
        )
        metadata = _require_mapping(
            result.get("metadata"), field=f"run.steps[{step_index}].result.metadata"
        )

        if action_type == "respond":
            message = arguments.get("message")
            if not isinstance(message, str) or not message:
                message = observation
            messages.append({"role": "assistant", "content": message})
            responded = True
            continue

        tool_call_id = f"step-{step_index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_call": {
                    "id": tool_call_id,
                    "name": action_type,
                    "arguments": dict(arguments),
                },
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": action_type,
                "content": observation,
                "success": success,
                "metadata": dict(metadata),
            }
        )

    final_message = _optional_str(artifact.run, "final_message")
    if final_message and not responded:
        messages.append({"role": "assistant", "content": final_message})
    return messages


def build_sft_record(artifact: LoadedRunArtifact) -> dict[str, Any]:
    if not artifact.task_success or artifact.policy_violation:
        raise ValueError("SFT records require successful, policy-compliant runs")
    metadata = _require_mapping(artifact.run.get("metadata"), field="run.metadata")
    return {
        "schema_version": SFT_SCHEMA,
        "example_id": f"{artifact.case_id}:{artifact.run_id}",
        "source_artifact": str(artifact.path),
        "source_run_id": artifact.run_id,
        "case_id": artifact.case_id,
        "architecture": artifact.architecture,
        "difficulty": metadata.get("difficulty"),
        "risk_level": metadata.get("risk_level"),
        "task_type": metadata.get("task_type"),
        "reward": artifact.final_reward,
        "messages": build_semantic_messages(artifact),
    }


def build_sft_records(
    artifacts: Sequence[LoadedRunArtifact],
) -> list[dict[str, Any]]:
    return [
        build_sft_record(artifact)
        for artifact in artifacts
        if artifact.task_success and not artifact.policy_violation
    ]


def _preference_key(artifact: LoadedRunArtifact) -> tuple[int, int, float, str]:
    return (
        int(artifact.task_success),
        int(not artifact.policy_violation),
        artifact.final_reward,
        artifact.run_id,
    )


def _pair_id(case_id: str, chosen_run_id: str, rejected_run_id: str) -> str:
    value = f"{case_id}\0{chosen_run_id}\0{rejected_run_id}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def build_preference_pairs(
    artifacts: Sequence[LoadedRunArtifact],
) -> list[dict[str, Any]]:
    groups: dict[str, list[LoadedRunArtifact]] = defaultdict(list)
    for artifact in artifacts:
        groups[artifact.case_id].append(artifact)

    pairs: list[dict[str, Any]] = []
    for case_id in sorted(groups):
        candidates = groups[case_id]
        chosen_candidates = [
            artifact
            for artifact in candidates
            if artifact.task_success and not artifact.policy_violation
        ]
        rejected_candidates = [
            artifact
            for artifact in candidates
            if not artifact.task_success or artifact.policy_violation
        ]
        if not chosen_candidates or not rejected_candidates:
            continue

        chosen = max(chosen_candidates, key=_preference_key)
        rejected = min(rejected_candidates, key=_preference_key)
        failure_type = _optional_str(rejected.evaluation, "failure_type")
        rejection_reason = failure_type or (
            "policy_violation" if rejected.policy_violation else "task_failure"
        )
        pairs.append(
            {
                "schema_version": PREFERENCE_SCHEMA,
                "pair_id": _pair_id(case_id, chosen.run_id, rejected.run_id),
                "case_id": case_id,
                "prompt": _require_str(
                    chosen.run, "initial_observation", field="chosen.run"
                ),
                "chosen": {
                    "run_id": chosen.run_id,
                    "architecture": chosen.architecture,
                    "reward": chosen.final_reward,
                    "messages": build_semantic_messages(chosen),
                },
                "rejected": {
                    "run_id": rejected.run_id,
                    "architecture": rejected.architecture,
                    "reward": rejected.final_reward,
                    "failure_type": failure_type,
                    "messages": build_semantic_messages(rejected),
                },
                "preference_margin": chosen.final_reward - rejected.final_reward,
                "rejection_reason": rejection_reason,
            }
        )
    return pairs


def _step_reward_components(
    *,
    action_type: str,
    result_success: bool,
    result_metadata: Mapping[str, object],
    previous_error_type: str | None,
    expected_outcome: str,
    final_reward: float,
    is_terminal_step: bool,
) -> dict[str, float]:
    components: dict[str, float] = {}
    error_type = result_metadata.get("error_type")
    if error_type == "policy_violation_attempt":
        components["policy_penalty"] = -1.0
    if error_type == "invalid_tool_call":
        components["invalid_tool_penalty"] = -0.25
    if (
        previous_error_type == "transient_tool_failure"
        and result_success
        and action_type == "get_payment_status"
    ):
        components["recovery_credit"] = 0.25
    if (
        action_type == "request_human_review"
        and result_success
        and expected_outcome == "escalate"
    ):
        components["correct_escalation_credit"] = 0.25
    if (
        action_type == "issue_refund"
        and result_success
        and expected_outcome == "refund"
    ):
        components["correct_action_credit"] = 0.25
    if is_terminal_step:
        components["terminal_outcome_reward"] = final_reward
    return components


def build_step_reward_records(
    artifacts: Sequence[LoadedRunArtifact],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        steps = _require_list(artifact.run.get("steps"), field="run.steps")
        expected_outcome = _require_str(
            artifact.evaluation, "expected_outcome", field="evaluation"
        )
        previous_error_type: str | None = None
        for position, step_value in enumerate(steps):
            step = _require_mapping(step_value, field=f"run.steps[{position}]")
            step_index_value = step.get("step_index")
            if isinstance(step_index_value, bool) or not isinstance(
                step_index_value, int
            ):
                raise ValueError(f"run.steps[{position}].step_index must be an integer")
            action = _require_mapping(
                step.get("action"), field=f"run.steps[{position}].action"
            )
            result = _require_mapping(
                step.get("result"), field=f"run.steps[{position}].result"
            )
            action_type = _require_str(
                action, "action_type", field=f"run.steps[{position}].action"
            )
            arguments = _require_mapping(
                action.get("arguments"),
                field=f"run.steps[{position}].action.arguments",
            )
            result_success = _require_bool(
                result, "success", field=f"run.steps[{position}].result"
            )
            observation = _require_str(
                result, "observation", field=f"run.steps[{position}].result"
            )
            result_metadata = _require_mapping(
                result.get("metadata"),
                field=f"run.steps[{position}].result.metadata",
            )
            is_terminal_step = position == len(steps) - 1
            components = _step_reward_components(
                action_type=action_type,
                result_success=result_success,
                result_metadata=result_metadata,
                previous_error_type=previous_error_type,
                expected_outcome=expected_outcome,
                final_reward=artifact.final_reward,
                is_terminal_step=is_terminal_step,
            )
            records.append(
                {
                    "schema_version": STEP_REWARD_SCHEMA,
                    "record_id": f"{artifact.run_id}:{step_index_value}",
                    "source_artifact": str(artifact.path),
                    "run_id": artifact.run_id,
                    "case_id": artifact.case_id,
                    "architecture": artifact.architecture,
                    "step_index": step_index_value,
                    "action_type": action_type,
                    "action_arguments": dict(arguments),
                    "tool_success": result_success,
                    "observation": observation,
                    "tool_metadata": dict(result_metadata),
                    "reward_components": components,
                    "step_reward": sum(components.values()),
                    "trajectory_final_reward": artifact.final_reward,
                    "failure_step": artifact.evaluation.get("failure_step"),
                    "failure_type": artifact.evaluation.get("failure_type"),
                }
            )
            error_type = result_metadata.get("error_type")
            previous_error_type = error_type if isinstance(error_type, str) else None
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def export_posttrain_datasets(
    artifact_paths: Iterable[Path],
    output_dir: Path,
) -> ExportSummary:
    artifacts = load_run_artifacts(artifact_paths)
    sft_records = build_sft_records(artifacts)
    preference_pairs = build_preference_pairs(artifacts)
    step_reward_records = build_step_reward_records(artifacts)

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / "sft.jsonl"
    preferences_path = output_dir / "preferences.jsonl"
    step_rewards_path = output_dir / "step_rewards.jsonl"
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(sft_path, sft_records)
    _write_jsonl(preferences_path, preference_pairs)
    _write_jsonl(step_rewards_path, step_reward_records)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_artifact_count": len(artifacts),
        "sft_record_count": len(sft_records),
        "preference_pair_count": len(preference_pairs),
        "step_reward_record_count": len(step_reward_records),
        "outputs": {
            "sft": str(sft_path),
            "preferences": str(preferences_path),
            "step_rewards": str(step_rewards_path),
        },
        "token_level_training_ready": False,
        "note": (
            "These are semantic post-training records. Token IDs and rollout "
            "log-probabilities require a real model-backed agent adapter."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ExportSummary(
        source_artifact_count=len(artifacts),
        sft_record_count=len(sft_records),
        preference_pair_count=len(preference_pairs),
        step_reward_record_count=len(step_reward_records),
        sft_path=sft_path,
        preferences_path=preferences_path,
        step_rewards_path=step_rewards_path,
        manifest_path=manifest_path,
    )
