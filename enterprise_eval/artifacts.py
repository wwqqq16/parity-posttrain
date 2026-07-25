from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from enterprise_eval.models import AgentRun, EvaluationResult


def _json_default(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_run_artifact(
    run: AgentRun,
    evaluation: EvaluationResult,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.case_id}-{run.run_id}.json"
    payload = {
        "schema_version": "enterprise-agent-run.v1",
        "run": asdict(run),
        "evaluation": evaluation.to_dict(),
    }
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path
