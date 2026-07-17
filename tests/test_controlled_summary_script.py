"""End-to-end test for the controlled summary CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def make_payload(
    *,
    use_cache: bool,
    latency_ms: float,
    mean_error: float,
    max_error: float,
) -> dict[str, object]:
    """Create one controlled parity artifact."""

    return {
        "condition": {
            "device": "cpu",
            "dtype": "torch.float32",
            "use_cache": use_cache,
        },
        "forced_rollout": {
            "latency_ms": latency_ms,
            "token_logprobs": [-0.1, -0.2],
            "use_cache": use_cache,
        },
        "parity": {
            "device": "cpu",
            "dtype": "torch.float32",
            "max_absolute_error": max_error,
            "mean_absolute_error": mean_error,
            "model_name": "test-model",
            "p95_absolute_error": max_error,
            "token_count": 2,
            "token_records": [],
            "tokens_over_tolerance": (
                1 if max_error > 1e-3 else 0
            ),
            "tolerance": 1e-3,
            "within_tolerance": max_error <= 1e-3,
        },
        "source": {
            "artifact": "source.json",
            "generated_text": "test",
            "generated_token_count": 2,
            "prompt_token_count": 10,
            "task_id": "basket_001",
            "turn_index": 0,
        },
        "trainer_logprobs": [-0.1, -0.2],
    }


def write_payload(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write one test artifact."""

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_summary_script_writes_combined_json(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "summarize_controlled_parity.py"
    )

    cached_path = tmp_path / "cached.json"
    uncached_path = tmp_path / "uncached.json"
    output_path = tmp_path / "summary.json"

    write_payload(
        cached_path,
        make_payload(
            use_cache=True,
            latency_ms=2.0,
            mean_error=0.01,
            max_error=0.04,
        ),
    )
    write_payload(
        uncached_path,
        make_payload(
            use_cache=False,
            latency_ms=10.0,
            mean_error=0.0001,
            max_error=0.0002,
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(cached_path),
            str(uncached_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload["source"]["task_id"] == "basket_001"
    assert len(payload["rows"]) == 2
    assert len(payload["comparisons"]) == 1

    comparison = payload["comparisons"][0]

    assert comparison["no_cache_latency_factor"] == 5.0
    assert comparison["max_error_reduction_factor"] == 200.0
    assert (
        comparison["mean_error_reduction_factor"]
        == 100.0
    )

    assert "Conditions: 2" in completed.stdout
    assert "no_cache_latency_factor=5.000x" in (
        completed.stdout
    )
    assert str(output_path) in completed.stdout
