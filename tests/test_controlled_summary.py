"""Tests for controlled parity summary construction."""

from __future__ import annotations

import pytest

from parity_posttrain.parity.controlled_summary import (
    build_controlled_parity_summary,
    parse_controlled_parity_payload,
)


def make_payload(
    *,
    device: str = "mps",
    dtype: str = "torch.float16",
    use_cache: bool,
    latency_ms: float,
    mean_error: float,
    max_error: float,
    task_id: str = "basket_001",
) -> dict[str, object]:
    """Create one representative controlled result."""

    return {
        "condition": {
            "device": device,
            "dtype": dtype,
            "use_cache": use_cache,
        },
        "forced_rollout": {
            "latency_ms": latency_ms,
            "token_logprobs": [-0.1, -0.2],
            "use_cache": use_cache,
        },
        "parity": {
            "device": device,
            "dtype": dtype,
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
            "artifact": "artifacts/agent_benchmark.json",
            "generated_text": "test",
            "generated_token_count": 2,
            "prompt_token_count": 10,
            "task_id": task_id,
            "turn_index": 0,
        },
        "trainer_logprobs": [-0.1, -0.2],
    }


def test_parse_controlled_parity_payload() -> None:
    payload = make_payload(
        use_cache=True,
        latency_ms=100.0,
        mean_error=0.002,
        max_error=0.03,
    )

    row = parse_controlled_parity_payload(payload)

    assert row.task_id == "basket_001"
    assert row.device == "mps"
    assert row.dtype == "torch.float16"
    assert row.use_cache is True
    assert row.latency_ms == 100.0
    assert row.max_absolute_error == 0.03
    assert row.tokens_over_tolerance == 1
    assert row.within_tolerance is False


def test_build_summary_compares_cache_modes() -> None:
    cached = make_payload(
        use_cache=True,
        latency_ms=2.0,
        mean_error=0.01,
        max_error=0.04,
    )
    uncached = make_payload(
        use_cache=False,
        latency_ms=10.0,
        mean_error=0.0001,
        max_error=0.0002,
    )

    summary = build_controlled_parity_summary(
        [uncached, cached]
    )

    assert len(summary.rows) == 2
    assert len(summary.comparisons) == 1
    assert summary.rows[0].use_cache is True
    assert summary.rows[1].use_cache is False

    comparison = summary.comparisons[0]

    assert comparison.device == "mps"
    assert comparison.dtype == "torch.float16"
    assert comparison.no_cache_latency_factor == pytest.approx(
        5.0
    )
    assert (
        comparison.max_error_reduction_factor
        == pytest.approx(200.0)
    )
    assert (
        comparison.mean_error_reduction_factor
        == pytest.approx(100.0)
    )


def test_summary_rejects_missing_cache_mode() -> None:
    cached = make_payload(
        use_cache=True,
        latency_ms=2.0,
        mean_error=0.01,
        max_error=0.04,
    )

    with pytest.raises(
        ValueError,
        match="both cache modes",
    ):
        build_controlled_parity_summary([cached])


def test_summary_rejects_different_source_sequence() -> None:
    cached = make_payload(
        use_cache=True,
        latency_ms=2.0,
        mean_error=0.01,
        max_error=0.04,
    )
    uncached = make_payload(
        use_cache=False,
        latency_ms=10.0,
        mean_error=0.0001,
        max_error=0.0002,
        task_id="different_task",
    )

    with pytest.raises(
        ValueError,
        match="same source sequence",
    ):
        build_controlled_parity_summary(
            [cached, uncached]
        )


def test_parser_rejects_cache_metadata_mismatch() -> None:
    payload = make_payload(
        use_cache=True,
        latency_ms=2.0,
        mean_error=0.01,
        max_error=0.04,
    )
    forced_rollout = payload["forced_rollout"]

    assert isinstance(forced_rollout, dict)
    forced_rollout["use_cache"] = False

    with pytest.raises(
        ValueError,
        match="disagree about use_cache",
    ):
        parse_controlled_parity_payload(payload)


def test_summary_serializes_to_json_safe_dict() -> None:
    from parity_posttrain.parity.controlled_summary import (
        controlled_parity_summary_to_dict,
    )

    cached = make_payload(
        use_cache=True,
        latency_ms=2.0,
        mean_error=0.01,
        max_error=0.04,
    )
    uncached = make_payload(
        use_cache=False,
        latency_ms=10.0,
        mean_error=0.0001,
        max_error=0.0002,
    )

    summary = build_controlled_parity_summary(
        [cached, uncached]
    )
    payload = controlled_parity_summary_to_dict(summary)

    assert payload["source"] == {
        "task_id": "basket_001",
        "turn_index": 0,
        "model_name": "test-model",
        "prompt_token_count": 10,
        "generated_token_count": 2,
        "tolerance": 1e-3,
    }

    rows = payload["rows"]
    comparisons = payload["comparisons"]

    assert isinstance(rows, list)
    assert isinstance(comparisons, list)
    assert len(rows) == 2
    assert len(comparisons) == 1

    comparison = comparisons[0]

    assert isinstance(comparison, dict)
    assert comparison["no_cache_latency_factor"] == 5.0
    assert comparison["max_error_reduction_factor"] == 200.0
    assert comparison["mean_error_reduction_factor"] == 100.0
