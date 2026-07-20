"""Tests for the controlled-parity matrix CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from parity_posttrain.parity.matrix import (
    ControlledParityMatrixCondition,
)
from parity_posttrain.parity.matrix_runner import (
    ControlledParityMatrixRun,
)


def load_script_module() -> ModuleType:
    """Load the matrix CLI as an importable module."""

    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_parity_matrix.py"
    )

    spec = importlib.util.spec_from_file_location(
        "run_parity_matrix",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not load run_parity_matrix.py"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize(
    (
        "uncached_passed",
        "known_mismatches",
        "expected_overall",
        "expected_regression",
        "expected_failed_slugs",
        "expected_known_failed_slugs",
        "expected_resolved_known_slugs",
        "expected_unexpected_failed_slugs",
        "expected_exit_code",
    ),
    [
        (
            True,
            [],
            True,
            True,
            [],
            [],
            [],
            [],
            0,
        ),
        (
            False,
            [],
            False,
            False,
            ["cpu_no_cache"],
            [],
            [],
            ["cpu_no_cache"],
            1,
        ),
        (
            False,
            ["cpu_no_cache"],
            False,
            True,
            ["cpu_no_cache"],
            ["cpu_no_cache"],
            [],
            [],
            0,
        ),
        (
            True,
            ["cpu_no_cache"],
            True,
            True,
            [],
            [],
            ["cpu_no_cache"],
            [],
            0,
        ),
    ],
)
def test_main_runs_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uncached_passed: bool,
    known_mismatches: list[str],
    expected_overall: bool,
    expected_regression: bool,
    expected_failed_slugs: list[str],
    expected_known_failed_slugs: list[str],
    expected_resolved_known_slugs: list[str],
    expected_unexpected_failed_slugs: list[str],
    expected_exit_code: int,
) -> None:
    module = load_script_module()

    condition_directory = tmp_path / "conditions"
    condition_directory.mkdir()

    cached_path = (
        condition_directory
        / "controlled_cpu_cache.json"
    )
    uncached_path = (
        condition_directory
        / "controlled_cpu_no_cache.json"
    )

    fake_provenance = {
        "seed": 17,
        "requested_model_revision": (
            "revision-test"
        ),
        "resolved_model_revision": (
            "resolved-revision-test"
        ),
    }

    cached_path.write_text(
        json.dumps(
            {
                "condition": "cached",
                "provenance": fake_provenance,
            }
        ),
        encoding="utf-8",
    )
    uncached_path.write_text(
        json.dumps(
            {
                "condition": "uncached",
                "provenance": fake_provenance,
            }
        ),
        encoding="utf-8",
    )

    cached_condition = (
        ControlledParityMatrixCondition(
            device="cpu",
            use_cache=True,
        )
    )
    uncached_condition = (
        ControlledParityMatrixCondition(
            device="cpu",
            use_cache=False,
        )
    )

    fake_runs = (
        ControlledParityMatrixRun(
            condition=cached_condition,
            command=(
                "python-test",
                "run_controlled_parity.py",
                "--use-cache",
            ),
            output_path=cached_path,
        ),
        ControlledParityMatrixRun(
            condition=uncached_condition,
            command=(
                "python-test",
                "run_controlled_parity.py",
                "--no-use-cache",
            ),
            output_path=uncached_path,
        ),
    )

    calls: dict[str, Any] = {}

    def fake_runner(
        **kwargs: Any,
    ) -> tuple[ControlledParityMatrixRun, ...]:
        calls["runner_kwargs"] = kwargs

        return fake_runs

    fake_rows = (
        SimpleNamespace(
            device="cpu",
            dtype="torch.float32",
            use_cache=True,
            latency_ms=10.0,
            mean_absolute_error=1e-6,
            max_absolute_error=2e-6,
            tokens_over_tolerance=0,
            within_tolerance=True,
        ),
        SimpleNamespace(
            device="cpu",
            dtype="torch.float32",
            use_cache=False,
            latency_ms=20.0,
            mean_absolute_error=3e-6,
            max_absolute_error=4e-6,
            tokens_over_tolerance=(
                0 if uncached_passed else 1
            ),
            within_tolerance=uncached_passed,
        ),
    )

    fake_summary = SimpleNamespace(
        rows=fake_rows,
    )

    def fake_builder(
        payloads: list[dict[str, Any]],
    ) -> SimpleNamespace:
        calls["payloads"] = payloads

        return fake_summary

    def fake_serializer(
        summary: SimpleNamespace,
    ) -> dict[str, Any]:
        assert summary is fake_summary

        return {
            "source": {
                "task_id": "basket_001",
            },
            "rows": [],
            "comparisons": [],
        }

    monkeypatch.setattr(
        module,
        "run_controlled_parity_matrix",
        fake_runner,
    )
    monkeypatch.setattr(
        module,
        "build_controlled_parity_summary",
        fake_builder,
    )
    monkeypatch.setattr(
        module,
        "controlled_parity_summary_to_dict",
        fake_serializer,
    )

    output = tmp_path / "summary.json"
    known_mismatch_args = [
        argument
        for slug in known_mismatches
        for argument in (
            "--known-mismatch",
            slug,
        )
    ]

    exit_code = module.main(
        [
            "--artifact",
            "artifacts/agent_benchmark.json",
            "--task-id",
            "basket_001",
            "--turn-index",
            "0",
            "--tolerance",
            "0.001",
            "--seed",
            "17",
            "--model-revision",
            "revision-test",
            "--no-include-mps",
            "--output-directory",
            str(condition_directory),
            *known_mismatch_args,
            "--output",
            str(output),
        ]
    )

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert exit_code == expected_exit_code
    assert payload["schema_version"] == 1
    assert (
        payload["overall_passed"]
        is expected_overall
    )
    assert (
        payload["regression_passed"]
        is expected_regression
    )
    assert payload["failed_condition_slugs"] == (
        expected_failed_slugs
    )
    assert payload["known_mismatch_slugs"] == (
        known_mismatches
    )
    assert payload[
        "known_failed_condition_slugs"
    ] == expected_known_failed_slugs
    assert payload[
        "resolved_known_mismatch_slugs"
    ] == expected_resolved_known_slugs
    assert payload[
        "unexpected_failed_condition_slugs"
    ] == expected_unexpected_failed_slugs
    assert payload["matrix"]["seed"] == 17
    assert (
        payload["matrix"][
            "requested_model_revision"
        ]
        == "revision-test"
    )
    assert (
        payload["matrix"][
            "resolved_model_revision"
        ]
        == "resolved-revision-test"
    )
    assert payload["matrix"]["include_mps"] is False
    assert payload["matrix"]["condition_count"] == 2

    assert [
        condition["slug"]
        for condition in payload["matrix"][
            "conditions"
        ]
    ] == [
        "cpu_cache",
        "cpu_no_cache",
    ]

    runner_kwargs = calls["runner_kwargs"]

    assert runner_kwargs["task_id"] == "basket_001"
    assert runner_kwargs["turn_index"] == 0
    assert runner_kwargs["tolerance"] == 1e-3
    assert runner_kwargs["seed"] == 17
    assert (
        runner_kwargs["model_revision"]
        == "revision-test"
    )
    assert runner_kwargs["include_mps"] is False

    assert calls["payloads"] == [
        {
            "condition": "cached",
            "provenance": fake_provenance,
        },
        {
            "condition": "uncached",
            "provenance": fake_provenance,
        },
    ]


def test_validate_known_mismatch_slugs_deduplicates(
) -> None:
    module = load_script_module()

    assert module.validate_known_mismatch_slugs(
        [
            "mps_cache",
            "mps_cache",
            "cpu_cache",
        ],
        available_slugs=[
            "cpu_cache",
            "mps_cache",
        ],
    ) == [
        "mps_cache",
        "cpu_cache",
    ]


def test_validate_known_mismatch_slugs_rejects_unknown(
) -> None:
    module = load_script_module()

    with pytest.raises(
        ValueError,
        match=(
            "unknown known-mismatch condition "
            "slug.*mps-fp16-cache"
        ),
    ):
        module.validate_known_mismatch_slugs(
            ["mps-fp16-cache"],
            available_slugs=[
                "cpu_cache",
                "cpu_no_cache",
                "mps_cache",
                "mps_no_cache",
            ],
        )


def test_validate_matrix_provenance_rejects_different_runs(
) -> None:
    module = load_script_module()

    with pytest.raises(
        ValueError,
        match="identical provenance",
    ):
        module.validate_matrix_provenance(
            [
                {
                    "provenance": {
                        "seed": 17,
                        "requested_model_revision": (
                            "revision-test"
                        ),
                        "resolved_model_revision": (
                            "resolved-a"
                        ),
                    }
                },
                {
                    "provenance": {
                        "seed": 17,
                        "requested_model_revision": (
                            "revision-test"
                        ),
                        "resolved_model_revision": (
                            "resolved-b"
                        ),
                    }
                },
            ],
            seed=17,
            requested_model_revision=(
                "revision-test"
            ),
        )


def test_validate_matrix_provenance_rejects_seed_mismatch(
) -> None:
    module = load_script_module()

    with pytest.raises(
        ValueError,
        match="seed does not match",
    ):
        module.validate_matrix_provenance(
            [
                {
                    "provenance": {
                        "seed": 0,
                        "requested_model_revision": (
                            "revision-test"
                        ),
                        "resolved_model_revision": (
                            "resolved-revision-test"
                        ),
                    }
                }
            ],
            seed=17,
            requested_model_revision=(
                "revision-test"
            ),
        )


def test_load_payload_rejects_non_object(
    tmp_path: Path,
) -> None:
    module = load_script_module()

    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(["not", "an", "object"]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a JSON object",
    ):
        module.load_controlled_payload(path)
