"""Tests for controlled-parity matrix planning."""

from pathlib import Path

import pytest

from parity_posttrain.parity.matrix import (
    ControlledParityMatrixCondition,
    build_controlled_parity_matrix,
)


def test_default_matrix_contains_four_conditions(
) -> None:
    conditions = build_controlled_parity_matrix()

    assert [
        (
            condition.device,
            condition.use_cache,
        )
        for condition in conditions
    ] == [
        ("cpu", True),
        ("cpu", False),
        ("mps", True),
        ("mps", False),
    ]

    assert [
        condition.slug
        for condition in conditions
    ] == [
        "cpu_cache",
        "cpu_no_cache",
        "mps_cache",
        "mps_no_cache",
    ]


def test_matrix_can_skip_mps() -> None:
    conditions = build_controlled_parity_matrix(
        include_mps=False
    )

    assert [
        condition.slug
        for condition in conditions
    ] == [
        "cpu_cache",
        "cpu_no_cache",
    ]


def test_cached_condition_builds_cli_arguments(
) -> None:
    condition = ControlledParityMatrixCondition(
        device="cpu",
        use_cache=True,
    )

    arguments = condition.cli_arguments(
        artifact=Path(
            "artifacts/agent_benchmark.json"
        ),
        task_id="basket_001",
        turn_index=0,
        tolerance=1e-3,
        output_directory=Path(
            "artifacts/parity_matrix"
        ),
    )

    assert arguments == [
        "--artifact",
        "artifacts/agent_benchmark.json",
        "--task-id",
        "basket_001",
        "--turn-index",
        "0",
        "--device",
        "cpu",
        "--use-cache",
        "--tolerance",
        "0.001",
        "--output",
        (
            "artifacts/parity_matrix/"
            "controlled_cpu_cache.json"
        ),
    ]


def test_uncached_condition_uses_negative_flag(
) -> None:
    condition = ControlledParityMatrixCondition(
        device="mps",
        use_cache=False,
    )

    arguments = condition.cli_arguments(
        artifact=Path("input.json"),
        task_id="basket_001",
        turn_index=1,
        tolerance=0.01,
        output_directory=Path("results"),
    )

    assert "--no-use-cache" in arguments
    assert "--use-cache" not in arguments

    assert arguments[-1] == (
        "results/controlled_mps_no_cache.json"
    )


def test_condition_rejects_invalid_inputs() -> None:
    condition = ControlledParityMatrixCondition(
        device="cpu",
        use_cache=True,
    )

    with pytest.raises(
        ValueError,
        match="task_id must not be empty",
    ):
        condition.cli_arguments(
            artifact=Path("input.json"),
            task_id=" ",
            turn_index=0,
            tolerance=1e-3,
            output_directory=Path("results"),
        )

    with pytest.raises(
        ValueError,
        match="turn_index must be non-negative",
    ):
        condition.cli_arguments(
            artifact=Path("input.json"),
            task_id="basket_001",
            turn_index=-1,
            tolerance=1e-3,
            output_directory=Path("results"),
        )

    with pytest.raises(
        ValueError,
        match="tolerance must be positive",
    ):
        condition.cli_arguments(
            artifact=Path("input.json"),
            task_id="basket_001",
            turn_index=0,
            tolerance=0.0,
            output_directory=Path("results"),
        )
