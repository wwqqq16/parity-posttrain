from __future__ import annotations

from dataclasses import replace

import pytest

from enterprise_eval.models import (
    ActionType,
    Difficulty,
    ExpectedOutcome,
    FailureProfile,
)
from enterprise_eval.task_factory import (
    TaskFactoryConfig,
    generate_failure_surface_cases,
    validate_task_suite,
)


def test_factory_builds_balanced_deterministic_suite() -> None:
    config = TaskFactoryConfig(
        seed=17,
        variants_per_cell=5,
    )
    first = generate_failure_surface_cases(config)
    second = generate_failure_surface_cases(config)

    assert first == second
    assert len(first) == 105
    assert len({case.case_id for case in first}) == 105
    assert len({case.order_id for case in first}) == 105

    validation = validate_task_suite(
        first,
        config=config,
    )
    assert validation.case_count == 105
    assert validation.coverage_cell_count == 21
    assert validation.cases_per_cell == 5
    assert validation.oracle_success_rate == 1.0


def test_factory_controls_difficulty_and_injection_position() -> None:
    cases = generate_failure_surface_cases(
        TaskFactoryConfig(variants_per_cell=1)
    )
    hard_payment_timeout = next(
        case
        for case in cases
        if case.difficulty is Difficulty.HARD
        and case.failure_profile
        is FailureProfile.TRANSIENT_TOOL_TIMEOUT
        and case.failure_injection_step == 2
    )

    assert len(hard_payment_timeout.user_messages) == 3
    assert (
        hard_payment_timeout.failure_injection_action
        is ActionType.GET_PAYMENT_STATUS
    )
    assert hard_payment_timeout.failure_injection_count == 1
    assert (
        hard_payment_timeout.expected_outcome
        is ExpectedOutcome.REFUND
    )


def test_validator_rejects_an_inconsistent_closed_outcome() -> None:
    config = TaskFactoryConfig(variants_per_cell=1)
    cases = list(generate_failure_surface_cases(config))
    transient_index = next(
        index
        for index, case in enumerate(cases)
        if case.failure_profile
        is FailureProfile.TRANSIENT_TOOL_TIMEOUT
    )
    cases[transient_index] = replace(
        cases[transient_index],
        expected_outcome=ExpectedOutcome.ESCALATE,
    )

    with pytest.raises(
        ValueError,
        match="expected outcome",
    ):
        validate_task_suite(cases, config=config)


def test_validator_rejects_a_missing_coverage_cell() -> None:
    config = TaskFactoryConfig(variants_per_cell=1)
    cases = list(generate_failure_surface_cases(config))
    cases.pop()

    with pytest.raises(
        ValueError,
        match="coverage grid",
    ):
        validate_task_suite(cases, config=config)
