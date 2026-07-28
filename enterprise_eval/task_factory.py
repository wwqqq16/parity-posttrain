"""Generate and validate controlled enterprise failure-surface tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random

from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.evaluator import RefundEvaluator
from enterprise_eval.models import (
    ActionType,
    Difficulty,
    ExpectedOutcome,
    FailureProfile,
    PaymentStatus,
    RefundCase,
    RiskLevel,
    TaskType,
)
from enterprise_eval.scripted_agent import SolvabilityOracleAgent

_INJECTION_ACTIONS = {
    0: ActionType.GET_ORDER,
    1: ActionType.CHECK_REFUND_POLICY,
    2: ActionType.GET_PAYMENT_STATUS,
}


@dataclass(frozen=True)
class TaskFactoryConfig:
    """Configuration for a balanced deterministic task suite."""

    seed: int = 17
    variants_per_cell: int = 5
    difficulties: tuple[Difficulty, ...] = (
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    )
    failure_profiles: tuple[FailureProfile, ...] = (
        FailureProfile.NONE,
        FailureProfile.TRANSIENT_TOOL_TIMEOUT,
        FailureProfile.PERSISTENT_TOOL_TIMEOUT,
    )
    injection_steps: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class TaskSuiteValidation:
    """Validation result for a generated task suite."""

    case_count: int
    coverage_cell_count: int
    cases_per_cell: int
    oracle_success_rate: float


def generate_failure_surface_cases(
    config: TaskFactoryConfig | None = None,
) -> tuple[RefundCase, ...]:
    """Build a balanced suite across difficulty, failure type, and position."""

    config = config or TaskFactoryConfig()
    if config.variants_per_cell <= 0:
        raise ValueError("variants_per_cell must be positive")
    if not config.difficulties:
        raise ValueError("at least one difficulty is required")
    if not config.failure_profiles:
        raise ValueError("at least one failure profile is required")
    if any(step not in _INJECTION_ACTIONS for step in config.injection_steps):
        raise ValueError("injection_steps must be selected from 0, 1, and 2")

    random = Random(config.seed)
    cases: list[RefundCase] = []
    sequence = 0

    for difficulty in config.difficulties:
        for failure_profile in config.failure_profiles:
            steps: tuple[int | None, ...] = (
                (None,)
                if failure_profile is FailureProfile.NONE
                else config.injection_steps
            )
            for injection_step in steps:
                for variant in range(config.variants_per_cell):
                    sequence += 1
                    cases.append(
                        _build_case(
                            random=random,
                            sequence=sequence,
                            difficulty=difficulty,
                            failure_profile=failure_profile,
                            injection_step=injection_step,
                            seed=config.seed,
                            variant=variant,
                        )
                    )

    return tuple(cases)


def validate_task_suite(
    cases: tuple[RefundCase, ...] | list[RefundCase],
    *,
    config: TaskFactoryConfig | None = None,
    expected_cases_per_cell: int | None = None,
) -> TaskSuiteValidation:
    """Validate coverage, closed outcomes, and oracle solvability."""

    if not cases:
        raise ValueError("generated task suite must not be empty")

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("generated case IDs must be unique")

    order_ids = [case.order_id for case in cases]
    if len(order_ids) != len(set(order_ids)):
        raise ValueError("generated order IDs must be unique")

    coverage = Counter(
        (
            case.difficulty,
            case.failure_profile,
            case.failure_injection_step,
        )
        for case in cases
    )
    if config is not None:
        expected_coverage = _expected_coverage(config)
        if set(coverage) != expected_coverage:
            raise ValueError(
                "generated task suite does not match "
                "the configured coverage grid"
            )
        expected_cases_per_cell = config.variants_per_cell

    if expected_cases_per_cell is not None and any(
        count != expected_cases_per_cell
        for count in coverage.values()
    ):
        raise ValueError(
            "generated task coverage cells must contain "
            f"{expected_cases_per_cell} cases"
        )

    for case in cases:
        _validate_case_contract(case)

    oracle_successes = 0
    for case in cases:
        env = RefundEnvironment(case)
        SolvabilityOracleAgent().run(env)
        if RefundEvaluator().evaluate(env).task_success:
            oracle_successes += 1

    if oracle_successes != len(cases):
        raise ValueError(
            "generated task suite contains a case "
            "that the solvability oracle cannot complete"
        )

    distinct_cell_sizes = set(coverage.values())
    cases_per_cell = (
        next(iter(distinct_cell_sizes))
        if len(distinct_cell_sizes) == 1
        else 0
    )
    return TaskSuiteValidation(
        case_count=len(cases),
        coverage_cell_count=len(coverage),
        cases_per_cell=cases_per_cell,
        oracle_success_rate=oracle_successes / len(cases),
    )


def _expected_coverage(
    config: TaskFactoryConfig,
) -> set[
    tuple[
        Difficulty,
        FailureProfile,
        int | None,
    ]
]:
    return {
        (
            difficulty,
            profile,
            injection_step,
        )
        for difficulty in config.difficulties
        for profile in config.failure_profiles
        for injection_step in (
            (None,)
            if profile is FailureProfile.NONE
            else config.injection_steps
        )
    }


def _build_case(
    *,
    random: Random,
    sequence: int,
    difficulty: Difficulty,
    failure_profile: FailureProfile,
    injection_step: int | None,
    seed: int,
    variant: int,
) -> RefundCase:
    order_id = f"ORD-{200_000 + abs(seed % 1_000) * 1_000 + sequence}"
    amount_cents = random.randrange(3_000, 25_000, 100)
    delivered_days_ago = random.randint(1, 20)
    user_messages, initial_order_id, corrected_order_id = _messages(
        order_id,
        difficulty=difficulty,
        sequence=sequence,
    )

    injection_count = {
        FailureProfile.NONE: 0,
        FailureProfile.TRANSIENT_TOOL_TIMEOUT: 1,
        FailureProfile.PERSISTENT_TOOL_TIMEOUT: 2,
    }[failure_profile]
    expected_outcome = (
        ExpectedOutcome.ESCALATE
        if failure_profile is FailureProfile.PERSISTENT_TOOL_TIMEOUT
        else ExpectedOutcome.REFUND
    )
    step_slug = "none" if injection_step is None else str(injection_step)
    case_id = (
        f"factory-s{seed}-{difficulty.value}-"
        f"{failure_profile.value}-p{step_slug}-v{variant}"
    )

    return RefundCase(
        case_id=case_id,
        order_id=order_id,
        user_messages=user_messages,
        delivered_days_ago=delivered_days_ago,
        amount_cents=amount_cents,
        payment_status=PaymentStatus.SETTLED,
        customer_claim="Generated eligible refund request.",
        task_type=TaskType.REFUND_REQUEST,
        expected_outcome=expected_outcome,
        difficulty=difficulty,
        risk_level={
            Difficulty.EASY: RiskLevel.LOW,
            Difficulty.MEDIUM: RiskLevel.MEDIUM,
            Difficulty.HARD: RiskLevel.HIGH,
        }[difficulty],
        initial_order_id=initial_order_id,
        corrected_order_id=corrected_order_id,
        injected_failures=(
            ()
            if failure_profile is FailureProfile.NONE
            else (failure_profile.value,)
        ),
        failure_profile=failure_profile,
        failure_injection_step=injection_step,
        failure_injection_action=(
            _INJECTION_ACTIONS[injection_step]
            if injection_step is not None
            else None
        ),
        failure_injection_count=injection_count,
        factory_seed=seed,
        factory_variant=variant,
    )


def _messages(
    order_id: str,
    *,
    difficulty: Difficulty,
    sequence: int,
) -> tuple[tuple[str, ...], str | None, str | None]:
    if difficulty is Difficulty.EASY:
        return (
            (f"Please refund my recent purchase, order {order_id}.",),
            None,
            None,
        )
    if difficulty is Difficulty.MEDIUM:
        return (
            (
                "I need help resolving a recent purchase.",
                f"Please refund the purchase under order {order_id}.",
            ),
            None,
            None,
        )

    wrong_order_id = f"ORD-{900_000 + sequence}"
    return (
        (
            f"Please refund order {wrong_order_id}.",
            f"Correction: the actual order number is {order_id}.",
            f"Please use {order_id} for the refund request.",
        ),
        wrong_order_id,
        order_id,
    )


def _validate_case_contract(case: RefundCase) -> None:
    profile = case.failure_profile
    if profile is FailureProfile.NONE:
        if (
            case.failure_injection_step is not None
            or case.failure_injection_action is not None
            or case.failure_injection_count != 0
        ):
            raise ValueError("no-failure cases must not schedule an injection")
        if case.expected_outcome is not ExpectedOutcome.REFUND:
            raise ValueError("no-failure cases must expect a refund")
        return

    injection_step = case.failure_injection_step
    if injection_step not in _INJECTION_ACTIONS:
        raise ValueError("failure cases must use injection step 0, 1, or 2")
    if case.failure_injection_action is not _INJECTION_ACTIONS[injection_step]:
        raise ValueError("failure injection action does not match its step")

    expected_count = (
        1
        if profile is FailureProfile.TRANSIENT_TOOL_TIMEOUT
        else 2
    )
    if case.failure_injection_count != expected_count:
        raise ValueError("failure injection count does not match its profile")

    expected_outcome = (
        ExpectedOutcome.REFUND
        if profile is FailureProfile.TRANSIENT_TOOL_TIMEOUT
        else ExpectedOutcome.ESCALATE
    )
    if case.expected_outcome is not expected_outcome:
        raise ValueError("expected outcome does not match its failure profile")
