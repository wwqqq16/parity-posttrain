"""Trajectory-level parity analysis across multiple agent turns."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from parity_posttrain.parity.logprob_parity import (
    ParityReport,
    build_parity_report,
    exceeds_tolerance,
    rescore_generated_tokens,
)
from parity_posttrain.rollout.hf_backend import (
    GenerationResult,
    synchronize_device,
)


@dataclass(slots=True)
class TurnParityReport:
    """Parity result for one model generation in an agent trajectory."""

    turn_index: int
    prompt_token_count: int
    generated_text: str
    rescore_latency_ms: float
    parity: ParityReport

    def validate(self) -> None:
        """Validate one turn-level parity report."""

        if self.turn_index < 0:
            raise ValueError("turn_index must not be negative")

        if self.prompt_token_count <= 0:
            raise ValueError(
                "prompt_token_count must be positive"
            )

        if self.rescore_latency_ms < 0:
            raise ValueError(
                "rescore_latency_ms must not be negative"
            )

        self.parity.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable turn report."""

        self.validate()
        return asdict(self)


@dataclass(slots=True)
class TrajectoryParityReport:
    """Aggregate parity metrics across all model turns."""

    model_name: str
    device: str
    dtype: str
    generation_count: int
    total_token_count: int
    tolerance: float
    mean_absolute_error: float
    max_absolute_error: float
    p95_absolute_error: float
    tokens_over_tolerance: int
    within_tolerance: bool
    total_rescore_latency_ms: float
    turn_reports: list[TurnParityReport]

    def validate(self) -> None:
        """Validate aggregate trajectory-level metrics."""

        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

        if self.generation_count <= 0:
            raise ValueError(
                "generation_count must be positive"
            )

        if self.generation_count != len(self.turn_reports):
            raise ValueError(
                "generation_count must match turn reports"
            )

        expected_token_count = sum(
            turn.parity.token_count
            for turn in self.turn_reports
        )

        if self.total_token_count != expected_token_count:
            raise ValueError(
                "total_token_count must match turn reports"
            )

        if self.total_token_count <= 0:
            raise ValueError(
                "total_token_count must be positive"
            )

        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if self.mean_absolute_error < 0:
            raise ValueError(
                "mean_absolute_error must not be negative"
            )

        if self.max_absolute_error < 0:
            raise ValueError(
                "max_absolute_error must not be negative"
            )

        if self.total_rescore_latency_ms < 0:
            raise ValueError(
                "total_rescore_latency_ms must not be negative"
            )

        for turn in self.turn_reports:
            turn.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trajectory report."""

        self.validate()
        return asdict(self)


def aggregate_turn_reports(
    turn_reports: list[TurnParityReport],
) -> TrajectoryParityReport:
    """Aggregate token errors across multiple generation turns."""

    if not turn_reports:
        raise ValueError("turn_reports must not be empty")

    for turn in turn_reports:
        turn.validate()

    first = turn_reports[0]
    model_name = first.parity.model_name
    device = first.parity.device
    dtype = first.parity.dtype
    tolerance = first.parity.tolerance

    for turn in turn_reports[1:]:
        parity = turn.parity

        if parity.model_name != model_name:
            raise ValueError(
                "all turns must use the same model"
            )

        if parity.device != device:
            raise ValueError(
                "all turns must use the same device"
            )

        if parity.dtype != dtype:
            raise ValueError(
                "all turns must use the same dtype"
            )

        if parity.tolerance != tolerance:
            raise ValueError(
                "all turns must use the same tolerance"
            )

    errors = [
        token.absolute_error
        for turn in turn_reports
        for token in turn.parity.token_records
    ]

    if not errors:
        raise ValueError(
            "turn reports must contain at least one token"
        )

    sorted_errors = sorted(errors)
    token_count = len(errors)
    p95_index = math.ceil(0.95 * token_count) - 1
    max_error = max(errors)

    report = TrajectoryParityReport(
        model_name=model_name,
        device=device,
        dtype=dtype,
        generation_count=len(turn_reports),
        total_token_count=token_count,
        tolerance=tolerance,
        mean_absolute_error=sum(errors) / token_count,
        max_absolute_error=max_error,
        p95_absolute_error=sorted_errors[p95_index],
        tokens_over_tolerance=sum(
            exceeds_tolerance(error, tolerance)
            for error in errors
        ),
        within_tolerance=not exceeds_tolerance(
            max_error,
            tolerance,
        ),
        total_rescore_latency_ms=round(
            sum(
                turn.rescore_latency_ms
                for turn in turn_reports
            ),
            3,
        ),
        turn_reports=turn_reports,
    )
    report.validate()

    return report


def rescore_agent_generations(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    dtype: str,
    generations: list[GenerationResult],
    tolerance: float = 1e-3,
) -> TrajectoryParityReport:
    """Rescore every model turn and build one trajectory report."""

    if not generations:
        raise ValueError("generations must not be empty")

    turn_reports: list[TurnParityReport] = []

    for turn_index, generation in enumerate(generations):
        generation.validate()

        synchronize_device(device)
        started = time.perf_counter()

        trainer_logprobs = rescore_generated_tokens(
            model=model,
            device=device,
            prompt_token_ids=generation.prompt_token_ids,
            generated_token_ids=(
                generation.generated_token_ids
            ),
        )

        synchronize_device(device)
        rescore_latency_ms = (
            time.perf_counter() - started
        ) * 1000

        token_texts = [
            cast(
                str,
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                ),
            )
            for token_id in generation.generated_token_ids
        ]

        parity = build_parity_report(
            model_name=generation.model_name,
            device=generation.device,
            dtype=dtype,
            token_ids=generation.generated_token_ids,
            token_texts=token_texts,
            rollout_logprobs=(
                generation.generated_token_logprobs
            ),
            trainer_logprobs=trainer_logprobs,
            tolerance=tolerance,
        )

        turn_reports.append(
            TurnParityReport(
                turn_index=turn_index,
                prompt_token_count=len(
                    generation.prompt_token_ids
                ),
                generated_text=(
                    generation.generated_text
                ),
                rescore_latency_ms=round(
                    rescore_latency_ms,
                    3,
                ),
                parity=parity,
            )
        )

    return aggregate_turn_reports(turn_reports)
