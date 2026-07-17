"""Suite-level aggregation for agent rollout benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from parity_posttrain.agent.runner import AgentRunStatus


@dataclass(slots=True)
class AgentBenchmarkRecord:
    """Metrics from one benchmark task."""

    task_id: str
    category: str
    status: AgentRunStatus
    answer_correct: bool
    reward: float
    tool_coverage: float
    used_tools: list[str]
    missing_tools: list[str]
    unexpected_tools: list[str]
    latency_ms: float
    generation_count: int
    generated_token_count: int
    parity_within_tolerance: bool
    parity_mean_absolute_error: float
    parity_max_absolute_error: float
    parity_tokens_over_tolerance: int
    error: str | None = None

    def validate(self) -> None:
        """Validate one benchmark record."""

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if not self.category.strip():
            raise ValueError("category must not be empty")

        if not 0.0 <= self.reward <= 1.0:
            raise ValueError("reward must be between zero and one")

        if not 0.0 <= self.tool_coverage <= 1.0:
            raise ValueError(
                "tool_coverage must be between zero and one"
            )

        if self.latency_ms < 0:
            raise ValueError("latency_ms must not be negative")

        if self.generation_count <= 0:
            raise ValueError(
                "generation_count must be positive"
            )

        if self.generated_token_count <= 0:
            raise ValueError(
                "generated_token_count must be positive"
            )

        if self.parity_mean_absolute_error < 0:
            raise ValueError(
                "parity mean error must not be negative"
            )

        if self.parity_max_absolute_error < 0:
            raise ValueError(
                "parity max error must not be negative"
            )

        if self.parity_tokens_over_tolerance < 0:
            raise ValueError(
                "parity violation count must not be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        self.validate()
        return asdict(self)


@dataclass(slots=True)
class CategorySummary:
    """Aggregate metrics for one task category."""

    task_count: int
    completed_count: int
    answer_correct_count: int
    completion_rate: float
    answer_accuracy: float
    mean_reward: float
    mean_tool_coverage: float
    mean_latency_ms: float
    parity_pass_rate: float


@dataclass(slots=True)
class AgentBenchmarkSummary:
    """Aggregate metrics for a benchmark suite."""

    task_count: int
    completed_count: int
    protocol_error_count: int
    max_steps_count: int
    answer_correct_count: int
    full_reward_count: int
    completion_rate: float
    answer_accuracy: float
    mean_reward: float
    mean_tool_coverage: float
    mean_latency_ms: float
    total_generated_tokens: int
    parity_pass_count: int
    parity_pass_rate: float
    mean_parity_absolute_error: float
    max_parity_absolute_error: float
    total_tokens_over_tolerance: int
    by_category: dict[str, CategorySummary]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return asdict(self)


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""

    if not values:
        raise ValueError("cannot compute mean of empty values")

    return sum(values) / len(values)


def _round_metric(value: float) -> float:
    """Round a benchmark metric for stable serialization."""

    return round(value, 6)


def _summarize_category(
    records: list[AgentBenchmarkRecord],
) -> CategorySummary:
    """Aggregate records belonging to one category."""

    task_count = len(records)
    completed_count = sum(
        record.status == "completed"
        for record in records
    )
    answer_correct_count = sum(
        record.answer_correct
        for record in records
    )
    parity_pass_count = sum(
        record.parity_within_tolerance
        for record in records
    )

    return CategorySummary(
        task_count=task_count,
        completed_count=completed_count,
        answer_correct_count=answer_correct_count,
        completion_rate=_round_metric(
            completed_count / task_count
        ),
        answer_accuracy=_round_metric(
            answer_correct_count / task_count
        ),
        mean_reward=_round_metric(
            _mean([record.reward for record in records])
        ),
        mean_tool_coverage=_round_metric(
            _mean(
                [
                    record.tool_coverage
                    for record in records
                ]
            )
        ),
        mean_latency_ms=_round_metric(
            _mean(
                [
                    record.latency_ms
                    for record in records
                ]
            )
        ),
        parity_pass_rate=_round_metric(
            parity_pass_count / task_count
        ),
    )


def build_benchmark_summary(
    records: list[AgentBenchmarkRecord],
) -> AgentBenchmarkSummary:
    """Aggregate task records into a suite-level report."""

    if not records:
        raise ValueError(
            "benchmark records must not be empty"
        )

    for record in records:
        record.validate()

    task_count = len(records)
    completed_count = sum(
        record.status == "completed"
        for record in records
    )
    protocol_error_count = sum(
        record.status == "protocol_error"
        for record in records
    )
    max_steps_count = sum(
        record.status == "max_steps"
        for record in records
    )
    answer_correct_count = sum(
        record.answer_correct
        for record in records
    )
    full_reward_count = sum(
        record.reward == 1.0
        for record in records
    )
    parity_pass_count = sum(
        record.parity_within_tolerance
        for record in records
    )

    categories = sorted(
        {record.category for record in records}
    )
    by_category = {
        category: _summarize_category(
            [
                record
                for record in records
                if record.category == category
            ]
        )
        for category in categories
    }

    summary = AgentBenchmarkSummary(
        task_count=task_count,
        completed_count=completed_count,
        protocol_error_count=protocol_error_count,
        max_steps_count=max_steps_count,
        answer_correct_count=answer_correct_count,
        full_reward_count=full_reward_count,
        completion_rate=_round_metric(
            completed_count / task_count
        ),
        answer_accuracy=_round_metric(
            answer_correct_count / task_count
        ),
        mean_reward=_round_metric(
            _mean([record.reward for record in records])
        ),
        mean_tool_coverage=_round_metric(
            _mean(
                [
                    record.tool_coverage
                    for record in records
                ]
            )
        ),
        mean_latency_ms=_round_metric(
            _mean(
                [
                    record.latency_ms
                    for record in records
                ]
            )
        ),
        total_generated_tokens=sum(
            record.generated_token_count
            for record in records
        ),
        parity_pass_count=parity_pass_count,
        parity_pass_rate=_round_metric(
            parity_pass_count / task_count
        ),
        mean_parity_absolute_error=_round_metric(
            _mean(
                [
                    record.parity_mean_absolute_error
                    for record in records
                ]
            )
        ),
        max_parity_absolute_error=max(
            record.parity_max_absolute_error
            for record in records
        ),
        total_tokens_over_tolerance=sum(
            record.parity_tokens_over_tolerance
            for record in records
        ),
        by_category=by_category,
    )

    return summary
