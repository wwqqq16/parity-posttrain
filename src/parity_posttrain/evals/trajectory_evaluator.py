"""Deterministic evaluation of tool-use agent trajectories."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from parity_posttrain.core.task import AgentTask
from parity_posttrain.core.trajectory import Trajectory

_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
)


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation result for one trajectory."""

    task_id: str
    answer_correct: bool
    expected_answer: str
    final_answer: str
    tool_coverage: float
    used_tools: list[str]
    missing_tools: list[str]
    unexpected_tools: list[str]
    tool_call_count: int
    reward: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _extract_number(text: str) -> Decimal | None:
    """Extract the final numeric value appearing in text."""

    normalized = text.replace(",", "")
    matches = _NUMBER_PATTERN.findall(normalized)

    if not matches:
        return None

    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def answers_match(expected: str, actual: str) -> bool:
    """Compare numeric answers when possible, otherwise compare normalized text."""

    expected_number = _extract_number(expected)
    actual_number = _extract_number(actual)

    if expected_number is not None and actual_number is not None:
        return abs(expected_number - actual_number) <= Decimal("0.005")

    normalized_expected = " ".join(expected.lower().split())
    normalized_actual = " ".join(actual.lower().split())

    return normalized_expected == normalized_actual


def extract_used_tools(trajectory: Trajectory) -> list[str]:
    """Extract tool names from assistant tool-call messages."""

    used_tools: list[str] = []

    for message in trajectory.messages:
        if message.role != "assistant" or message.tool_call is None:
            continue

        name = message.tool_call.get("name")

        if isinstance(name, str) and name.strip():
            used_tools.append(name.strip())

    return used_tools


def evaluate_trajectory(
    task: AgentTask,
    trajectory: Trajectory,
) -> EvaluationResult:
    """Evaluate answer correctness and tool-use quality."""

    task.validate()
    trajectory.validate()

    if trajectory.task_id != task.task_id:
        raise ValueError(
            "trajectory task_id does not match the evaluation task"
        )

    final_answer = trajectory.messages[-1].content.strip()
    answer_correct = answers_match(task.expected_answer, final_answer)

    used_tools = extract_used_tools(trajectory)
    required_tool_set = set(task.required_tools)
    used_tool_set = set(used_tools)

    missing_tools = sorted(required_tool_set - used_tool_set)
    unexpected_tools = sorted(used_tool_set - required_tool_set)

    if task.required_tools:
        covered_count = len(required_tool_set) - len(missing_tools)
        tool_coverage = covered_count / len(required_tool_set)
    else:
        tool_coverage = 1.0

    answer_score = 1.0 if answer_correct else 0.0
    unexpected_penalty = 0.1 * len(unexpected_tools)

    reward = (
        0.7 * answer_score
        + 0.3 * tool_coverage
        - unexpected_penalty
    )
    reward = round(max(0.0, min(1.0, reward)), 4)

    return EvaluationResult(
        task_id=task.task_id,
        answer_correct=answer_correct,
        expected_answer=task.expected_answer,
        final_answer=final_answer,
        tool_coverage=round(tool_coverage, 4),
        used_tools=used_tools,
        missing_tools=missing_tools,
        unexpected_tools=unexpected_tools,
        tool_call_count=len(used_tools),
        reward=reward,
    )
