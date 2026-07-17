"""Task definitions for deterministic agent evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from parity_posttrain.agent.tool_registry import available_tools


@dataclass(slots=True)
class AgentTask:
    """A deterministic task used to produce and evaluate a trajectory."""

    task_id: str
    prompt: str
    expected_answer: str
    required_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the task definition."""

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")

        if not self.expected_answer.strip():
            raise ValueError("expected_answer must not be empty")

        if len(self.required_tools) != len(set(self.required_tools)):
            raise ValueError("required_tools must not contain duplicates")

        supported_tools = set(available_tools())
        unknown_tools = set(self.required_tools) - supported_tools

        if unknown_tools:
            names = ", ".join(sorted(unknown_tools))
            raise ValueError(f"task contains unknown tools: {names}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        self.validate()
        return asdict(self)
