"""Core data structures for multi-turn agent trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    """One message in a multi-turn agent trajectory."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_call: dict[str, Any] | None = None


@dataclass(slots=True)
class Trajectory:
    """A complete rollout produced for one agent task."""

    task_id: str
    messages: list[Message]
    reward: float | None = None
    latency_ms: float | None = None
    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate structural invariants of the trajectory."""

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if not self.messages:
            raise ValueError("trajectory must contain at least one message")

        if not any(message.role == "user" for message in self.messages):
            raise ValueError("trajectory must contain a user message")

        if self.messages[-1].role != "assistant":
            raise ValueError(
                "a completed trajectory must end with an assistant message"
            )

        for index, message in enumerate(self.messages):
            if message.role != "tool":
                continue

            if index == 0:
                raise ValueError("a tool result cannot be the first message")

            previous = self.messages[index - 1]
            if previous.role != "assistant" or previous.tool_call is None:
                raise ValueError(
                    "every tool result must follow an assistant tool call"
                )

        if self.token_logprobs and len(self.token_ids) != len(self.token_logprobs):
            raise ValueError(
                "token_ids and token_logprobs must have equal lengths"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        self.validate()
        return asdict(self)
