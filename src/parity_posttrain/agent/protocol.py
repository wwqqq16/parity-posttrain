"""Structured action protocol for model-driven tool use."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from parity_posttrain.agent.tool_registry import available_tools


@dataclass(frozen=True, slots=True)
class ToolCallAction:
    """A request from the model to execute one registered tool."""

    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable action."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalAnswerAction:
    """A final answer returned by the model."""

    answer: str
    type: Literal["final"] = "final"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable action."""

        return asdict(self)


type AgentAction = ToolCallAction | FinalAnswerAction


def _json_candidates(text: str) -> list[str]:
    """Return likely JSON substrings from a model response."""

    stripped = text.strip()

    if not stripped:
        return []

    candidates = [stripped]

    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()

        if len(lines) >= 3:
            fenced_body = "\n".join(lines[1:-1]).strip()

            if fenced_body:
                candidates.append(fenced_body)

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")

    if object_start >= 0 and object_end > object_start:
        candidates.append(
            stripped[object_start : object_end + 1]
        )

    return list(dict.fromkeys(candidates))


def _load_action_payload(text: str) -> dict[str, Any]:
    """Load the first valid JSON object from a model response."""

    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)

    raise ValueError(
        "agent response does not contain a valid JSON object"
    )


def parse_agent_action(text: str) -> AgentAction:
    """Parse and validate one structured model action."""

    payload = _load_action_payload(text)
    action_type = payload.get("type")

    if action_type == "tool_call":
        raw_name = payload.get("name")
        raw_arguments = payload.get("arguments")

        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                "tool_call action requires a non-empty tool name"
            )

        name = raw_name.strip()

        if name not in available_tools():
            raise ValueError(f"unknown tool: {name}")

        if not isinstance(raw_arguments, dict):
            raise ValueError(
                "tool_call arguments must be a JSON object"
            )

        if not all(
            isinstance(key, str)
            for key in raw_arguments
        ):
            raise ValueError(
                "tool_call argument keys must be strings"
            )

        arguments = cast(dict[str, Any], raw_arguments)

        return ToolCallAction(
            name=name,
            arguments=arguments.copy(),
        )

    if action_type == "final":
        raw_answer = payload.get("answer")

        if not isinstance(raw_answer, str):
            raise ValueError(
                "final action requires a string answer"
            )

        answer = raw_answer.strip()

        if not answer:
            raise ValueError(
                "final action answer must not be empty"
            )

        return FinalAnswerAction(answer=answer)

    raise ValueError(
        "agent action type must be 'tool_call' or 'final'"
    )
