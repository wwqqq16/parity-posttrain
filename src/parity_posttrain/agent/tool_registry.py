"""Unified registry and execution interface for local agent tools."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Final, cast

from parity_posttrain.agent.tools import (
    calculator,
    currency_converter,
    product_catalog,
)

ToolFunction = Callable[..., object]


@dataclass(slots=True)
class ToolExecution:
    """The result of executing one structured tool call."""

    name: str
    arguments: dict[str, Any]
    result: object

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable execution record."""

        return asdict(self)

    def to_message_content(self) -> str:
        """Serialize the result for insertion into a tool message."""

        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, sort_keys=True)

        return str(self.result)


_TOOL_FUNCTIONS: Final[dict[str, ToolFunction]] = {
    "calculator": calculator,
    "product_catalog": product_catalog,
    "currency_converter": currency_converter,
}


_TOOL_SCHEMAS: Final[list[dict[str, Any]]] = [
    {
        "name": "calculator",
        "description": "Safely evaluate a basic arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "product_catalog",
        "description": "Look up a product in the deterministic local catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "Identifier of the product.",
                }
            },
            "required": ["product_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "currency_converter",
        "description": "Convert money using deterministic fixed exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Non-negative amount to convert.",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Destination currency code.",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False,
        },
    },
]


def available_tools() -> tuple[str, ...]:
    """Return supported tool names in stable sorted order."""

    return tuple(sorted(_TOOL_FUNCTIONS))


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return a defensive copy of the model-facing tool schemas."""

    return copy.deepcopy(_TOOL_SCHEMAS)


def execute_tool_call(tool_call: dict[str, Any]) -> ToolExecution:
    """Validate and execute a structured tool call."""

    name = tool_call.get("name")
    raw_arguments = tool_call.get("arguments", {})

    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call must contain a non-empty string name")

    normalized_name = name.strip()

    if normalized_name not in _TOOL_FUNCTIONS:
        raise ValueError(f"unknown tool: {normalized_name}")

    if not isinstance(raw_arguments, dict):
        raise ValueError("tool arguments must be an object")

    if not all(isinstance(key, str) for key in raw_arguments):
        raise ValueError("tool argument keys must be strings")

    arguments = cast(dict[str, Any], raw_arguments)
    tool_function = _TOOL_FUNCTIONS[normalized_name]

    try:
        result = tool_function(**arguments)
    except TypeError as error:
        raise ValueError(
            f"invalid arguments for tool: {normalized_name}"
        ) from error

    return ToolExecution(
        name=normalized_name,
        arguments=arguments.copy(),
        result=result,
    )
