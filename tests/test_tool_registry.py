import json

import pytest

from parity_posttrain.agent.tool_registry import (
    available_tools,
    execute_tool_call,
    get_tool_schemas,
)


def test_available_tools_are_stable() -> None:
    assert available_tools() == (
        "calculator",
        "currency_converter",
        "product_catalog",
    )


def test_execute_calculator_tool_call() -> None:
    execution = execute_tool_call(
        {
            "name": "calculator",
            "arguments": {"expression": "(10 + 2) * 3"},
        }
    )

    assert execution.name == "calculator"
    assert execution.result == 36
    assert execution.to_message_content() == "36"


def test_execute_product_catalog_tool_call() -> None:
    execution = execute_tool_call(
        {
            "name": "product_catalog",
            "arguments": {"product_id": "webcam"},
        }
    )

    result = json.loads(execution.to_message_content())

    assert result["name"] == "Webcam"
    assert result["price"] == 54.25


def test_tool_schemas_are_defensive_copies() -> None:
    first = get_tool_schemas()
    first[0]["name"] = "modified"

    second = get_tool_schemas()

    assert second[0]["name"] == "calculator"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        execute_tool_call(
            {
                "name": "internet_search",
                "arguments": {},
            }
        )


def test_invalid_tool_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid arguments"):
        execute_tool_call(
            {
                "name": "calculator",
                "arguments": {"wrong_key": "1 + 1"},
            }
        )
