import pytest

from parity_posttrain.core.task import AgentTask


def test_valid_agent_task() -> None:
    task = AgentTask(
        task_id="shopping_001",
        prompt="Find the price of the webcam and convert it to EUR.",
        expected_answer="50.23",
        required_tools=["product_catalog", "currency_converter"],
        metadata={"category": "shopping"},
    )

    task.validate()

    result = task.to_dict()

    assert result["task_id"] == "shopping_001"
    assert result["required_tools"] == [
        "product_catalog",
        "currency_converter",
    ]


def test_task_requires_non_empty_prompt() -> None:
    task = AgentTask(
        task_id="invalid_prompt",
        prompt="",
        expected_answer="42",
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        task.validate()


def test_task_rejects_unknown_tool() -> None:
    task = AgentTask(
        task_id="invalid_tool",
        prompt="Search the internet.",
        expected_answer="result",
        required_tools=["internet_search"],
    )

    with pytest.raises(ValueError, match="unknown tools"):
        task.validate()


def test_task_rejects_duplicate_tools() -> None:
    task = AgentTask(
        task_id="duplicate_tools",
        prompt="Calculate a value.",
        expected_answer="2",
        required_tools=["calculator", "calculator"],
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        task.validate()
