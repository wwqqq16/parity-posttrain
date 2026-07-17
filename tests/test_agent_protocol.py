import pytest

from parity_posttrain.agent.protocol import (
    FinalAnswerAction,
    ToolCallAction,
    parse_agent_action,
)


def test_parse_plain_tool_call() -> None:
    action = parse_agent_action(
        """
        {
          "type": "tool_call",
          "name": "calculator",
          "arguments": {"expression": "17 * 6"}
        }
        """
    )

    assert isinstance(action, ToolCallAction)
    assert action.name == "calculator"
    assert action.arguments == {"expression": "17 * 6"}


def test_parse_fenced_tool_call() -> None:
    response = (
        "```json\n"
        "{\n"
        '  "type": "tool_call",\n'
        '  "name": "product_catalog",\n'
        '  "arguments": {"product_id": "webcam"}\n'
        "}\n"
        "```"
    )

    action = parse_agent_action(response)

    assert isinstance(action, ToolCallAction)
    assert action.name == "product_catalog"
    assert action.arguments == {"product_id": "webcam"}


def test_parse_final_answer() -> None:
    action = parse_agent_action(
        '{"type": "final", "answer": "102"}'
    )

    assert isinstance(action, FinalAnswerAction)
    assert action.answer == "102"


def test_parse_json_surrounded_by_model_text() -> None:
    action = parse_agent_action(
        'Here is the requested action:\n'
        '{"type": "final", "answer": "50.23"}'
    )

    assert isinstance(action, FinalAnswerAction)
    assert action.answer == "50.23"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        parse_agent_action(
            """
            {
              "type": "tool_call",
              "name": "internet_search",
              "arguments": {}
            }
            """
        )


def test_non_object_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="arguments must be a JSON object",
    ):
        parse_agent_action(
            """
            {
              "type": "tool_call",
              "name": "calculator",
              "arguments": ["17 * 6"]
            }
            """
        )


def test_empty_final_answer_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="answer must not be empty",
    ):
        parse_agent_action(
            '{"type": "final", "answer": "   "}'
        )


def test_malformed_response_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="valid JSON object",
    ):
        parse_agent_action("I think the answer is 102.")
