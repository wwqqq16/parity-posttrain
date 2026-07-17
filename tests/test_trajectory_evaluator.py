import pytest

from parity_posttrain.core.task import AgentTask
from parity_posttrain.core.trajectory import Message, Trajectory
from parity_posttrain.evals.trajectory_evaluator import (
    answers_match,
    evaluate_trajectory,
)


def build_shopping_task() -> AgentTask:
    return AgentTask(
        task_id="shopping_test",
        prompt="Find the webcam price and convert it to EUR.",
        expected_answer="50.23",
        required_tools=["product_catalog", "currency_converter"],
    )


def build_complete_trajectory(final_answer: str = "50.23") -> Trajectory:
    return Trajectory(
        task_id="shopping_test",
        messages=[
            Message(
                role="user",
                content="Find the webcam price and convert it to EUR.",
            ),
            Message(
                role="assistant",
                tool_call={
                    "name": "product_catalog",
                    "arguments": {"product_id": "webcam"},
                },
            ),
            Message(
                role="tool",
                name="product_catalog",
                content='{"price": 54.25, "currency": "USD"}',
            ),
            Message(
                role="assistant",
                tool_call={
                    "name": "currency_converter",
                    "arguments": {
                        "amount": 54.25,
                        "from_currency": "USD",
                        "to_currency": "EUR",
                    },
                },
            ),
            Message(
                role="tool",
                name="currency_converter",
                content="50.23",
            ),
            Message(role="assistant", content=final_answer),
        ],
    )


def test_complete_correct_trajectory_receives_full_reward() -> None:
    result = evaluate_trajectory(
        build_shopping_task(),
        build_complete_trajectory(),
    )

    assert result.answer_correct is True
    assert result.tool_coverage == 1.0
    assert result.missing_tools == []
    assert result.unexpected_tools == []
    assert result.reward == 1.0


def test_numeric_answer_can_appear_inside_text() -> None:
    result = evaluate_trajectory(
        build_shopping_task(),
        build_complete_trajectory("The final price is €50.23."),
    )

    assert result.answer_correct is True


def test_correct_answer_without_tools_receives_partial_reward() -> None:
    trajectory = Trajectory(
        task_id="shopping_test",
        messages=[
            Message(role="user", content="Solve the task."),
            Message(role="assistant", content="50.23"),
        ],
    )

    result = evaluate_trajectory(build_shopping_task(), trajectory)

    assert result.answer_correct is True
    assert result.tool_coverage == 0.0
    assert result.reward == 0.7


def test_wrong_answer_with_required_tools_receives_tool_credit() -> None:
    result = evaluate_trajectory(
        build_shopping_task(),
        build_complete_trajectory("49.00"),
    )

    assert result.answer_correct is False
    assert result.tool_coverage == 1.0
    assert result.reward == 0.3


def test_unexpected_tool_is_penalized() -> None:
    task = AgentTask(
        task_id="calculator_test",
        prompt="Calculate 6 * 7.",
        expected_answer="42",
        required_tools=["calculator"],
    )
    trajectory = Trajectory(
        task_id="calculator_test",
        messages=[
            Message(role="user", content="Calculate 6 * 7."),
            Message(
                role="assistant",
                tool_call={
                    "name": "calculator",
                    "arguments": {"expression": "6 * 7"},
                },
            ),
            Message(role="tool", name="calculator", content="42"),
            Message(
                role="assistant",
                tool_call={
                    "name": "currency_converter",
                    "arguments": {
                        "amount": 42,
                        "from_currency": "USD",
                        "to_currency": "EUR",
                    },
                },
            ),
            Message(
                role="tool",
                name="currency_converter",
                content="38.89",
            ),
            Message(role="assistant", content="42"),
        ],
    )

    result = evaluate_trajectory(task, trajectory)

    assert result.answer_correct is True
    assert result.unexpected_tools == ["currency_converter"]
    assert result.reward == pytest.approx(0.9)


def test_task_id_mismatch_is_rejected() -> None:
    trajectory = build_complete_trajectory()
    trajectory.task_id = "different_task"

    with pytest.raises(ValueError, match="task_id does not match"):
        evaluate_trajectory(build_shopping_task(), trajectory)


def test_answers_match_equivalent_numeric_formats() -> None:
    assert answers_match("50.23", "$50.230") is True
    assert answers_match("1,000.00", "The answer is 1000") is True
    assert answers_match("50.23", "50.24") is False
