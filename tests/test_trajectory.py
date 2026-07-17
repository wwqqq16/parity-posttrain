import pytest

from parity_posttrain.core.trajectory import Message, Trajectory


def test_valid_tool_use_trajectory() -> None:
    trajectory = Trajectory(
        task_id="calculator_001",
        messages=[
            Message(role="user", content="What is 17 * 6?"),
            Message(
                role="assistant",
                tool_call={
                    "name": "calculator",
                    "arguments": {"expression": "17 * 6"},
                },
            ),
            Message(role="tool", name="calculator", content="102"),
            Message(role="assistant", content="The answer is 102."),
        ],
        reward=1.0,
        token_ids=[10, 11, 12],
        token_logprobs=[-0.1, -0.2, -0.3],
    )

    trajectory.validate()

    result = trajectory.to_dict()
    assert result["task_id"] == "calculator_001"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["reward"] == 1.0


def test_completed_trajectory_must_end_with_assistant() -> None:
    trajectory = Trajectory(
        task_id="invalid_001",
        messages=[
            Message(role="user", content="Calculate something."),
            Message(
                role="assistant",
                tool_call={
                    "name": "calculator",
                    "arguments": {"expression": "1 + 1"},
                },
            ),
            Message(role="tool", name="calculator", content="2"),
        ],
    )

    with pytest.raises(ValueError, match="must end with an assistant"):
        trajectory.validate()


def test_token_ids_and_logprobs_must_align() -> None:
    trajectory = Trajectory(
        task_id="invalid_tokens",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ],
        token_ids=[1, 2, 3],
        token_logprobs=[-0.1, -0.2],
    )

    with pytest.raises(ValueError, match="must have equal lengths"):
        trajectory.validate()


def test_tool_result_requires_preceding_tool_call() -> None:
    trajectory = Trajectory(
        task_id="invalid_tool_sequence",
        messages=[
            Message(role="user", content="Calculate 1 + 1."),
            Message(role="assistant", content="I will calculate it."),
            Message(role="tool", name="calculator", content="2"),
            Message(role="assistant", content="The answer is 2."),
        ],
    )

    with pytest.raises(ValueError, match="must follow an assistant tool call"):
        trajectory.validate()
