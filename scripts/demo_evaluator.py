"""Demonstrate deterministic evaluation of an agent trajectory."""

from __future__ import annotations

import json

from parity_posttrain.core.task import AgentTask
from parity_posttrain.core.trajectory import Message, Trajectory
from parity_posttrain.evals.trajectory_evaluator import evaluate_trajectory


def main() -> None:
    """Evaluate one successful multi-tool trajectory."""

    task = AgentTask(
        task_id="shopping_demo",
        prompt="Find the webcam price and convert it to EUR.",
        expected_answer="50.23",
        required_tools=["product_catalog", "currency_converter"],
    )

    trajectory = Trajectory(
        task_id="shopping_demo",
        messages=[
            Message(role="user", content=task.prompt),
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
            Message(
                role="assistant",
                content="The final price is €50.23.",
            ),
        ],
    )

    result = evaluate_trajectory(task, trajectory)

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
