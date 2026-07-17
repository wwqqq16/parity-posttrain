"""Deterministic task generation for agent trajectory benchmarks."""

from __future__ import annotations

from parity_posttrain.agent.tools import (
    calculator,
    currency_converter,
    product_catalog,
)
from parity_posttrain.core.task import AgentTask


def build_sample_tasks() -> list[AgentTask]:
    """Build a deterministic collection of agent evaluation tasks."""

    tasks: list[AgentTask] = []

    calculator_cases = [
        ("17 * 6", "102"),
        ("(14 + 8) / 2", "11"),
        ("7 * 9 - 13", "50"),
        ("(25 - 7) * 3", "54"),
        ("144 / 12 + 5", "17"),
        ("2 ** 6 - 9", "55"),
    ]

    for index, (expression, expected) in enumerate(calculator_cases, start=1):
        tasks.append(
            AgentTask(
                task_id=f"calculator_{index:03d}",
                prompt=(
                    f"Use the calculator tool to evaluate: {expression}. "
                    "Return only the final numeric answer."
                ),
                expected_answer=expected,
                required_tools=["calculator"],
                metadata={
                    "category": "calculator",
                    "difficulty": "single_tool",
                    "expression": expression,
                },
            )
        )

    product_ids = [
        "wireless_mouse",
        "mechanical_keyboard",
        "usb_c_hub",
        "webcam",
        "laptop_stand",
    ]

    for index, product_id in enumerate(product_ids, start=1):
        product = product_catalog(product_id)
        expected = f"{product['price']:.2f}"

        tasks.append(
            AgentTask(
                task_id=f"catalog_{index:03d}",
                prompt=(
                    f"Look up the product '{product_id}' in the product catalog. "
                    "Return only its price in USD."
                ),
                expected_answer=expected,
                required_tools=["product_catalog"],
                metadata={
                    "category": "catalog",
                    "difficulty": "single_tool",
                    "product_id": product_id,
                },
            )
        )

    conversion_cases = [
        (100, "USD", "EUR"),
        (50, "EUR", "USD"),
        (25, "GBP", "USD"),
        (10_000, "JPY", "USD"),
        (80, "CAD", "USD"),
    ]

    for index, (amount, source, destination) in enumerate(
        conversion_cases,
        start=1,
    ):
        converted = currency_converter(amount, source, destination)

        tasks.append(
            AgentTask(
                task_id=f"currency_{index:03d}",
                prompt=(
                    f"Use the currency converter to convert {amount} "
                    f"{source} to {destination}. Return only the converted amount."
                ),
                expected_answer=f"{converted:.2f}",
                required_tools=["currency_converter"],
                metadata={
                    "category": "currency",
                    "difficulty": "single_tool",
                    "amount": amount,
                    "from_currency": source,
                    "to_currency": destination,
                },
            )
        )

    shopping_cases = [
        ("wireless_mouse", "EUR"),
        ("mechanical_keyboard", "GBP"),
        ("usb_c_hub", "CAD"),
        ("webcam", "EUR"),
        ("laptop_stand", "GBP"),
    ]

    for index, (product_id, destination) in enumerate(shopping_cases, start=1):
        product = product_catalog(product_id)
        converted = currency_converter(
            product["price"],
            product["currency"],
            destination,
        )

        tasks.append(
            AgentTask(
                task_id=f"shopping_{index:03d}",
                prompt=(
                    f"Look up '{product_id}', then convert its price from "
                    f"{product['currency']} to {destination}. "
                    "Return only the converted amount."
                ),
                expected_answer=f"{converted:.2f}",
                required_tools=[
                    "product_catalog",
                    "currency_converter",
                ],
                metadata={
                    "category": "shopping",
                    "difficulty": "multi_tool",
                    "product_id": product_id,
                    "to_currency": destination,
                },
            )
        )

    basket_cases = [
        ("wireless_mouse", "usb_c_hub", "EUR"),
        ("webcam", "laptop_stand", "GBP"),
        ("mechanical_keyboard", "wireless_mouse", "CAD"),
    ]

    for index, (first_id, second_id, destination) in enumerate(
        basket_cases,
        start=1,
    ):
        first = product_catalog(first_id)
        second = product_catalog(second_id)

        total_usd = calculator(f"{first['price']} + {second['price']}")
        converted = currency_converter(total_usd, "USD", destination)

        tasks.append(
            AgentTask(
                task_id=f"basket_{index:03d}",
                prompt=(
                    f"Look up '{first_id}' and '{second_id}', calculate their "
                    f"combined USD price, then convert the total to {destination}. "
                    "Return only the converted amount."
                ),
                expected_answer=f"{converted:.2f}",
                required_tools=[
                    "product_catalog",
                    "calculator",
                    "currency_converter",
                ],
                metadata={
                    "category": "basket",
                    "difficulty": "three_tool",
                    "product_ids": [first_id, second_id],
                    "to_currency": destination,
                },
            )
        )

    for task in tasks:
        task.validate()

    return tasks
