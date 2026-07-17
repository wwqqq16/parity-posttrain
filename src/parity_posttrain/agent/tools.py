"""Deterministic local tools used by agent rollouts."""

from __future__ import annotations

import ast
import math
from typing import Final, TypedDict

Number = int | float


class ProductRecord(TypedDict):
    """JSON-serializable product information."""

    name: str
    price: float
    currency: str


_PRODUCT_CATALOG: Final[dict[str, ProductRecord]] = {
    "wireless_mouse": {
        "name": "Wireless Mouse",
        "price": 24.99,
        "currency": "USD",
    },
    "mechanical_keyboard": {
        "name": "Mechanical Keyboard",
        "price": 79.50,
        "currency": "USD",
    },
    "usb_c_hub": {
        "name": "USB-C Hub",
        "price": 39.00,
        "currency": "USD",
    },
    "webcam": {
        "name": "Webcam",
        "price": 54.25,
        "currency": "USD",
    },
    "laptop_stand": {
        "name": "Laptop Stand",
        "price": 31.75,
        "currency": "USD",
    },
}

# Fixed rates make experiments deterministic.
# Each value means: one unit of this currency is worth this many USD.
_USD_PER_UNIT: Final[dict[str, float]] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CAD": 0.73,
}


def _evaluate_expression(node: ast.AST) -> Number:
    """Recursively evaluate a restricted arithmetic expression."""

    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_expression(node.operand)

        if isinstance(node.op, ast.UAdd):
            return operand

        if isinstance(node.op, ast.USub):
            return -operand

        raise ValueError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            result = left / right
        elif isinstance(node.op, ast.Pow):
            if not float(right).is_integer():
                raise ValueError("power exponent must be an integer")
            if abs(right) > 10:
                raise ValueError("power exponent is too large")
            result = left**right
        else:
            raise ValueError("unsupported binary operator")

        numeric_result = float(result)
        if not math.isfinite(numeric_result):
            raise ValueError("result must be finite")
        if abs(numeric_result) > 1_000_000_000_000:
            raise ValueError("result magnitude is too large")

        return result

    raise ValueError("expression contains unsupported syntax")


def calculator(expression: str) -> Number:
    """Safely evaluate a basic arithmetic expression.

    Supported operations are addition, subtraction, multiplication,
    division, integer powers, unary plus/minus, and parentheses.
    """

    expression = expression.strip()

    if not expression:
        raise ValueError("expression must not be empty")

    if len(expression) > 100:
        raise ValueError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid arithmetic expression") from error

    result = _evaluate_expression(tree)

    if isinstance(result, float):
        if result.is_integer():
            return int(result)
        return round(result, 10)

    return result


def product_catalog(product_id: str) -> ProductRecord:
    """Return information for a product in the fixed local catalog."""

    normalized_id = product_id.strip().lower()

    if not normalized_id:
        raise ValueError("product_id must not be empty")

    try:
        product = _PRODUCT_CATALOG[normalized_id]
    except KeyError as error:
        raise ValueError(f"unknown product_id: {normalized_id}") from error

    return product.copy()


def currency_converter(
    amount: Number,
    from_currency: str,
    to_currency: str,
) -> float:
    """Convert an amount using deterministic fixed exchange rates."""

    numeric_amount = float(amount)

    if not math.isfinite(numeric_amount):
        raise ValueError("amount must be finite")

    if numeric_amount < 0:
        raise ValueError("amount must not be negative")

    source = from_currency.strip().upper()
    destination = to_currency.strip().upper()

    if source not in _USD_PER_UNIT:
        raise ValueError(f"unsupported source currency: {source}")

    if destination not in _USD_PER_UNIT:
        raise ValueError(f"unsupported destination currency: {destination}")

    amount_in_usd = numeric_amount * _USD_PER_UNIT[source]
    converted = amount_in_usd / _USD_PER_UNIT[destination]

    return round(converted, 2)
