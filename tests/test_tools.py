import pytest

from parity_posttrain.agent.tools import (
    calculator,
    currency_converter,
    product_catalog,
)


def test_calculator_basic_arithmetic() -> None:
    assert calculator("17 * 6") == 102
    assert calculator("10 / 4") == 2.5


def test_calculator_parentheses_and_unary_operator() -> None:
    assert calculator("-(3 + 4) * 2") == -14


def test_calculator_rejects_function_calls() -> None:
    with pytest.raises(ValueError, match="unsupported syntax"):
        calculator("__import__('os').system('echo unsafe')")


def test_calculator_rejects_large_exponent() -> None:
    with pytest.raises(ValueError, match="exponent is too large"):
        calculator("2 ** 100")


def test_calculator_rejects_division_by_zero() -> None:
    with pytest.raises(ValueError, match="division by zero"):
        calculator("5 / 0")


def test_product_catalog_returns_known_product() -> None:
    product = product_catalog("wireless_mouse")

    assert product["name"] == "Wireless Mouse"
    assert product["price"] == 24.99
    assert product["currency"] == "USD"


def test_product_catalog_is_case_insensitive() -> None:
    product = product_catalog("USB_C_HUB")

    assert product["name"] == "USB-C Hub"


def test_product_catalog_rejects_unknown_product() -> None:
    with pytest.raises(ValueError, match="unknown product_id"):
        product_catalog("spaceship")


def test_currency_converter_uses_fixed_rates() -> None:
    assert currency_converter(10, "EUR", "USD") == 10.8
    assert currency_converter(12.7, "GBP", "USD") == 16.13


def test_currency_converter_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        currency_converter(-1, "USD", "EUR")

    with pytest.raises(ValueError, match="unsupported destination currency"):
        currency_converter(10, "USD", "ABC")
