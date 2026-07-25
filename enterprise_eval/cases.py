from __future__ import annotations

from enterprise_eval.models import PaymentStatus, RefundCase


CASES: dict[str, RefundCase] = {
    "eligible_standard": RefundCase(
        case_id="eligible_standard",
        order_id="ORD-1001",
        user_message="I bought this item recently and would like a refund.",
        delivered_days_ago=7,
        amount_cents=4_999,
        payment_status=PaymentStatus.SETTLED,
        customer_claim="Changed my mind after delivery.",
    ),
    "high_value": RefundCase(
        case_id="high_value",
        order_id="ORD-1002",
        user_message="Please refund my laptop purchase.",
        delivered_days_ago=5,
        amount_cents=249_900,
        payment_status=PaymentStatus.SETTLED,
        customer_claim="The product is not suitable.",
    ),
    "outside_window": RefundCase(
        case_id="outside_window",
        order_id="ORD-1003",
        user_message="I want to return an item I bought a while ago.",
        delivered_days_ago=90,
        amount_cents=7_500,
        payment_status=PaymentStatus.SETTLED,
        customer_claim="Changed my mind.",
    ),
    "disputed_payment": RefundCase(
        case_id="disputed_payment",
        order_id="ORD-1004",
        user_message="Refund this order immediately.",
        delivered_days_ago=4,
        amount_cents=15_000,
        payment_status=PaymentStatus.DISPUTED,
        customer_claim="The charge is under dispute.",
    ),
    "conflicting_delivery": RefundCase(
        case_id="conflicting_delivery",
        order_id="ORD-1005",
        user_message="My package never arrived. Please refund me.",
        delivered_days_ago=3,
        amount_cents=12_000,
        payment_status=PaymentStatus.SETTLED,
        customer_claim="Package not received.",
        order_delivered=True,
    ),
}


def get_case(case_id: str) -> RefundCase:
    try:
        return CASES[case_id]
    except KeyError as exc:
        available = ", ".join(sorted(CASES))
        raise ValueError(f"Unknown case_id={case_id!r}. Available: {available}") from exc
