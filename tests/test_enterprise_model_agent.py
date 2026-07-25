"""Tests for the model-backed enterprise agent loop."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from enterprise_eval.cases import CASES
from enterprise_eval.environment import RefundEnvironment
from enterprise_eval.model_agent import (
    ModelBackedRefundAgent,
    parse_model_action,
)
from enterprise_eval.models import ActionType


@dataclass
class FakeGeneration:
    model_name: str
    device: str
    prompt_text: str
    generated_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_token_logprobs: list[float]
    latency_ms: float
    tokens_per_second: float


class FakeBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> FakeGeneration:
        del max_new_tokens
        if self.calls >= len(self.outputs):
            raise AssertionError("fake backend ran out of outputs")
        output = self.outputs[self.calls]
        self.calls += 1
        generated_ids = list(range(100, 100 + max(1, len(output.split()))))
        return FakeGeneration(
            model_name="fake-model",
            device="cpu",
            prompt_text="\n".join(message["content"] for message in messages),
            generated_text=output,
            prompt_token_ids=[1, 2, 3],
            generated_token_ids=generated_ids,
            generated_token_logprobs=[-0.1] * len(generated_ids),
            latency_ms=1.0,
            tokens_per_second=10.0,
        )


def test_parses_markdown_wrapped_action() -> None:
    parsed = parse_model_action(
        '```json\n{"action":"get_order","arguments":{"order_id":"ORD-1"}}\n```'
    )
    assert parsed.action.action_type is ActionType.GET_ORDER
    assert parsed.action.arguments == {"order_id": "ORD-1"}


def test_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        parse_model_action('{"action":"delete_database","arguments":{}}')


def test_rejects_missing_required_argument() -> None:
    with pytest.raises(ValueError, match="requires a non-empty order_id"):
        parse_model_action('{"action":"issue_refund","arguments":{}}')


def test_runs_successful_model_backed_refund() -> None:
    backend = FakeBackend(
        [
            '{"action":"get_order","arguments":{"order_id":"ORD-1001"}}',
            '{"action":"check_refund_policy","arguments":{"order_id":"ORD-1001"}}',
            '{"action":"get_payment_status","arguments":{"order_id":"ORD-1001"}}',
            '{"action":"issue_refund","arguments":{"order_id":"ORD-1001"}}',
            '{"action":"respond","arguments":{"message":"Your refund has been issued."}}',
        ]
    )
    env = RefundEnvironment(CASES["eligible_standard"])

    ModelBackedRefundAgent(backend).run(env)

    assert env.run is not None
    assert env.run.architecture == "model"
    assert env.run.completed
    assert env.state.refund_issued
    assert env.run.component_calls == 5
    records = env.run.metadata["model_generations"]
    assert isinstance(records, list)
    assert len(records) == 5
    assert records[0]["prompt_token_ids"] == [1, 2, 3]
    assert len(records[0]["generated_token_ids"]) == len(
        records[0]["generated_token_logprobs"]
    )


def test_protocol_error_safely_escalates() -> None:
    backend = FakeBackend(["not valid JSON"])
    env = RefundEnvironment(CASES["eligible_standard"])

    ModelBackedRefundAgent(backend).run(env)

    assert env.run is not None
    assert env.run.completed
    assert env.state.human_review_requested
    assert not env.state.refund_issued
    protocol_errors = env.run.metadata["protocol_errors"]
    assert isinstance(protocol_errors, list)
    assert len(protocol_errors) == 1


def test_action_budget_exhaustion_safely_escalates() -> None:
    backend = FakeBackend(
        [
            '{"action":"get_order","arguments":{"order_id":"ORD-1001"}}',
        ]
    )
    env = RefundEnvironment(CASES["eligible_standard"])

    ModelBackedRefundAgent(backend, max_steps=1).run(env)

    assert env.run is not None
    assert env.run.completed
    assert env.state.human_review_requested
    assert env.run.component_calls == 1
