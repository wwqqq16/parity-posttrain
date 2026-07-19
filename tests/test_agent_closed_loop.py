"""Tests for the integrated agent closed-loop pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from parity_posttrain.core.task import AgentTask
from parity_posttrain.rollout.hf_backend import (
    GenerationResult,
    HuggingFaceRolloutBackend,
)
from parity_posttrain.training import (
    run_agent_closed_loop_experiment,
)


class TinyCausalModel(torch.nn.Module):
    """Small causal model used for closed-loop training."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_bias = torch.nn.Parameter(
            torch.tensor(
                [0.0, 0.2, -0.1, 0.1],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Return position-independent logits."""

        del attention_mask
        del use_cache
        del return_dict

        logits = self.logit_bias.view(
            1,
            1,
            -1,
        ).expand(
            input_ids.shape[0],
            input_ids.shape[1],
            -1,
        )

        return SimpleNamespace(logits=logits)


class FakeTokenizer:
    """Tokenizer surface required by the pipeline."""

    pad_token_id: int | None = 0


def make_generation(
    text: str,
    token_ids: list[int],
) -> GenerationResult:
    """Create one deterministic generation."""

    return GenerationResult(
        model_name="tiny-model",
        device="cpu",
        prompt_text="prompt",
        generated_text=text,
        prompt_token_ids=[1],
        generated_token_ids=token_ids,
        generated_token_logprobs=[
            -0.1
            for _ in token_ids
        ],
        latency_ms=1.0,
        tokens_per_second=100.0,
    )


class FakeBackend(HuggingFaceRolloutBackend):
    """HF-compatible backend with predetermined outputs."""

    def __init__(
        self,
        model: TinyCausalModel,
        generations: list[GenerationResult],
    ) -> None:
        self.model = model
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")
        self.generations = generations.copy()
        self.call_count = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        """Return the next deterministic generation."""

        del messages
        del max_new_tokens

        self.call_count += 1

        if not self.generations:
            raise RuntimeError(
                "fake backend has no generation left"
            )

        return self.generations.pop(0)


def make_task(task_id: str) -> AgentTask:
    """Create one calculator task."""

    return AgentTask(
        task_id=task_id,
        prompt=(
            "Use the calculator tool to evaluate 17 * 6."
        ),
        expected_answer="102",
        required_tools=["calculator"],
        metadata={
            "category": "calculator",
        },
    )


def write_artifact(
    path: Path,
    model: TinyCausalModel,
) -> None:
    """Write a two-task training artifact."""

    logprobs = torch.log_softmax(
        model.logit_bias.detach(),
        dim=-1,
    )

    def task_payload(
        *,
        task_id: str,
        status: str,
        reward: float,
        answer_correct: bool,
        generated_token_id: int,
    ) -> dict[str, object]:
        return {
            "benchmark_record": {
                "task_id": task_id,
                "category": "calculator",
                "status": status,
                "reward": reward,
                "answer_correct": answer_correct,
                "generation_count": 1,
                "generated_token_count": 1,
            },
            "run": {
                "generations": [
                    {
                        "model_name": "tiny-model",
                        "prompt_token_ids": [1],
                        "generated_token_ids": [
                            generated_token_id
                        ],
                        "generated_token_logprobs": [
                            float(
                                logprobs[
                                    generated_token_id
                                ].item()
                            )
                        ],
                    }
                ]
            },
        }

    payload = {
        "tasks": [
            task_payload(
                task_id="positive",
                status="completed",
                reward=1.0,
                answer_correct=True,
                generated_token_id=2,
            ),
            task_payload(
                task_id="negative",
                status="protocol_error",
                reward=0.0,
                answer_correct=False,
                generated_token_id=3,
            ),
        ]
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def completed_generations() -> list[GenerationResult]:
    """Return two successful two-turn task rollouts."""

    tool_call = (
        '{"type":"tool_call",'
        '"name":"calculator",'
        '"arguments":{"expression":"17 * 6"}}'
    )
    final = '{"type":"final","answer":"102"}'

    return [
        make_generation(tool_call, [1]),
        make_generation(final, [2]),
        make_generation(tool_call, [1]),
        make_generation(final, [2]),
    ]


def test_runs_persistent_agent_closed_loop(
    tmp_path: Path,
) -> None:
    model = TinyCausalModel()
    model.logit_bias.requires_grad_(False)
    initial_parameters = (
        model.logit_bias.detach().clone()
    )
    artifact_path = tmp_path / "benchmark.json"
    write_artifact(artifact_path, model)

    backend = FakeBackend(
        model,
        completed_generations(),
    )

    summary = run_agent_closed_loop_experiment(
        backend=backend,
        artifact_path=artifact_path,
        tasks=(
            make_task("positive"),
            make_task("negative"),
        ),
        trainable_parameter_names=("logit_bias",),
        steps=2,
        learning_rate=0.1,
        max_gradient_norm=10.0,
    )

    assert summary.optimizer_steps == 2
    assert summary.before.total_reward == pytest.approx(
        1.0
    )
    assert summary.after.total_reward == pytest.approx(
        2.0
    )
    assert summary.reward_delta == pytest.approx(1.0)
    assert backend.call_count == 4

    assert not torch.equal(
        initial_parameters,
        model.logit_bias.detach(),
    )
    assert model.logit_bias.requires_grad is False


def test_rejects_non_integer_pad_token(
    tmp_path: Path,
) -> None:
    model = TinyCausalModel()
    artifact_path = tmp_path / "benchmark.json"
    write_artifact(artifact_path, model)

    backend = FakeBackend(model, [])
    backend.tokenizer.pad_token_id = None

    with pytest.raises(
        ValueError,
        match="no integer pad_token_id",
    ):
        run_agent_closed_loop_experiment(
            backend=backend,
            artifact_path=artifact_path,
            tasks=(
                make_task("positive"),
                make_task("negative"),
            ),
            trainable_parameter_names=(
                "logit_bias",
            ),
            steps=1,
            learning_rate=0.1,
        )


def test_restores_gradient_flags_on_rerollout_error(
    tmp_path: Path,
) -> None:
    model = TinyCausalModel()
    model.logit_bias.requires_grad_(False)
    artifact_path = tmp_path / "benchmark.json"
    write_artifact(artifact_path, model)

    backend = FakeBackend(model, [])

    with pytest.raises(
        RuntimeError,
        match="no generation left",
    ):
        run_agent_closed_loop_experiment(
            backend=backend,
            artifact_path=artifact_path,
            tasks=(
                make_task("positive"),
                make_task("negative"),
            ),
            trainable_parameter_names=(
                "logit_bias",
            ),
            steps=1,
            learning_rate=0.1,
        )

    assert model.logit_bias.requires_grad is False
