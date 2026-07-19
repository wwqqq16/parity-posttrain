"""Tests for closed-loop policy-training execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from parity_posttrain.training import (
    ClosedLoopTaskSnapshot,
    build_trajectory_training_example,
    collate_training_examples,
    fingerprint_generated_token_ids,
    run_closed_loop_experiment,
)


class TinyCausalModel(torch.nn.Module):
    """Small trainable causal model."""

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


def make_snapshot(
    *,
    task_id: str,
    status: str,
    reward: float,
    answer_correct: bool,
) -> ClosedLoopTaskSnapshot:
    """Create one task snapshot."""

    return ClosedLoopTaskSnapshot(
        task_id=task_id,
        category=task_id,
        status=status,
        reward=reward,
        answer_correct=answer_correct,
        generation_count=1,
        generated_token_count=1,
        trajectory_fingerprint=(
            fingerprint_generated_token_ids(
                ((1,),)
            )
        ),
    )


def make_before_snapshots() -> tuple[
    ClosedLoopTaskSnapshot,
    ...,
]:
    """Create one successful and one failed task."""

    return (
        make_snapshot(
            task_id="positive",
            status="completed",
            reward=1.0,
            answer_correct=True,
        ),
        make_snapshot(
            task_id="negative",
            status="protocol_error",
            reward=0.0,
            answer_correct=False,
        ),
    )


def make_batch(
    model: TinyCausalModel,
):
    """Create a controlled two-example batch."""

    logprobs = torch.log_softmax(
        model.logit_bias.detach(),
        dim=-1,
    )

    positive = build_trajectory_training_example(
        task_id="positive",
        turn_index=0,
        status="completed",
        model_name="tiny-model",
        reward=1.0,
        prompt_token_ids=[1],
        generated_token_ids=[2],
        rollout_logprobs=[
            float(logprobs[2].item())
        ],
    )
    negative = build_trajectory_training_example(
        task_id="negative",
        turn_index=0,
        status="protocol_error",
        model_name="tiny-model",
        reward=0.0,
        prompt_token_ids=[1],
        generated_token_ids=[3],
        rollout_logprobs=[
            float(logprobs[3].item())
        ],
    )

    return collate_training_examples(
        (positive, negative),
        pad_token_id=0,
    )


def test_runs_training_then_rerollout() -> None:
    model = TinyCausalModel()
    model.train()
    before_parameters = (
        model.logit_bias.detach().clone()
    )
    callback_state: dict[str, bool] = {}

    def rerollout(
        updated_model: torch.nn.Module,
    ) -> tuple[ClosedLoopTaskSnapshot, ...]:
        callback_state["evaluation_mode"] = (
            not updated_model.training
        )
        callback_state["parameters_changed"] = (
            not torch.equal(
                before_parameters,
                model.logit_bias.detach(),
            )
        )

        return (
            make_snapshot(
                task_id="positive",
                status="completed",
                reward=1.0,
                answer_correct=True,
            ),
            make_snapshot(
                task_id="negative",
                status="completed",
                reward=1.0,
                answer_correct=True,
            ),
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    summary = run_closed_loop_experiment(
        model=model,
        optimizer=optimizer,
        batch=make_batch(model),
        before_snapshots=make_before_snapshots(),
        rerollout=rerollout,
        source_artifact="artifact.json",
        model_name="tiny-model",
        steps=2,
        normalization="trajectory",
        max_gradient_norm=10.0,
    )

    assert callback_state == {
        "evaluation_mode": True,
        "parameters_changed": True,
    }
    assert model.training is True
    assert not torch.equal(
        before_parameters,
        model.logit_bias.detach(),
    )

    assert summary.optimizer_steps == 2
    assert len(summary.training_steps) == 2
    assert summary.reward_delta == pytest.approx(1.0)
    assert summary.completed_count_delta == 1
    assert summary.correct_answer_count_delta == 1
    assert summary.tasks[1].reward_delta == pytest.approx(
        1.0
    )


def test_rejects_after_task_order_mismatch() -> None:
    model = TinyCausalModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    def rerollout(
        updated_model: torch.nn.Module,
    ) -> tuple[ClosedLoopTaskSnapshot, ...]:
        del updated_model

        return tuple(
            reversed(make_before_snapshots())
        )

    with pytest.raises(
        ValueError,
        match="task IDs must match",
    ):
        run_closed_loop_experiment(
            model=model,
            optimizer=optimizer,
            batch=make_batch(model),
            before_snapshots=make_before_snapshots(),
            rerollout=rerollout,
            source_artifact="artifact.json",
            model_name="tiny-model",
            steps=1,
        )


def test_rejects_duplicate_before_task_ids() -> None:
    model = TinyCausalModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    duplicate = make_before_snapshots()[0]

    with pytest.raises(
        ValueError,
        match="before task IDs must be unique",
    ):
        run_closed_loop_experiment(
            model=model,
            optimizer=optimizer,
            batch=make_batch(model),
            before_snapshots=(
                duplicate,
                duplicate,
            ),
            rerollout=lambda _: (
                duplicate,
                duplicate,
            ),
            source_artifact="artifact.json",
            model_name="tiny-model",
            steps=1,
        )


def test_rejects_empty_after_snapshots() -> None:
    model = TinyCausalModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    with pytest.raises(
        ValueError,
        match="after snapshots must not be empty",
    ):
        run_closed_loop_experiment(
            model=model,
            optimizer=optimizer,
            batch=make_batch(model),
            before_snapshots=make_before_snapshots(),
            rerollout=lambda _: (),
            source_artifact="artifact.json",
            model_name="tiny-model",
            steps=1,
        )
