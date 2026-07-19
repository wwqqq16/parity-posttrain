"""Run a persistent agent closed-loop training experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.core.task import AgentTask
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)
from parity_posttrain.training.artifact import (
    load_training_examples,
)
from parity_posttrain.training.batch import (
    collate_training_examples,
)
from parity_posttrain.training.closed_loop import (
    ClosedLoopSummary,
    ClosedLoopTaskSnapshot,
)
from parity_posttrain.training.closed_loop_artifact import (
    load_closed_loop_snapshots,
)
from parity_posttrain.training.closed_loop_rerollout import (
    rerollout_agent_tasks,
)
from parity_posttrain.training.closed_loop_runner import (
    run_closed_loop_experiment,
)
from parity_posttrain.training.comparison_runner import (
    select_training_examples,
)
from parity_posttrain.training.objective import (
    PolicyNormalization,
)
from parity_posttrain.training.parameters import (
    prepare_trainable_parameters,
)


def _validate_tasks(
    tasks: Sequence[AgentTask],
) -> tuple[AgentTask, ...]:
    """Validate selected closed-loop tasks."""

    selected = tuple(tasks)

    if not selected:
        raise ValueError(
            "tasks must not be empty"
        )

    task_ids: set[str] = set()

    for task in selected:
        task.validate()

        if task.task_id in task_ids:
            raise ValueError(
                "task IDs must be unique"
            )

        task_ids.add(task.task_id)

    return selected


def run_agent_closed_loop_experiment(
    *,
    backend: HuggingFaceRolloutBackend,
    artifact_path: Path,
    tasks: Sequence[AgentTask],
    trainable_parameter_names: Sequence[str],
    steps: int,
    learning_rate: float,
    normalization: PolicyNormalization = "trajectory",
    clip_epsilon: float = 0.2,
    max_gradient_norm: float = 1.0,
    max_agent_steps: int = 6,
    max_new_tokens: int = 128,
) -> ClosedLoopSummary:
    """Train from an artifact and rerun tasks on the updated backend."""

    selected_tasks = _validate_tasks(tasks)

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
    ):
        raise ValueError(
            "steps must be a positive integer"
        )

    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise ValueError(
            "learning_rate must be positive and finite"
        )

    examples = load_training_examples(
        artifact_path
    )
    task_ids = tuple(
        task.task_id
        for task in selected_tasks
    )
    selected_examples = select_training_examples(
        examples,
        task_ids,
    )

    model_name = selected_examples[0].model_name
    before_snapshots = load_closed_loop_snapshots(
        artifact_path,
        task_ids=task_ids,
    )

    pad_token_id = (
        backend.tokenizer.pad_token_id
    )

    if not isinstance(pad_token_id, int):
        raise ValueError(
            "tokenizer has no integer pad_token_id"
        )

    batch = collate_training_examples(
        selected_examples,
        pad_token_id=pad_token_id,
        device=backend.device,
    )
    runner = AgentRunner(
        backend,
        max_steps=max_agent_steps,
        max_new_tokens=max_new_tokens,
    )
    parameter_selection = (
        prepare_trainable_parameters(
            backend.model,
            trainable_parameter_names,
        )
    )

    try:
        optimizer = torch.optim.SGD(
            parameter_selection.parameters,
            lr=learning_rate,
        )

        def rerollout(
            updated_model: torch.nn.Module,
        ) -> tuple[ClosedLoopTaskSnapshot, ...]:
            if updated_model is not backend.model:
                raise ValueError(
                    "updated model must be backend.model"
                )

            return rerollout_agent_tasks(
                runner=runner,
                tasks=selected_tasks,
            )

        return run_closed_loop_experiment(
            model=backend.model,
            optimizer=optimizer,
            batch=batch,
            before_snapshots=before_snapshots,
            rerollout=rerollout,
            source_artifact=str(artifact_path),
            model_name=model_name,
            steps=steps,
            normalization=normalization,
            clip_epsilon=clip_epsilon,
            max_gradient_norm=max_gradient_norm,
        )
    finally:
        parameter_selection.restore_requires_grad()
