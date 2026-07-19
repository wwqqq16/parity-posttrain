"""Rerun agent tasks and build closed-loop snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from parity_posttrain.agent.runner import (
    AgentRunResult,
)
from parity_posttrain.core.task import AgentTask
from parity_posttrain.evals.trajectory_evaluator import (
    evaluate_trajectory,
)
from parity_posttrain.training.closed_loop import (
    ClosedLoopTaskSnapshot,
)
from parity_posttrain.training.trajectory_fingerprint import (
    fingerprint_generated_token_ids,
)


class AgentTaskRunner(Protocol):
    """Interface required for agent-task rerollout."""

    def run(
        self,
        task: AgentTask,
    ) -> AgentRunResult:
        """Run one agent task."""


def _task_category(task: AgentTask) -> str:
    """Read and validate one task category."""

    category = task.metadata.get("category")

    if (
        not isinstance(category, str)
        or not category.strip()
    ):
        raise ValueError(
            f"task {task.task_id} has no valid category"
        )

    return category


def _validate_tasks(
    tasks: Sequence[AgentTask],
) -> tuple[AgentTask, ...]:
    """Validate selected rerollout tasks."""

    resolved = tuple(tasks)

    if not resolved:
        raise ValueError(
            "tasks must not be empty"
        )

    task_ids: set[str] = set()

    for task in resolved:
        task.validate()
        _task_category(task)

        if task.task_id in task_ids:
            raise ValueError(
                "task IDs must be unique"
            )

        task_ids.add(task.task_id)

    return resolved


def rerollout_agent_tasks(
    *,
    runner: AgentTaskRunner,
    tasks: Sequence[AgentTask],
) -> tuple[ClosedLoopTaskSnapshot, ...]:
    """Rerun agent tasks and convert outcomes to snapshots."""

    selected_tasks = _validate_tasks(tasks)
    snapshots: list[ClosedLoopTaskSnapshot] = []

    for task in selected_tasks:
        run_result = runner.run(task)

        if not run_result.generations:
            raise ValueError(
                f"task {task.task_id} produced no generations"
            )

        evaluation = evaluate_trajectory(
            task,
            run_result.trajectory,
        )

        generated_token_count = sum(
            len(generation.generated_token_ids)
            for generation in run_result.generations
        )

        snapshot = ClosedLoopTaskSnapshot(
            task_id=task.task_id,
            category=_task_category(task),
            status=run_result.status,
            reward=evaluation.reward,
            answer_correct=(
                evaluation.answer_correct
            ),
            generation_count=len(
                run_result.generations
            ),
            generated_token_count=(
                generated_token_count
            ),
            trajectory_fingerprint=(
                fingerprint_generated_token_ids(
                    tuple(
                        tuple(
                            generation.generated_token_ids
                        )
                        for generation
                        in run_result.generations
                    )
                )
            ),
        )
        snapshot.validate()
        snapshots.append(snapshot)

    return tuple(snapshots)
