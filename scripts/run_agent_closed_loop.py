"""Run a persistent agent closed-loop training experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from parity_posttrain.core.task import AgentTask
from parity_posttrain.data.task_factory import (
    build_sample_tasks,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)
from parity_posttrain.training import (
    PolicyNormalization,
    closed_loop_summary_to_dict,
    load_training_examples,
    run_agent_closed_loop_experiment,
    select_training_examples,
)

_DEFAULT_TASK_IDS = [
    "catalog_004",
    "shopping_004",
]


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train on selected agent trajectories and "
            "rerun the same tasks with the updated model."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "artifacts/agent_benchmark.json"
        ),
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=_DEFAULT_TASK_IDS,
    )
    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "mps",
            "cuda",
        ],
        default="cpu",
    )
    parser.add_argument(
        "--trainable-parameters",
        nargs="+",
        default=["model.norm.weight"],
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "token",
            "sequence",
            "trajectory",
        ],
        default="trajectory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/agent_closed_loop.json"
        ),
    )

    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    """Validate and construct a requested device."""

    if (
        name == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is not available"
        )

    if (
        name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return torch.device(name)


def select_tasks(
    task_ids: Sequence[str],
) -> tuple[AgentTask, ...]:
    """Select deterministic tasks in requested order."""

    requested = tuple(task_ids)

    if not requested:
        raise ValueError(
            "task_ids must not be empty"
        )

    if any(
        not isinstance(task_id, str)
        or not task_id.strip()
        for task_id in requested
    ):
        raise ValueError(
            "task_ids must contain non-empty strings"
        )

    if len(set(requested)) != len(requested):
        raise ValueError(
            "task_ids must be unique"
        )

    tasks_by_id = {
        task.task_id: task
        for task in build_sample_tasks()
    }
    missing = tuple(
        task_id
        for task_id in requested
        if task_id not in tasks_by_id
    )

    if missing:
        raise ValueError(
            "tasks were not found: "
            + ", ".join(missing)
        )

    return tuple(
        tasks_by_id[task_id]
        for task_id in requested
    )


def resolve_model_name(
    *,
    artifact_path: Path,
    task_ids: Sequence[str],
) -> str:
    """Read the single model name used by selected examples."""

    examples = load_training_examples(
        artifact_path
    )
    selected = select_training_examples(
        examples,
        task_ids,
    )

    return selected[0].model_name


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the experiment and write its JSON artifact."""

    args = parse_args(argv)
    task_ids = tuple(args.task_ids)
    tasks = select_tasks(task_ids)
    model_name = resolve_model_name(
        artifact_path=args.artifact,
        task_ids=task_ids,
    )
    device = resolve_device(args.device)

    backend = HuggingFaceRolloutBackend(
        model_name=model_name,
        device=device,
    )
    normalization = cast(
        PolicyNormalization,
        args.normalization,
    )

    summary = run_agent_closed_loop_experiment(
        backend=backend,
        artifact_path=args.artifact,
        tasks=tasks,
        trainable_parameter_names=(
            args.trainable_parameters
        ),
        steps=args.steps,
        learning_rate=args.learning_rate,
        normalization=normalization,
        clip_epsilon=args.clip_epsilon,
        max_gradient_norm=(
            args.max_gradient_norm
        ),
        max_agent_steps=args.max_agent_steps,
        max_new_tokens=args.max_new_tokens,
    )

    payload = closed_loop_summary_to_dict(
        summary
    )
    payload["experiment"] = {
        "device": str(backend.device),
        "dtype": str(backend.dtype),
        "task_ids": list(task_ids),
        "trainable_parameter_names": list(
            args.trainable_parameters
        ),
        "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "max_gradient_norm": (
            args.max_gradient_norm
        ),
        "max_agent_steps": args.max_agent_steps,
        "max_new_tokens": args.max_new_tokens,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("AGENT CLOSED-LOOP EXPERIMENT")
    print("=" * 72)
    print("Model:", summary.model_name)
    print("Device:", backend.device)
    print("Dtype:", backend.dtype)
    print("Tasks:", summary.before.task_count)
    print("Normalization:", summary.normalization)
    print("Optimizer steps:", summary.optimizer_steps)
    print(
        "Before total reward:",
        summary.before.total_reward,
    )
    print(
        "After total reward:",
        summary.after.total_reward,
    )
    print(
        "Total reward delta:",
        summary.reward_delta,
    )
    print(
        "Before answer accuracy:",
        summary.before.answer_accuracy,
    )
    print(
        "After answer accuracy:",
        summary.after.answer_accuracy,
    )
    print()

    for step in summary.training_steps:
        print(
            f"Step {step.step_index}:",
            f"loss={step.loss:.12e}",
            (
                "gradient_norm="
                f"{step.gradient_norm:.12e}"
            ),
            f"mean_ratio={step.mean_ratio:.12e}",
            (
                "approximate_kl="
                f"{step.approximate_kl:.12e}"
            ),
            (
                "parameter_delta="
                f"{step.parameter_delta:.12e}"
            ),
        )

    print()

    for task in summary.tasks:
        print(
            f"{task.task_id}:",
            (
                f"reward={task.before.reward}"
                f"->{task.after.reward}"
            ),
            (
                f"status={task.before.status}"
                f"->{task.after.status}"
            ),
            (
                "answer_correct="
                f"{task.before.answer_correct}"
                f"->{task.after.answer_correct}"
            ),
            (
                "trajectory_changed="
                f"{task.trajectory_changed}"
            ),
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
