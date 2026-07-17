"""Run a representative multi-category agent benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.benchmarks.agent_benchmark import (
    AgentBenchmarkRecord,
    build_benchmark_summary,
)
from parity_posttrain.core.task import AgentTask
from parity_posttrain.data.task_factory import build_sample_tasks
from parity_posttrain.evals.trajectory_evaluator import (
    evaluate_trajectory,
)
from parity_posttrain.parity.trajectory_parity import (
    rescore_agent_generations,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)

_SMOKE_TASK_IDS = [
    "calculator_001",
    "catalog_004",
    "currency_001",
    "shopping_004",
    "basket_001",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a five-category agent rollout and parity benchmark."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--max-steps",
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
        default=Path("artifacts/agent_benchmark.json"),
    )
    return parser.parse_args()


def select_smoke_tasks() -> list[AgentTask]:
    """Select one representative task from each category."""

    tasks_by_id = {
        task.task_id: task
        for task in build_sample_tasks()
    }

    missing = [
        task_id
        for task_id in _SMOKE_TASK_IDS
        if task_id not in tasks_by_id
    ]

    if missing:
        names = ", ".join(missing)
        raise ValueError(f"benchmark tasks not found: {names}")

    return [
        tasks_by_id[task_id]
        for task_id in _SMOKE_TASK_IDS
    ]


def get_task_category(task: AgentTask) -> str:
    """Read and validate a task category."""

    category = task.metadata.get("category")

    if not isinstance(category, str) or not category.strip():
        raise ValueError(
            f"task {task.task_id} has no valid category"
        )

    return category


def main() -> None:
    """Run all selected tasks and save aggregate metrics."""

    args = parse_args()
    tasks = select_smoke_tasks()

    backend = HuggingFaceRolloutBackend(args.model)
    runner = AgentRunner(
        backend,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )

    records: list[AgentBenchmarkRecord] = []
    task_results: list[dict[str, object]] = []

    for index, task in enumerate(tasks, start=1):
        category = get_task_category(task)

        print()
        print(
            f"[{index}/{len(tasks)}]",
            task.task_id,
            f"category={category}",
        )
        print("Prompt:", task.prompt)

        run_result = runner.run(task)
        evaluation = evaluate_trajectory(
            task,
            run_result.trajectory,
        )

        parity = rescore_agent_generations(
            model=backend.model,
            tokenizer=backend.tokenizer,
            device=backend.device,
            dtype=str(backend.dtype),
            generations=run_result.generations,
            tolerance=args.tolerance,
        )

        generated_token_count = sum(
            len(generation.generated_token_ids)
            for generation in run_result.generations
        )

        record = AgentBenchmarkRecord(
            task_id=task.task_id,
            category=category,
            status=run_result.status,
            answer_correct=evaluation.answer_correct,
            reward=evaluation.reward,
            tool_coverage=evaluation.tool_coverage,
            used_tools=evaluation.used_tools,
            missing_tools=evaluation.missing_tools,
            unexpected_tools=evaluation.unexpected_tools,
            latency_ms=run_result.trajectory.latency_ms,
            generation_count=len(run_result.generations),
            generated_token_count=generated_token_count,
            parity_within_tolerance=(
                parity.within_tolerance
            ),
            parity_mean_absolute_error=(
                parity.mean_absolute_error
            ),
            parity_max_absolute_error=(
                parity.max_absolute_error
            ),
            parity_tokens_over_tolerance=(
                parity.tokens_over_tolerance
            ),
            error=run_result.error,
        )
        record.validate()
        records.append(record)

        task_results.append(
            {
                "task": task.to_dict(),
                "run": run_result.to_dict(),
                "evaluation": evaluation.to_dict(),
                "parity": parity.to_dict(),
                "benchmark_record": record.to_dict(),
            }
        )

        print("Status:", record.status)
        print("Answer correct:", record.answer_correct)
        print("Reward:", record.reward)
        print("Used tools:", record.used_tools)
        print("Missing tools:", record.missing_tools)
        print("Generations:", record.generation_count)
        print("Generated tokens:", record.generated_token_count)
        print("Latency ms:", record.latency_ms)
        print(
            "Parity max error:",
            record.parity_max_absolute_error,
        )
        print(
            "Parity passed:",
            record.parity_within_tolerance,
        )

    summary = build_benchmark_summary(records)

    payload = {
        "model": args.model,
        "tolerance": args.tolerance,
        "task_ids": _SMOKE_TASK_IDS,
        "summary": summary.to_dict(),
        "tasks": task_results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    print("Tasks:", summary.task_count)
    print("Completed:", summary.completed_count)
    print("Protocol errors:", summary.protocol_error_count)
    print("Max-step stops:", summary.max_steps_count)
    print("Correct answers:", summary.answer_correct_count)
    print("Completion rate:", summary.completion_rate)
    print("Answer accuracy:", summary.answer_accuracy)
    print("Mean reward:", summary.mean_reward)
    print("Mean tool coverage:", summary.mean_tool_coverage)
    print("Mean latency ms:", summary.mean_latency_ms)
    print(
        "Total generated tokens:",
        summary.total_generated_tokens,
    )
    print("Parity pass rate:", summary.parity_pass_rate)
    print(
        "Mean parity absolute error:",
        summary.mean_parity_absolute_error,
    )
    print(
        "Max parity absolute error:",
        summary.max_parity_absolute_error,
    )
    print(
        "Tokens over tolerance:",
        summary.total_tokens_over_tolerance,
    )

    print()
    print("CATEGORY BREAKDOWN")

    for category, category_summary in (
        summary.by_category.items()
    ):
        print(
            category,
            f"completion={category_summary.completion_rate:.3f}",
            f"accuracy={category_summary.answer_accuracy:.3f}",
            f"reward={category_summary.mean_reward:.3f}",
            f"tool_coverage={category_summary.mean_tool_coverage:.3f}",
            f"parity={category_summary.parity_pass_rate:.3f}",
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
