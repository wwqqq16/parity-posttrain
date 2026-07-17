"""Run one real multi-turn tool-use agent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.core.task import AgentTask
from parity_posttrain.evals.trajectory_evaluator import (
    evaluate_trajectory,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real tool-use agent smoke test."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agent_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task = AgentTask(
        task_id="agent_smoke",
        prompt=(
            "Use the calculator tool to evaluate 17 * 6. "
            "Do not solve it mentally. Return the tool result."
        ),
        expected_answer="102",
        required_tools=["calculator"],
    )

    backend = HuggingFaceRolloutBackend(args.model)
    runner = AgentRunner(
        backend,
        max_steps=3,
        max_new_tokens=96,
    )

    run_result = runner.run(task)
    evaluation = evaluate_trajectory(
        task,
        run_result.trajectory,
    )

    payload = {
        "run": run_result.to_dict(),
        "evaluation": evaluation.to_dict(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Status:", run_result.status)
    print("Error:", run_result.error)
    print("Generations:", len(run_result.generations))
    print("Total latency ms:", run_result.trajectory.latency_ms)
    print("Answer correct:", evaluation.answer_correct)
    print("Used tools:", evaluation.used_tools)
    print("Reward:", evaluation.reward)
    print()

    for index, message in enumerate(
        run_result.trajectory.messages
    ):
        print(
            f"{index:02d}",
            f"role={message.role}",
            f"name={message.name!r}",
            f"content={message.content!r}",
            f"tool_call={message.tool_call!r}",
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
