"""Run a real agent and measure parity across all model turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.core.task import AgentTask
from parity_posttrain.evals.trajectory_evaluator import (
    evaluate_trajectory,
)
from parity_posttrain.parity.trajectory_parity import (
    rescore_agent_generations,
)
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run multi-turn agent parity analysis."
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
        "--output",
        type=Path,
        default=Path("artifacts/agent_parity.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run an agent, rescore every turn, and save a report."""

    args = parse_args()

    task = AgentTask(
        task_id="agent_parity",
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

    parity = rescore_agent_generations(
        model=backend.model,
        tokenizer=backend.tokenizer,
        device=backend.device,
        dtype=str(backend.dtype),
        generations=run_result.generations,
        tolerance=args.tolerance,
    )

    payload = {
        "run": run_result.to_dict(),
        "evaluation": evaluation.to_dict(),
        "parity": parity.to_dict(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Agent status:", run_result.status)
    print("Answer correct:", evaluation.answer_correct)
    print("Reward:", evaluation.reward)
    print("Model turns:", parity.generation_count)
    print("Total generated tokens:", parity.total_token_count)
    print("Tolerance:", parity.tolerance)
    print(
        "Mean absolute error:",
        parity.mean_absolute_error,
    )
    print(
        "Max absolute error:",
        parity.max_absolute_error,
    )
    print(
        "P95 absolute error:",
        parity.p95_absolute_error,
    )
    print(
        "Tokens over tolerance:",
        parity.tokens_over_tolerance,
    )
    print("Within tolerance:", parity.within_tolerance)
    print(
        "Total rescore latency ms:",
        parity.total_rescore_latency_ms,
    )

    for turn in parity.turn_reports:
        print()
        print(
            f"Turn {turn.turn_index}:",
            repr(turn.generated_text),
        )
        print(
            "Prompt tokens:",
            turn.prompt_token_count,
        )
        print(
            "Generated tokens:",
            turn.parity.token_count,
        )
        print(
            "Max error:",
            turn.parity.max_absolute_error,
        )
        print(
            "Within tolerance:",
            turn.parity.within_tolerance,
        )

        for token in turn.parity.token_records:
            print(
                f"  {token.index:02d}",
                f"id={token.token_id}",
                f"token={token.token_text!r}",
                f"rollout={token.rollout_logprob:.8f}",
                f"trainer={token.trainer_logprob:.8f}",
                f"error={token.absolute_error:.8f}",
            )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
