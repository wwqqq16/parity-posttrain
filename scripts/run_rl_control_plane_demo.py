# ruff: noqa: E402
"""Demonstrate deterministic reset, dense rewards, replay, and guarding."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.cases import CASES
from enterprise_eval.models import ActionType, AgentAction
from enterprise_eval.rl_environment import (
    EnterpriseRefundRLEnvironment,
    RLStepResult,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic enterprise RL episode twice "
            "and probe the irreversible-action guard."
        )
    )
    parser.add_argument(
        "--case",
        default="eligible_standard",
        choices=sorted(CASES),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/rl_control_plane/demo.json"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    case = CASES[args.case]
    actions = _refund_actions(case.order_id)

    first = _run_episode(case.case_id, actions)
    second = _run_episode(case.case_id, actions)
    replay_identical = (
        first["fingerprints"]
        == second["fingerprints"]
        and first["rewards"] == second["rewards"]
    )

    unsafe_env = EnterpriseRefundRLEnvironment(
        case,
        enforce_execution_guard=True,
    )
    unsafe_reset = unsafe_env.reset()
    unsafe_transition = unsafe_env.step(
        AgentAction(
            ActionType.ISSUE_REFUND,
            {"order_id": case.order_id},
        )
    )

    payload = {
        "schema_version": (
            "enterprise-rl-control-plane-demo.v1"
        ),
        "case_id": case.case_id,
        "replay_identical": replay_identical,
        "episode": first,
        "unsafe_action_probe": {
            "initial_fingerprint": (
                unsafe_reset.state_fingerprint
            ),
            "transition": _transition_record(
                unsafe_transition
            ),
            "refund_issued": (
                unsafe_env.environment.state.refund_issued
            ),
        },
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

    print("Enterprise RL control-plane demo")
    print("=" * 74)
    print(
        f"{'STEP':<5} {'ACTION':<24} "
        f"{'REWARD':>8} {'DONE':<6} "
        "STATE FINGERPRINT"
    )
    print("-" * 74)
    for index, transition in enumerate(
        first["transitions"]
    ):
        assert isinstance(transition, dict)
        fingerprint = str(
            transition["state_fingerprint"]
        )
        print(
            f"{index:<5} "
            f"{str(transition['action_type']):<24} "
            f"{float(transition['reward']):>8.2f} "
            f"{str(transition['terminated']):<6} "
            f"{fingerprint[:16]}"
        )
    print()
    print(
        "Deterministic replay:",
        replay_identical,
    )
    print(
        "Episode return:",
        f"{float(first['total_return']):.2f}",
    )
    print(
        "Unsafe refund probe:",
        unsafe_transition.info["tool_metadata"][
            "error_type"
        ],
    )
    print(
        "Unsafe probe refund issued:",
        unsafe_env.environment.state.refund_issued,
    )
    print("Artifact:", args.output)
    return 0 if replay_identical else 1


def _run_episode(
    case_id: str,
    actions: tuple[AgentAction, ...],
) -> dict[str, Any]:
    env = EnterpriseRefundRLEnvironment(
        CASES[case_id],
        enforce_execution_guard=True,
    )
    reset = env.reset()
    transitions = [
        env.step(action)
        for action in actions
    ]
    return {
        "initial_observation": reset.observation,
        "fingerprints": [
            reset.state_fingerprint,
            *[
                transition.state_fingerprint
                for transition in transitions
            ],
        ],
        "rewards": [
            transition.reward
            for transition in transitions
        ],
        "total_return": sum(
            transition.reward
            for transition in transitions
        ),
        "transitions": [
            _transition_record(transition)
            for transition in transitions
        ],
    }


def _transition_record(
    transition: RLStepResult,
) -> dict[str, Any]:
    return {
        "action_type": transition.info[
            "action_type"
        ],
        "observation": transition.observation,
        "reward": transition.reward,
        "reward_components": transition.info[
            "reward_components"
        ],
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "state_fingerprint": (
            transition.state_fingerprint
        ),
        "tool_success": transition.info[
            "tool_success"
        ],
        "tool_metadata": transition.info[
            "tool_metadata"
        ],
    }


def _refund_actions(
    order_id: str,
) -> tuple[AgentAction, ...]:
    return (
        AgentAction(
            ActionType.GET_ORDER,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.CHECK_REFUND_POLICY,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.GET_PAYMENT_STATUS,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.ISSUE_REFUND,
            {"order_id": order_id},
        ),
        AgentAction(
            ActionType.RESPOND,
            {"message": "Your refund has been issued."},
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
