# ruff: noqa: E402
"""Call the enterprise rollout service through real gRPC."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.distributed.proto import (
    control_plane_pb2,
    control_plane_pb2_grpc,
)

_ACTIONS = (
    ("get_order", {"order_id": "ORD-1001"}),
    ("check_refund_policy", {"order_id": "ORD-1001"}),
    ("get_payment_status", {"order_id": "ORD-1001"}),
    ("issue_refund", {"order_id": "ORD-1001"}),
    (
        "respond",
        {"message": "Your refund has been issued."},
    ),
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real gRPC enterprise rollout."
    )
    parser.add_argument(
        "--target",
        default="127.0.0.1:50051",
    )
    return parser.parse_args(argv)


async def run(target: str) -> int:
    async with grpc.aio.insecure_channel(target) as channel:
        stub = control_plane_pb2_grpc.RolloutServiceStub(
            channel
        )
        reset = await stub.Reset(
            control_plane_pb2.ResetRequest(
                case_id="eligible_standard",
            )
        )
        total_return = 0.0
        transitions = []
        for index, (action_type, arguments) in enumerate(
            _ACTIONS
        ):
            transition = await stub.Step(
                control_plane_pb2.StepRequest(
                    episode_id=reset.episode_id,
                    request_id=f"grpc-demo-{index}",
                    action_type=action_type,
                    arguments=arguments,
                )
            )
            transitions.append(transition)
            total_return += transition.reward

        duplicate = await stub.Step(
            control_plane_pb2.StepRequest(
                episode_id=reset.episode_id,
                request_id="grpc-demo-4",
                action_type="respond",
                arguments={
                    "message": (
                        "Your refund has been issued."
                    )
                },
            )
        )

    final = transitions[-1]
    idempotent = (
        duplicate.state_fingerprint
        == final.state_fingerprint
        and duplicate.reward == final.reward
    )

    print("Enterprise gRPC rollout demo")
    print("=" * 72)
    for index, transition in enumerate(transitions):
        print(
            f"{index:<3} "
            f"{transition.action_type:<24} "
            f"reward={transition.reward:>4.2f} "
            f"fingerprint="
            f"{transition.state_fingerprint[:16]}"
        )
    print()
    print("Episode return:", f"{total_return:.2f}")
    print(
        "Final fingerprint:",
        final.state_fingerprint[:16],
    )
    print("Duplicate request idempotent:", idempotent)
    return 0 if idempotent else 1


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args.target))


if __name__ == "__main__":
    raise SystemExit(main())
