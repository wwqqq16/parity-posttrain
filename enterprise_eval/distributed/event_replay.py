"""Replay a control-plane event stream against the core environment."""

from __future__ import annotations

from collections.abc import Sequence

from enterprise_eval.cases import CASES
from enterprise_eval.distributed.events import (
    ControlPlaneEvent,
)
from enterprise_eval.models import (
    ActionType,
    AgentAction,
    Architecture,
)
from enterprise_eval.rl_environment import (
    EnterpriseRefundRLEnvironment,
)


def replay_fingerprint(
    events: Sequence[ControlPlaneEvent],
) -> str:
    """Replay action events and return the final state fingerprint."""

    if not events:
        raise ValueError("event stream must not be empty")
    ordered = sorted(events, key=lambda event: event.sequence)
    _validate_sequence(ordered)

    started = ordered[0]
    if started.event_type != "episode.started":
        raise ValueError(
            "event stream must begin with episode.started"
        )
    case_id = str(started.payload["case_id"])
    architecture = Architecture(
        str(started.payload["architecture"])
    )
    environment = EnterpriseRefundRLEnvironment(
        CASES[case_id],
        enforce_execution_guard=True,
    )
    environment.reset(architecture=architecture)

    for event in ordered:
        if event.event_type != "action.requested":
            continue
        environment.step(
            AgentAction(
                action_type=ActionType(
                    str(event.payload["action_type"])
                ),
                arguments=dict(event.payload["arguments"]),
            )
        )

    return environment.state_fingerprint()


def _validate_sequence(
    events: Sequence[ControlPlaneEvent],
) -> None:
    episode_ids = {
        event.episode_id for event in events
    }
    if len(episode_ids) != 1:
        raise ValueError(
            "replay requires exactly one episode"
        )
    sequences = [event.sequence for event in events]
    if sequences != list(range(len(events))):
        raise ValueError(
            "event stream has missing or duplicate sequence "
            f"values: {sequences}"
        )
