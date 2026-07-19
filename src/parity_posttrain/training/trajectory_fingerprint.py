"""Stable fingerprints for generated agent trajectories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence


def fingerprint_generated_token_ids(
    generations: Sequence[Sequence[int]],
) -> str:
    """Hash ordered generated token IDs with turn boundaries."""

    resolved_generations = tuple(
        tuple(token_ids)
        for token_ids in generations
    )

    if not resolved_generations:
        raise ValueError(
            "generations must not be empty"
        )

    canonical_generations: list[list[int]] = []

    for generation_index, token_ids in enumerate(
        resolved_generations
    ):
        if not token_ids:
            raise ValueError(
                "generated token sequences must not be empty"
            )

        canonical_token_ids: list[int] = []

        for token_index, token_id in enumerate(token_ids):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError(
                    "generated token IDs must be "
                    "non-negative integers: "
                    f"generation={generation_index}, "
                    f"token={token_index}"
                )

            canonical_token_ids.append(token_id)

        canonical_generations.append(
            canonical_token_ids
        )

    canonical_bytes = json.dumps(
        canonical_generations,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        canonical_bytes
    ).hexdigest()
