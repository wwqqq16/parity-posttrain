"""Summarize controlled fixed-sequence parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from parity_posttrain.parity.controlled_summary import (
    build_controlled_parity_summary,
    controlled_parity_summary_to_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Combine controlled parity condition artifacts "
            "into one summary."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Controlled parity JSON artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/controlled_parity_summary.json"
        ),
        help="Path for the combined summary JSON.",
    )

    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    """Load and validate one top-level JSON object."""

    raw = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must contain a top-level JSON object"
        )

    if not all(isinstance(key, str) for key in raw):
        raise ValueError(
            f"{path} contains a non-string top-level key"
        )

    return cast(dict[str, object], raw)


def format_factor(value: float | None) -> str:
    """Format an optional comparison factor."""

    if value is None:
        return "undefined"

    return f"{value:.3f}x"


def main() -> None:
    """Build, print, and write the controlled parity summary."""

    args = parse_args()
    payloads = [
        load_payload(path)
        for path in args.inputs
    ]

    summary = build_controlled_parity_summary(
        payloads
    )
    output_payload = (
        controlled_parity_summary_to_dict(summary)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            output_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    first_row = summary.rows[0]

    print("CONTROLLED PARITY SUMMARY")
    print("Task:", first_row.task_id)
    print("Turn:", first_row.turn_index)
    print("Model:", first_row.model_name)
    print("Conditions:", len(summary.rows))
    print("Tolerance:", first_row.tolerance)

    print()
    print("CONDITIONS")

    for row in summary.rows:
        print(
            row.device,
            row.dtype,
            f"cache={row.use_cache}",
            f"latency_ms={row.latency_ms:.3f}",
            (
                "mean_error="
                f"{row.mean_absolute_error:.9g}"
            ),
            (
                "max_error="
                f"{row.max_absolute_error:.9g}"
            ),
            (
                "tokens_over_tolerance="
                f"{row.tokens_over_tolerance}"
            ),
            f"passed={row.within_tolerance}",
        )

    print()
    print("CACHE COMPARISONS")

    for comparison in summary.comparisons:
        print(
            comparison.device,
            comparison.dtype,
            (
                "no_cache_latency_factor="
                f"{format_factor(
                    comparison.no_cache_latency_factor
                )}"
            ),
            (
                "max_error_reduction_factor="
                f"{format_factor(
                    comparison.max_error_reduction_factor
                )}"
            ),
            (
                "mean_error_reduction_factor="
                f"{format_factor(
                    comparison.mean_error_reduction_factor
                )}"
            ),
        )

    print()
    print("Output:", args.output)


if __name__ == "__main__":
    main()
