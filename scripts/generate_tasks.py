"""Generate the deterministic sample agent task dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parity_posttrain.data.task_factory import build_sample_tasks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic agent benchmark tasks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample_tasks.jsonl"),
        help="Destination JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate tasks and write them as JSON Lines."""

    args = parse_args()
    tasks = build_sample_tasks()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task.to_dict(), sort_keys=True))
            file.write("\n")

    print(f"Generated {len(tasks)} tasks")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
