# ruff: noqa: E402
"""Generate, validate, and evaluate a controlled enterprise failure surface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enterprise_eval.failure_surface import (
    build_failure_surface,
    failure_surface_markdown,
)
from enterprise_eval.models import Architecture
from enterprise_eval.runner import BenchmarkRow, run_task_suite
from enterprise_eval.task_factory import (
    TaskFactoryConfig,
    generate_failure_surface_cases,
    validate_task_suite,
)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure enterprise-agent success and recovery across "
            "difficulty, failure type, and injection position."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=("single", "planner-critic", "both"),
        default="both",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--variants-per-cell",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/failure_surface/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/failure_surface/summary.json"
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(argv)
    config = TaskFactoryConfig(
        seed=args.seed,
        variants_per_cell=args.variants_per_cell,
    )
    cases = generate_failure_surface_cases(config)
    validation = validate_task_suite(
        cases,
        config=config,
    )

    architectures = (
        (Architecture.SINGLE, Architecture.PLANNER_CRITIC)
        if args.architecture == "both"
        else (Architecture(args.architecture),)
    )
    rows: list[BenchmarkRow] = []
    for architecture in architectures:
        rows.extend(
            run_task_suite(
                cases,
                args.output_dir,
                architecture=architecture,
            )
        )

    cells = build_failure_surface(rows)
    payload = {
        "schema_version": "enterprise-failure-surface.v1",
        "factory": {
            "seed": config.seed,
            "variants_per_cell": config.variants_per_cell,
            "difficulties": [
                difficulty.value
                for difficulty in config.difficulties
            ],
            "failure_profiles": [
                profile.value
                for profile in config.failure_profiles
            ],
            "injection_steps": list(config.injection_steps),
        },
        "validation": asdict(validation),
        "architectures": [
            architecture.value
            for architecture in architectures
        ],
        "run_count": len(rows),
        "cells": [
            cell.to_dict()
            for cell in cells
        ],
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

    print(
        "Validated task factory:",
        f"{validation.case_count} cases,",
        f"{validation.coverage_cell_count} coverage cells,",
        (
            "oracle solvability "
            f"{validation.oracle_success_rate:.1%}"
        ),
    )
    print(
        f"Evaluated {len(rows)} runs across "
        f"{len(architectures)} architecture(s)."
    )
    print()
    print(failure_surface_markdown(cells))
    print()
    print("Machine-readable artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
