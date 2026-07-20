"""Run and summarize a controlled-parity matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from parity_posttrain.parity.controlled_summary import (
    build_controlled_parity_summary,
    controlled_parity_summary_to_dict,
)
from parity_posttrain.parity.matrix import (
    build_controlled_parity_matrix,
)
from parity_posttrain.parity.matrix_runner import (
    run_controlled_parity_matrix,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run controlled parity across device/cache "
            "conditions and write one summary artifact."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help=(
            "Agent benchmark artifact containing the "
            "stored generated token sequence."
        ),
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task containing the generation to replay.",
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=0,
        help="Generation turn index to replay.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum accepted absolute logprob error.",
    )
    parser.add_argument(
        "--include-mps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include MPS cached and uncached "
            "conditions."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/parity_matrix"),
        help="Directory for individual condition artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/controlled_parity_matrix.json"
        ),
        help="Path for the unified matrix summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed shared by all matrix runs.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Optional Hugging Face branch, tag, "
            "or commit shared by all matrix runs."
        ),
    )
    parser.add_argument(
        "--known-mismatch",
        action="append",
        default=[],
        metavar="CONDITION_SLUG",
        help=(
            "Condition allowed to fail without causing "
            "a regression exit code. May be repeated."
        ),
    )

    return parser.parse_args(argv)


def load_controlled_payload(
    path: Path,
) -> dict[str, Any]:
    """Load one controlled-parity condition artifact."""

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "controlled-parity artifact must "
            "contain a JSON object"
        )

    return cast(dict[str, Any], payload)


def validate_matrix_provenance(
    payloads: Sequence[dict[str, Any]],
    *,
    seed: int,
    requested_model_revision: str | None,
) -> dict[str, Any]:
    """Validate shared provenance across matrix runs."""

    if not payloads:
        raise ValueError(
            "at least one condition artifact is required"
        )

    reference_value = payloads[0].get(
        "provenance"
    )

    if not isinstance(reference_value, dict):
        raise ValueError(
            "condition artifact 0 must contain "
            "a provenance object"
        )

    reference = cast(
        dict[str, Any],
        reference_value,
    )

    required_fields = (
        "seed",
        "requested_model_revision",
        "resolved_model_revision",
    )

    for field in required_fields:
        if field not in reference:
            raise ValueError(
                "condition provenance is missing "
                f"required field: {field}"
            )

    for index, payload in enumerate(
        payloads[1:],
        start=1,
    ):
        provenance_value = payload.get(
            "provenance"
        )

        if not isinstance(
            provenance_value,
            dict,
        ):
            raise ValueError(
                f"condition artifact {index} must "
                "contain a provenance object"
            )

        if provenance_value != reference:
            raise ValueError(
                "all condition artifacts must use "
                "identical provenance"
            )

    if reference["seed"] != seed:
        raise ValueError(
            "condition provenance seed does not "
            "match matrix seed"
        )

    if (
        reference["requested_model_revision"]
        != requested_model_revision
    ):
        raise ValueError(
            "condition requested model revision "
            "does not match matrix request"
        )

    resolved_model_revision = reference[
        "resolved_model_revision"
    ]

    if (
        resolved_model_revision is not None
        and not isinstance(
            resolved_model_revision,
            str,
        )
    ):
        raise ValueError(
            "resolved_model_revision must be "
            "a string or null"
        )

    return reference


def validate_known_mismatch_slugs(
    declared_slugs: Sequence[str],
    *,
    available_slugs: Sequence[str],
) -> list[str]:
    """Validate and deduplicate known mismatch slugs."""

    unique_declared_slugs = list(
        dict.fromkeys(declared_slugs)
    )
    available_slug_set = set(available_slugs)
    unknown_slugs = [
        slug
        for slug in unique_declared_slugs
        if slug not in available_slug_set
    ]

    if unknown_slugs:
        joined_slugs = ", ".join(unknown_slugs)
        raise ValueError(
            "unknown known-mismatch condition "
            f"slug(s): {joined_slugs}"
        )

    return unique_declared_slugs


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the matrix and write a unified summary."""

    args = parse_args(argv)

    planned_conditions = (
        build_controlled_parity_matrix(
            include_mps=args.include_mps,
        )
    )
    declared_known_mismatch_slugs = (
        validate_known_mismatch_slugs(
            args.known_mismatch,
            available_slugs=[
                condition.slug
                for condition in planned_conditions
            ],
        )
    )

    runs = run_controlled_parity_matrix(
        artifact=args.artifact,
        task_id=args.task_id,
        turn_index=args.turn_index,
        tolerance=args.tolerance,
        output_directory=args.output_directory,
        seed=args.seed,
        model_revision=args.model_revision,
        include_mps=args.include_mps,
    )

    condition_payloads = [
        load_controlled_payload(run.output_path)
        for run in runs
    ]

    matrix_provenance = validate_matrix_provenance(
        condition_payloads,
        seed=args.seed,
        requested_model_revision=(
            args.model_revision
        ),
    )
    resolved_model_revision = (
        matrix_provenance[
            "resolved_model_revision"
        ]
    )

    summary = build_controlled_parity_summary(
        condition_payloads
    )

    payload = controlled_parity_summary_to_dict(
        summary
    )

    failed_condition_slugs = [
        run.condition.slug
        for run, row in zip(
            runs,
            summary.rows,
            strict=True,
        )
        if not row.within_tolerance
    ]

    known_mismatch_slugs = set(
        declared_known_mismatch_slugs
    )
    known_failed_condition_slugs = [
        slug
        for slug in declared_known_mismatch_slugs
        if slug in failed_condition_slugs
    ]
    resolved_known_mismatch_slugs = [
        slug
        for slug in declared_known_mismatch_slugs
        if slug not in failed_condition_slugs
    ]
    unexpected_failed_condition_slugs = [
        slug
        for slug in failed_condition_slugs
        if slug not in known_mismatch_slugs
    ]

    overall_passed = not failed_condition_slugs
    regression_passed = (
        not unexpected_failed_condition_slugs
    )

    payload["schema_version"] = 1
    payload["overall_passed"] = overall_passed
    payload["regression_passed"] = regression_passed
    payload["failed_condition_slugs"] = (
        failed_condition_slugs
    )
    payload["known_mismatch_slugs"] = (
        declared_known_mismatch_slugs
    )
    payload["known_failed_condition_slugs"] = (
        known_failed_condition_slugs
    )
    payload["resolved_known_mismatch_slugs"] = (
        resolved_known_mismatch_slugs
    )
    payload["unexpected_failed_condition_slugs"] = (
        unexpected_failed_condition_slugs
    )
    payload["matrix"] = {
        "seed": args.seed,
        "requested_model_revision": (
            args.model_revision
        ),
        "resolved_model_revision": (
            resolved_model_revision
        ),
        "include_mps": args.include_mps,
        "condition_count": len(runs),
        "conditions": [
            {
                "slug": run.condition.slug,
                "device": run.condition.device,
                "use_cache": (
                    run.condition.use_cache
                ),
                "artifact": str(run.output_path),
                "command": list(run.command),
            }
            for run in runs
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

    print("Task:", args.task_id)
    print("Turn:", args.turn_index)
    print("Tolerance:", args.tolerance)
    print("Conditions:", len(runs))
    print()

    print("Parity matrix:")

    for row in summary.rows:
        print(
            f"- device={row.device}",
            f"dtype={row.dtype}",
            f"cache={row.use_cache}",
            f"latency_ms={row.latency_ms:.6f}",
            (
                "mean_error="
                f"{row.mean_absolute_error:.12e}"
            ),
            (
                "max_error="
                f"{row.max_absolute_error:.12e}"
            ),
            (
                "tokens_over_tolerance="
                f"{row.tokens_over_tolerance}"
            ),
            f"passed={row.within_tolerance}",
        )

    print()
    print("Matrix status:")
    print("Overall passed:", overall_passed)
    print("Regression passed:", regression_passed)
    print(
        "Known mismatches:",
        declared_known_mismatch_slugs,
    )
    print(
        "Known failures:",
        known_failed_condition_slugs,
    )
    print(
        "Resolved known mismatches:",
        resolved_known_mismatch_slugs,
    )
    print(
        "Unexpected failures:",
        unexpected_failed_condition_slugs,
    )
    print("Seed:", args.seed)
    print(
        "Requested model revision:",
        args.model_revision,
    )
    print(
        "Resolved model revision:",
        resolved_model_revision,
    )
    print()
    print("Output:", args.output)

    return 0 if regression_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
