"""Tests for the committed controlled-parity fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "basket_001_turn_0.json"
)


def test_basket_parity_fixture_contract() -> None:
    payload = cast(
        dict[str, Any],
        json.loads(
            FIXTURE_PATH.read_text(
                encoding="utf-8"
            )
        ),
    )

    tasks = cast(
        list[dict[str, Any]],
        payload["tasks"],
    )

    assert len(tasks) == 1

    task_result = tasks[0]
    assert (
        task_result["benchmark_record"]["task_id"]
        == "basket_001"
    )

    generations = task_result["run"]["generations"]
    assert len(generations) == 1

    generation = generations[0]

    assert generation["model_name"] == (
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert len(generation["prompt_token_ids"]) == 562
    assert len(generation["generated_token_ids"]) == 63
    assert isinstance(
        generation["generated_text"],
        str,
    )
    assert generation["generated_text"]
