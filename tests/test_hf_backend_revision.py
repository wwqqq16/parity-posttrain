"""Tests for pinned Hugging Face model revisions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import parity_posttrain.rollout.hf_backend as backend_module
from parity_posttrain.rollout.hf_backend import (
    HuggingFaceRolloutBackend,
)


class FakeLoadedModel:
    """Minimal causal model for backend construction."""

    def __init__(
        self,
        calls: dict[str, object],
    ) -> None:
        self.calls = calls
        self.config = SimpleNamespace(
            _commit_hash="c" * 40,
        )
        self.generation_config = SimpleNamespace(
            _commit_hash="d" * 40,
        )

    def to(
        self,
        device: torch.device,
    ) -> FakeLoadedModel:
        self.calls["model_device"] = device
        return self

    def eval(self) -> FakeLoadedModel:
        self.calls["model_eval"] = True
        return self


def test_backend_passes_revision_to_both_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def load_tokenizer(
        model_name: str,
        **kwargs: object,
    ) -> object:
        calls["tokenizer_model_name"] = model_name
        calls["tokenizer_kwargs"] = kwargs

        return SimpleNamespace(
            pad_token_id=0,
            eos_token="<eos>",
        )

    def load_model(
        model_name: str,
        **kwargs: object,
    ) -> FakeLoadedModel:
        calls["model_name"] = model_name
        calls["model_kwargs"] = kwargs
        return FakeLoadedModel(calls)

    monkeypatch.setattr(
        backend_module,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=load_tokenizer,
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "AutoModelForCausalLM",
        SimpleNamespace(
            from_pretrained=load_model,
        ),
    )

    backend = HuggingFaceRolloutBackend(
        "test-model",
        torch.device("cpu"),
        revision="revision-123",
    )

    assert backend.model_name == "test-model"
    assert backend.model_revision == "revision-123"
    assert backend.resolved_model_revision == (
        "c" * 40
    )
    assert calls["tokenizer_kwargs"] == {
        "revision": "revision-123",
    }
    assert calls["model_kwargs"] == {
        "dtype": torch.float32,
        "revision": "revision-123",
    }
    assert calls["model_device"] == torch.device(
        "cpu"
    )
    assert calls["model_eval"] is True


def test_backend_rejects_empty_revision() -> None:
    with pytest.raises(
        ValueError,
        match="revision must not be empty",
    ):
        HuggingFaceRolloutBackend(
            "test-model",
            torch.device("cpu"),
            revision=" ",
        )
