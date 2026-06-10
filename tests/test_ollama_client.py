"""Tests for the object-based Ollama response helpers."""

from __future__ import annotations

from ollama import GenerateResponse, ListResponse

from investment_monitor.analysis.ollama_client import (
    has_model,
    model_matches,
    model_names,
    response_text,
)


def _list(*names: str) -> ListResponse:
    return ListResponse(models=[ListResponse.Model(model=n) for n in names])


class TestModelNames:
    def test_extracts_tags_from_object_response(self):
        resp = _list("qwen2.5:7b", "phi3:mini")
        assert model_names(resp) == ["qwen2.5:7b", "phi3:mini"]

    def test_empty_when_no_models(self):
        assert model_names(_list()) == []


class TestModelMatches:
    def test_exact_match(self):
        assert model_matches(["qwen2.5:7b"], "qwen2.5:7b") is True

    def test_different_size_same_family_does_not_match(self):
        # The bug the doctor report surfaced: 7b must not satisfy 32b.
        assert model_matches(["qwen2.5:7b"], "qwen2.5:32b") is False

    def test_unpinned_tag_matches_any_size(self):
        assert model_matches(["qwen2.5:7b"], "qwen2.5") is True

    def test_latest_satisfied_by_bare_install(self):
        assert model_matches(["phi3"], "phi3:latest") is True

    def test_no_match(self):
        assert model_matches(["llama2:7b", "phi3:mini"], "nonexistent:model") is False


class TestHasModel:
    def test_has_model_uses_object_response(self):
        resp = _list("llama2:7b", "phi3:mini")
        assert has_model(resp, "phi3:mini") is True
        assert has_model(resp, "qwen2.5:32b") is False


class TestResponseText:
    def test_strips_text(self):
        assert response_text(GenerateResponse(response="  hi  ")) == "hi"

    def test_empty_when_missing(self):
        # An object without a populated response yields "".
        assert response_text(GenerateResponse(response="")) == ""
