"""Tests for Phase 5 — model routing + semantic memory.

All pure/DB-level (no Ollama): the embeddings client is faked so cosine search,
dedup, and the SemanticMemory integration are deterministic.
"""

from __future__ import annotations

from investment_monitor.analysis.model_router import ModelRouter
from investment_monitor.analysis.semantic_memory import SemanticMemory
from investment_monitor.config import Settings
from investment_monitor.storage import (
    cosine_similarity,
    get_session,
    init_db,
    is_duplicate,
    save_embedding,
    search_similar,
)


# --------------------------------------------------------------------------- #
# Model router
# --------------------------------------------------------------------------- #
def test_model_router_resolves_roles_and_falls_back():
    settings = Settings(
        ollama_model="phi3:mini",
        model_roles={"synthesis": "qwen2.5:14b", "embedding": "nomic-embed-text"},
    )
    router = ModelRouter(settings)
    assert router.get_model("synthesis") == "qwen2.5:14b"
    assert router.get_model("embedding") == "nomic-embed-text"
    # Unknown / unconfigured role -> default model.
    assert router.get_model("triage") == "phi3:mini"
    assert router.get_model("nonexistent") == "phi3:mini"


# --------------------------------------------------------------------------- #
# Cosine
# --------------------------------------------------------------------------- #
def test_cosine_similarity():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1]) == 0.0          # degenerate
    assert cosine_similarity([0, 0], [0, 0]) == 0.0   # zero vector
    assert cosine_similarity([1, 2, 3], [1, 2]) == 0.0  # length mismatch


# --------------------------------------------------------------------------- #
# Store + search
# --------------------------------------------------------------------------- #
def test_search_similar_ranks_and_filters(tmp_path):
    db = tmp_path / "m.db"
    init_db(db)
    with get_session() as s:
        save_embedding(s, kind="thesis", text="a", vector=[1.0, 0.0, 0.0], symbol="AAA")
        save_embedding(s, kind="thesis", text="c", vector=[0.9, 0.1, 0.0], symbol="CCC")
        save_embedding(s, kind="thesis", text="b", vector=[0.0, 1.0, 0.0], symbol="BBB")
        save_embedding(s, kind="news", text="n", vector=[1.0, 0.0, 0.0], symbol="AAA")
    with get_session() as s:
        results = search_similar(s, [1.0, 0.0, 0.0], kind="thesis", top_k=2)
        texts = [row.text for row, _ in results]
        assert texts == ["a", "c"]  # most similar first, news excluded by kind filter
        assert results[0][1] == 1.0  # exact match scores 1.0


def test_is_duplicate_threshold(tmp_path):
    db = tmp_path / "m.db"
    init_db(db)
    with get_session() as s:
        save_embedding(s, kind="thesis", text="a", vector=[1.0, 0.0, 0.0])
    with get_session() as s:
        assert is_duplicate(s, [1.0, 0.0, 0.0], kind="thesis", threshold=0.95) is True
        assert is_duplicate(s, [0.0, 1.0, 0.0], kind="thesis", threshold=0.95) is False


# --------------------------------------------------------------------------- #
# SemanticMemory integration (fake embeddings client)
# --------------------------------------------------------------------------- #
class _FakeEmb:
    def __init__(self, vectors, available=True):
        self._v = vectors
        self._available = available

    def is_available(self):
        return self._available

    def embed_one(self, text):
        return self._v.get(text)

    def embed(self, texts):
        return [self._v.get(t) for t in texts]


def test_semantic_memory_remember_and_recall(tmp_path):
    db = tmp_path / "m.db"
    init_db(db)
    vectors = {
        "great moat": [1.0, 0.0, 0.0],
        "similar moat": [0.95, 0.05, 0.0],
        "unrelated": [0.0, 1.0, 0.0],
    }
    mem = SemanticMemory(_FakeEmb(vectors))
    with get_session() as s:
        assert mem.remember(s, kind="thesis", text="great moat", symbol="AAA") is not None
        mem.remember(s, kind="thesis", text="unrelated", symbol="ZZZ")
    with get_session() as s:
        hits = mem.recall(s, "similar moat", kind="thesis", top_k=1)
        assert hits and hits[0][0].text == "great moat"
        assert mem.is_duplicate(s, "great moat", kind="thesis") is True


def test_semantic_memory_degrades_when_unavailable(tmp_path):
    db = tmp_path / "m.db"
    init_db(db)
    mem = SemanticMemory(_FakeEmb({}, available=False))  # embed_one returns None
    with get_session() as s:
        assert mem.remember(s, kind="thesis", text="x") is None
        assert mem.recall(s, "x") == []
        assert mem.is_duplicate(s, "x") is False
