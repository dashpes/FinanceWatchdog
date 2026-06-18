"""Semantic memory (Phase 5): embed + recall theses/news for the autonomous investor.

Ties the embeddings client to the memory store so the agent can (a) recall similar
past setups when forming a thesis and (b) skip near-duplicate work. Entirely
optional and best-effort: if the embedding model isn't available, every method
degrades to a no-op so the rest of the pipeline runs unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from investment_monitor.storage import (
    is_duplicate,
    save_embedding,
    search_similar,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.analysis.embeddings import EmbeddingsClient
    from investment_monitor.storage import MemoryEmbedding


class SemanticMemory:
    """Embed-and-recall over the memory store (best-effort)."""

    def __init__(self, embeddings: "EmbeddingsClient") -> None:
        self._emb = embeddings

    def available(self) -> bool:
        return self._emb.is_available()

    def remember(
        self,
        session: "Session",
        *,
        kind: str,
        text: str,
        symbol: str | None = None,
        ref_id: int | None = None,
    ) -> int | None:
        """Embed ``text`` and store it; returns the row id or None if unavailable."""
        vector = self._emb.embed_one(text)
        if vector is None:
            return None
        return save_embedding(
            session, kind=kind, text=text, vector=vector, symbol=symbol, ref_id=ref_id
        )

    def recall(
        self,
        session: "Session",
        query_text: str,
        *,
        kind: str | None = None,
        top_k: int = 5,
    ) -> list[tuple["MemoryEmbedding", float]]:
        """Most-similar stored snippets to ``query_text`` ([] if unavailable)."""
        vector = self._emb.embed_one(query_text)
        if vector is None:
            return []
        return search_similar(session, vector, kind=kind, top_k=top_k)

    def is_duplicate(
        self, session: "Session", text: str, *, kind: str | None = None, threshold: float = 0.95
    ) -> bool:
        vector = self._emb.embed_one(text)
        if vector is None:
            return False
        return is_duplicate(session, vector, kind=kind, threshold=threshold)
