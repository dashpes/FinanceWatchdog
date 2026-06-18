"""Operations for the semantic memory store (Phase 5)."""

from __future__ import annotations

import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from .memory_models import MemoryEmbedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (pure; 0.0 on degenerate input)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def save_embedding(
    session: Session,
    *,
    kind: str,
    text: str,
    vector: list[float],
    symbol: str | None = None,
    ref_id: int | None = None,
) -> int:
    """Persist an embedding, returning its id."""
    row = MemoryEmbedding(kind=kind, text=text, vector=list(vector), symbol=symbol, ref_id=ref_id)
    session.add(row)
    session.flush()
    return row.id


def search_similar(
    session: Session,
    query_vector: list[float],
    *,
    kind: str | None = None,
    symbol: str | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[MemoryEmbedding, float]]:
    """Return up to ``top_k`` stored embeddings most similar to ``query_vector``.

    Brute-force cosine over the (optionally kind/symbol-filtered) rows — fine at
    this scale. Results are ``(row, score)`` sorted by descending similarity.
    """
    if not query_vector:
        return []
    stmt = select(MemoryEmbedding)
    if kind:
        stmt = stmt.where(MemoryEmbedding.kind == kind)
    if symbol:
        stmt = stmt.where(MemoryEmbedding.symbol == symbol)
    rows = list(session.scalars(stmt))
    scored = [
        (row, cosine_similarity(query_vector, row.vector or []))
        for row in rows
    ]
    scored = [pair for pair in scored if pair[1] >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def is_duplicate(
    session: Session,
    query_vector: list[float],
    *,
    kind: str | None = None,
    threshold: float = 0.95,
) -> bool:
    """True if a near-identical embedding already exists (dedup before LLM work)."""
    top = search_similar(session, query_vector, kind=kind, top_k=1)
    return bool(top) and top[0][1] >= threshold
