"""SQLAlchemy ORM for the autonomous investor's semantic memory (Phase 5).

Stores embedding vectors for theses/news/reports so the agent can recall similar
past setups and de-duplicate. Brute-force cosine over JSON vectors is fine at this
scale; a pgvector swap is a later optimization.
"""

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base


class MemoryEmbedding(Base):
    """One embedded text snippet, linked back to its source row."""

    __tablename__ = "memory_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # thesis|news|report
    ref_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    vector: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_memory_kind_symbol", "kind", "symbol"),)
