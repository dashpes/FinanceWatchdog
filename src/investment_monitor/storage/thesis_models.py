"""SQLAlchemy ORM models for the autonomous investor's thesis store (Phase 3).

A ``Thesis`` is the persistent memory that lets the advisor *maintain* a view on a
name over time: the narrative argument, a conviction score, the sized target
weight, and — critically — the **invalidation conditions** that deterministically
trip an exit. The LLM writes/updates the narrative and conviction; execution is
still governed entirely by the deterministic guardrail gate.
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base


class ThesisStatus(str, Enum):
    """Lifecycle of a thesis."""

    DRAFT = "draft"              # generated, not yet eligible to size/trade
    ACTIVE = "active"           # live, drives target weight
    WATCH = "watch"             # held but conviction softening / under review
    INVALIDATED = "invalidated"  # an invalidation condition fired -> force-exit candidate
    EXITED = "exited"           # position closed; kept for memory/audit


# Statuses whose theses contribute to the live target allocation.
LIVE_THESIS_STATUSES = (ThesisStatus.ACTIVE.value, ThesisStatus.WATCH.value)


class Thesis(Base):
    """A maintained investment thesis for one symbol."""

    __tablename__ = "theses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    # Multi-account ready from day one (nullable while a single account is live).
    account_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # 0.0-1.0, deterministically clamped. Raw value; effective conviction applies decay.
    conviction: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # 0.0-1.0, OUTPUT of the deterministic sizing function (cached, recomputed each run).
    target_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Structured predicates (JSON). entry_conditions document the setup;
    # invalidation_conditions are evaluated deterministically by check_invalidation.
    entry_conditions: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    invalidation_conditions: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # References into other tables for audit: {report_id, score_id, sim_id, alert_ids:[...]}.
    evidence_refs: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ThesisStatus.DRAFT.value, index=True
    )

    # Append-only audit of conviction changes: [{ts, conviction, trigger}].
    conviction_history: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # When the thesis was last re-evaluated (drives conviction time-decay).
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_thesis_symbol_account", "symbol", "account_id"),
        Index("ix_thesis_status", "status"),
    )
