"""SQLAlchemy ORM model for the shadow ledger.

A ``ShadowEntry`` is a thesis the system CONSIDERED but did not trade — a confluence
finding that missed the promotion floor, overflowed the per-run cap, failed the
liquidity/run-up guards, a discovery candidate just under the score floor, or a buy
the guardrail gate rejected. Each entry snapshots the entry price so the counter-
factual outcome ("what would have happened had we taken it?") can be marked daily
and closed at a fixed horizon.

Why: a small live cash account realizes outcomes very slowly, so the Phase 6
learning loop starves. Shadow outcomes accrue 10-50x faster at zero risk and feed
the SAME ``learning_events`` ledger under a distinct kind (``shadow_outcome``), so
real-money accuracy stats stay uncontaminated while skip-policy quality (are the
guards saving us money or costing us money?) becomes measurable.

Brand-new table (never an ``ALTER``) so ``Base.metadata.create_all`` auto-creates it
on the live DB with zero migration.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base

# Where the skipped candidate came from.
SHADOW_SOURCE_CONFLUENCE = "confluence"
SHADOW_SOURCE_DISCOVERY = "discovery"
SHADOW_SOURCE_GATE = "gate_reject"

SHADOW_STATUS_OPEN = "open"
SHADOW_STATUS_CLOSED = "closed"


class ShadowEntry(Base):
    """One considered-but-not-traded thesis, tracked hypothetically."""

    __tablename__ = "shadow_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    account_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # WHY it was skipped (below_score_floor | cap_overflow | run_up | illiquid |
    # reentry_guard | gate:<code> ...) — the dimension the report groups by.
    skip_reason: Mapped[str] = mapped_column(String(40), nullable=False)
    # Soft reference into the source table (finding/candidate/robo_order id).
    ref_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Snapshot of the would-have-been thesis at skip time.
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    conviction: Mapped[float | None] = mapped_column(Float, nullable=True)

    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    # Latest close at record time. None = no price known; the entry is kept for the
    # record but can never be marked or closed.
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default=SHADOW_STATUS_OPEN, index=True
    )
    # Running counterfactual mark (fraction, e.g. -0.12), refreshed while open and
    # frozen at close.
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("symbol", "source", "entry_date", name="uq_shadow_symbol_source_date"),
        Index("ix_shadow_status_source", "status", "source"),
    )
