"""SQLAlchemy ORM model for the learning/feedback ledger (Phase 6).

``learning_events`` is the single append-only audit table for the robo advisor's
feedback loop: every realized thesis outcome, accuracy-based sizing modifier, and
(future) adapted scoring weight is one row here.

It is the **system of record** — the full history lives in this table at *zero cost
to the LLM context window*. Only compact, EWMA-smoothed aggregates are ever injected
into prompts or sizing (see ``learning_operations.accuracy_stats_for_symbol``), so
the feedback loop can grow without bound yet never rot the context.

It is a brand-new table (never an ``ALTER``), so ``Base.metadata.create_all`` auto-
creates it on the existing live ``data/portfolio.db`` with zero migration.
"""

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base

# Event kinds recorded in the ledger.
LEARNING_KIND_OUTCOME = "thesis_outcome"
LEARNING_KIND_ACCURACY_MODIFIER = "accuracy_modifier"
LEARNING_KIND_WEIGHT_ADAPTATION = "weight_adaptation"
# Counterfactual outcome of a considered-but-not-traded thesis (shadow ledger). A
# DISTINCT kind so accuracy_stats_for_symbol (kind == thesis_outcome) never mixes
# hypothetical outcomes into real-money accuracy sizing.
LEARNING_KIND_SHADOW_OUTCOME = "shadow_outcome"


class LearningEvent(Base):
    """One append-only row in the feedback ledger."""

    __tablename__ = "learning_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    symbol: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    # Multi-account ready (nullable while a single account is live).
    account_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Soft reference to theses.id (no hard FK, to stay create_all-friendly).
    thesis_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The trading day this event represents (UTC). Used to de-duplicate intraday
    # re-evals to ONE outcome per symbol/day, so a thesis re-evaluated 4x/day does not
    # flood the accuracy window with autocorrelated copies of the same return.
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    # --- Outcome fields (kind == thesis_outcome) -------------------------------
    # The conviction the LLM HELD at eval time paired with the REALIZED price return
    # since entry. This is the raw training signal — grounded in market prices, never
    # in the LLM's self-assessment.
    conviction: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)  # fraction, e.g. -0.40
    # 1 if the (long-only) thesis was directionally right (realized_return > 0) else 0.
    direction_correct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Brier score (conviction - outcome)^2; lower is better, penalizes confident-and-wrong.
    brier: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Modifier / adaptation fields (kind == accuracy_modifier|weight_adaptation) --
    # applied=False means the signal was computed and logged but NOT acted on (shadow).
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    before_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    after_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Flexible blob for anything kind-specific (reassign-not-mutate to persist).
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_learning_event_kind_symbol", "kind", "symbol"),
    )
