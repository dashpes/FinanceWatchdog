"""SQLAlchemy ORM model for confluence findings (the insight engine's output).

A ``ConfluenceFinding`` is a FIRST-CLASS insight — a stated "look here", not a number
buried in a sizing tilt. It records that multiple independent actors (and, as more
sources come online, multiple distinct data SOURCES) line up on the same ticker in a
window: e.g. "7 insiders made open-market purchases of CBKM this week."

This is the output the original aggregate→insight vision was missing. New table, so
``create_all`` / the schema reconcile handle it with zero migration.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base

# Finding kinds.
FINDING_INSIDER_CLUSTER = "insider_cluster"   # >=N distinct insiders buying one name
FINDING_MULTI_SOURCE = "multi_source"         # >=2 distinct sources agreeing on one name
FINDING_CONGRESS_CLUSTER = "congress_cluster"  # >=N distinct members of Congress buying one name


class ConfluenceFinding(Base):
    """One stated cross-source / cross-actor insight for a ticker on a given day."""

    __tablename__ = "confluence_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # Confluence strength — super-additive in distinct actors AND distinct sources.
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    window_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # How many distinct data SOURCES (insider/congress/volume/...) contributed.
    n_sources: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # How many distinct ACTORS across sources (e.g. distinct insiders + politicians).
    n_actors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Price return since the buying began (latest close vs close at the median event
    # date) — the payoff/risk anchor: "look here, and it's already up X%". Nullable
    # (no price data for the ticker).
    price_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Bounded list of the contributing evidence units (reassign-not-mutate to persist).
    evidence: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # The stated insight, human-readable.
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # The day this finding was computed — the de-dup key (one per ticker/kind/day).
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_confluence_ticker_kind_date", "ticker", "kind", "as_of_date"),
    )
