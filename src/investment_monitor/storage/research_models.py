"""SQLAlchemy ORM models for research and stock discovery."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base


# Valid statuses for StockCandidate
CANDIDATE_STATUSES = (
    "discovered",
    "screening",
    "researched",
    "watchlist",
    "rejected",
    "archived",
)


class ResearchProfile(Base):
    """Investment profile and preferences - singleton per system."""

    __tablename__ = "research_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, default="default")
    investment_style: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    risk_tolerance: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sector_preferences: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON-encoded list
    value_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    growth_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    quality_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    momentum_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    sentiment_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("name", name="uq_research_profile_name"),)


class StockCandidate(Base):
    """Stock candidates discovered for research."""

    __tablename__ = "stock_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    discovery_source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="discovered", index=True
    )
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("ticker", name="uq_stock_candidate_ticker"),
        Index("ix_stock_candidate_status", "status"),
    )


class CandidateScore(Base):
    """Factor scores for a stock candidate."""

    __tablename__ = "candidate_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    value_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    growth_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    momentum_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("ticker", name="uq_candidate_score_ticker"),
        Index("ix_candidate_score_ticker", "ticker"),
    )


class ResearchReport(Base):
    """Research reports for stock candidates."""

    __tablename__ = "research_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bull_case: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bear_case: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_research_report_ticker", "ticker"),)


class PerformanceTracker(Base):
    """Track performance of stock candidates over time."""

    __tablename__ = "performance_trackers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    return_30d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    return_60d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    return_90d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("ticker", "entry_date", name="uq_performance_ticker_date"),
        Index("ix_performance_tracker_ticker", "ticker"),
    )


class CongressionalTrade(Base):
    """Congressional trading disclosures."""

    __tablename__ = "congressional_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    politician: Mapped[str] = mapped_column(String(200), nullable=False)
    party: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    chamber: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # House/Senate
    trade_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # buy/sell/exchange
    amount_range: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "$1,001-$15,000"
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    disclosure_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "politician",
            "trade_date",
            "trade_type",
            "amount_range",
            name="uq_congressional_trade",
        ),
        Index("ix_congressional_trade_ticker", "ticker"),
        Index("ix_congressional_trade_politician", "politician"),
        Index("ix_congressional_trade_date", "trade_date"),
    )


class SimulationResult(Base):
    """Stores Monte Carlo simulation results for a ticker."""

    __tablename__ = "simulation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Parameters used
    num_simulations: Mapped[int] = mapped_column(Integer, nullable=False)
    lookback_days: Mapped[int] = mapped_column(Integer, nullable=False)
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    drift: Mapped[float] = mapped_column(Float, nullable=False)

    # Results stored as JSON blobs per horizon
    results_30d: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    results_90d: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    results_252d: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Sensitivity analysis summary
    sensitivity_analysis: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_simulation_result_ticker", "ticker"),
        Index("ix_simulation_result_run_date", "run_date"),
    )
