"""CRUD operations for research database models."""

from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .research_models import (
    CandidateScore,
    CongressionalTrade,
    PerformanceTracker,
    ResearchProfile,
    ResearchReport,
    StockCandidate,
)


# Profile operations
def get_or_create_default_profile(session: Session) -> ResearchProfile:
    """Get the default profile or create one if none exists."""
    stmt = select(ResearchProfile).where(ResearchProfile.name == "default")
    profile = session.scalar(stmt)
    if profile is None:
        profile = ResearchProfile(name="default")
        session.add(profile)
        session.flush()
    return profile


def save_profile(session: Session, profile: ResearchProfile) -> int:
    """Save/update a profile, returning its ID."""
    session.add(profile)
    session.flush()
    return profile.id


# Candidate operations
def save_candidate(session: Session, candidate: StockCandidate) -> int:
    """Save a stock candidate, returning its ID."""
    session.add(candidate)
    session.flush()
    return candidate.id


def get_candidate_by_ticker(session: Session, ticker: str) -> StockCandidate | None:
    """Get candidate by ticker."""
    stmt = select(StockCandidate).where(StockCandidate.ticker == ticker)
    return session.scalar(stmt)


def get_candidates_by_status(
    session: Session, status: str, limit: int = 100
) -> list[StockCandidate]:
    """Get candidates with given status."""
    stmt = (
        select(StockCandidate)
        .where(StockCandidate.status == status)
        .order_by(StockCandidate.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_top_candidates(
    session: Session, limit: int = 20, min_score: float | None = None
) -> list[StockCandidate]:
    """Get top scoring candidates."""
    stmt = select(StockCandidate).where(StockCandidate.composite_score.isnot(None))
    if min_score is not None:
        stmt = stmt.where(StockCandidate.composite_score >= min_score)
    stmt = stmt.order_by(StockCandidate.composite_score.desc()).limit(limit)
    return list(session.scalars(stmt))


# Score operations
def save_score(session: Session, score: CandidateScore) -> int:
    """Save a candidate score, returning its ID."""
    session.add(score)
    session.flush()
    return score.id


def get_latest_score(session: Session, ticker: str) -> CandidateScore | None:
    """Get latest score for ticker."""
    stmt = (
        select(CandidateScore)
        .where(CandidateScore.ticker == ticker)
        .order_by(CandidateScore.created_at.desc(), CandidateScore.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def get_score_history(
    session: Session, ticker: str, limit: int = 10
) -> list[CandidateScore]:
    """Get score history for a ticker."""
    stmt = (
        select(CandidateScore)
        .where(CandidateScore.ticker == ticker)
        .order_by(CandidateScore.created_at.desc(), CandidateScore.id.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


# Report operations
def save_report(session: Session, report: ResearchReport) -> int:
    """Save a research report, returning its ID."""
    session.add(report)
    session.flush()
    return report.id


def get_latest_report(session: Session, ticker: str) -> ResearchReport | None:
    """Get latest report for ticker."""
    stmt = (
        select(ResearchReport)
        .where(ResearchReport.ticker == ticker)
        .order_by(ResearchReport.created_at.desc(), ResearchReport.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


# Performance operations
def save_performance_record(session: Session, record: PerformanceTracker) -> int:
    """Save a performance record, returning its ID."""
    session.add(record)
    session.flush()
    return record.id


def get_records_needing_update(
    session: Session, days_since_update: int = 7
) -> list[PerformanceTracker]:
    """Get records that need return updates.

    Returns records that either:
    - Have never been updated (updated_at == created_at)
    - Haven't been updated in the specified number of days
    """
    cutoff = datetime.now() - timedelta(days=days_since_update)
    stmt = (
        select(PerformanceTracker)
        .where(PerformanceTracker.updated_at <= cutoff)
        .order_by(PerformanceTracker.updated_at.asc())
    )
    return list(session.scalars(stmt))


# Congressional trade operations
def save_congressional_trade(session: Session, trade: CongressionalTrade) -> int:
    """Save a congressional trade, returning its ID."""
    session.add(trade)
    session.flush()
    return trade.id


def get_trades_for_ticker(
    session: Session, ticker: str, days: int = 90
) -> list[CongressionalTrade]:
    """Get recent trades for a ticker."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(CongressionalTrade)
        .where(
            CongressionalTrade.ticker == ticker,
            CongressionalTrade.trade_date >= cutoff,
        )
        .order_by(CongressionalTrade.trade_date.desc())
    )
    return list(session.scalars(stmt))
