"""CRUD for confluence findings (the insight engine's output)."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .insight_models import ConfluenceFinding


def save_finding(session: Session, finding: ConfluenceFinding) -> int:
    """Persist a finding, returning its id."""
    session.add(finding)
    session.flush()
    return finding.id


def finding_exists_for_date(
    session: Session, ticker: str, kind: str, as_of_date: date
) -> bool:
    """True if a finding for this ticker/kind is already recorded for the day."""
    stmt = select(ConfluenceFinding.id).where(
        ConfluenceFinding.ticker == ticker,
        ConfluenceFinding.kind == kind,
        ConfluenceFinding.as_of_date == as_of_date,
    )
    return session.scalar(stmt.limit(1)) is not None


def get_recent_findings(
    session: Session, *, kind: str | None = None, min_score: float = 0.0,
    limit: int = 50, max_age_days: int | None = None,
) -> list[ConfluenceFinding]:
    """Most relevant recent findings (newest day first, then strongest score).

    ``max_age_days`` bounds how stale a finding may be — required by the promotion
    bridge so it never acts on a multi-week-old finding as if it were fresh.
    """
    stmt = select(ConfluenceFinding).where(ConfluenceFinding.score >= min_score)
    if kind:
        stmt = stmt.where(ConfluenceFinding.kind == kind)
    if max_age_days is not None:
        cutoff = date.today() - timedelta(days=max_age_days)
        stmt = stmt.where(
            ConfluenceFinding.as_of_date.is_not(None),
            ConfluenceFinding.as_of_date >= cutoff,
        )
    stmt = stmt.order_by(
        ConfluenceFinding.as_of_date.desc(), ConfluenceFinding.score.desc()
    ).limit(max(1, limit))
    return list(session.scalars(stmt))
