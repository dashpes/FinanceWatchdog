"""CRUD for SEC 8-K material events."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .event_models import MaterialEvent


def material_event_exists(session: Session, sec_url: str) -> bool:
    """True if this filing is already ingested (dedup key: EDGAR URL)."""
    stmt = select(MaterialEvent.id).where(MaterialEvent.sec_url == sec_url)
    return session.scalar(stmt.limit(1)) is not None


def get_material_events(
    session: Session, ticker: str, *, days: int = 30
) -> list[MaterialEvent]:
    """Recent 8-K events for a ticker (newest first)."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(MaterialEvent)
        .where(MaterialEvent.ticker == ticker, MaterialEvent.filed_date >= cutoff)
        .order_by(MaterialEvent.filed_date.desc())
    )
    return list(session.scalars(stmt))
