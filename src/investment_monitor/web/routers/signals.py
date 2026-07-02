"""Signals: confluence findings and the insider-transaction drill-down."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from investment_monitor.storage.models import InsiderTransaction

from ..deps import get_read_session
from ._serialize import iso, num

router = APIRouter(tags=["signals"])


@router.get("/signals/confluence")
def confluence(
    days: int = Query(30, ge=1, le=365),
    kind: str | None = None,
    min_score: float = 0.0,
    limit: int = Query(100, ge=1, le=500),
    session: Session = Depends(get_read_session),
) -> dict:
    from investment_monitor.storage.insight_operations import get_recent_findings

    findings = get_recent_findings(
        session, kind=kind, min_score=min_score, limit=limit, max_age_days=days
    )
    return {
        "findings": [
            {
                "id": f.id,
                "ticker": f.ticker,
                "kind": f.kind,
                "score": f.score,
                "window_days": f.window_days,
                "n_sources": f.n_sources,
                "n_actors": f.n_actors,
                "total_value": num(f.total_value),
                "price_change_pct": num(f.price_change_pct),
                "narrative": f.narrative,
                "evidence": f.evidence,
                "as_of_date": iso(f.as_of_date),
            }
            for f in findings
        ]
    }


@router.get("/signals/insiders")
def insiders(
    ticker: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_read_session),
) -> dict:
    stmt = select(InsiderTransaction).order_by(
        InsiderTransaction.filing_date.desc(), InsiderTransaction.id.desc()
    )
    if ticker:
        stmt = stmt.where(InsiderTransaction.ticker == ticker.upper())
    rows = list(session.scalars(stmt.offset(offset).limit(limit)))
    return {
        "transactions": [
            {
                "ticker": t.ticker,
                "filing_date": iso(t.filing_date),
                "trade_date": iso(t.trade_date),
                "owner_name": t.owner_name,
                "owner_title": t.owner_title,
                "transaction_type": t.transaction_type,
                "shares": num(t.shares),
                "price_per_share": num(t.price_per_share),
                "total_value": num(t.total_value),
                "sec_url": t.sec_url,
            }
            for t in rows
        ],
        "offset": offset,
        "limit": limit,
    }
