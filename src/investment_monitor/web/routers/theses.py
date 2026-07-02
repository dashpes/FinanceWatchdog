"""Theses: the book of ideas — list, full narrative detail, and monitoring."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from investment_monitor.storage.learning_models import LearningEvent
from investment_monitor.storage.robo_models import RoboOrder
from investment_monitor.storage.thesis_models import Thesis

from ..deps import get_read_session
from ._serialize import iso, num, order_dict

router = APIRouter(tags=["theses"])


def _thesis_summary(t: Thesis) -> dict:
    narrative = (t.narrative or "").strip()
    return {
        "id": t.id,
        "symbol": t.symbol,
        "status": t.status,
        "conviction": t.conviction,
        "target_weight": t.target_weight,
        "excerpt": narrative[:200] + ("…" if len(narrative) > 200 else ""),
        "created_at": iso(t.created_at),
        "updated_at": iso(t.updated_at),
        "last_evaluated_at": iso(t.last_evaluated_at),
    }


@router.get("/theses")
def list_theses(
    status: str | None = None, session: Session = Depends(get_read_session)
) -> dict:
    stmt = select(Thesis).order_by(Thesis.updated_at.desc())
    if status:
        stmt = stmt.where(Thesis.status == status)
    theses = list(session.scalars(stmt))
    return {"theses": [_thesis_summary(t) for t in theses]}


def _get_thesis(session: Session, symbol: str) -> Thesis:
    stmt = (
        select(Thesis)
        .where(Thesis.symbol == symbol.upper())
        .order_by(Thesis.updated_at.desc())
    )
    thesis = session.scalars(stmt).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail=f"no thesis for {symbol.upper()}")
    return thesis


@router.get("/theses/{symbol}")
def thesis_detail(symbol: str, session: Session = Depends(get_read_session)) -> dict:
    t = _get_thesis(session, symbol)
    orders = (
        session.query(RoboOrder)
        .filter(RoboOrder.thesis_id == t.id)
        .order_by(RoboOrder.created_at.asc())
        .all()
    )
    events = (
        session.query(LearningEvent)
        .filter(LearningEvent.symbol == t.symbol)
        .order_by(LearningEvent.id.desc())
        .limit(50)
        .all()
    )
    return {
        **_thesis_summary(t),
        "narrative": t.narrative,
        "entry_conditions": t.entry_conditions,
        "invalidation_conditions": t.invalidation_conditions,
        "evidence_refs": t.evidence_refs,
        "conviction_history": t.conviction_history,
        "orders": [order_dict(o) for o in orders],
        "learning_events": [
            {
                "kind": e.kind,
                "as_of_date": iso(e.as_of_date),
                "conviction": e.conviction,
                "realized_return": e.realized_return,
                "direction_correct": e.direction_correct,
                "brier": e.brier,
                "applied": e.applied,
                "before_value": e.before_value,
                "after_value": e.after_value,
                "note": e.note,
            }
            for e in events
        ],
    }


@router.get("/theses/{symbol}/monitor")
def thesis_monitor(symbol: str, session: Session = Depends(get_read_session)) -> dict:
    """How the idea is going: entry fill vs price path, conviction trajectory."""
    from investment_monitor.storage.operations import get_prices

    t = _get_thesis(session, symbol)

    # The robo's own opening fill for this thesis (mirrors rebalance's entry logic).
    entry = (
        session.query(RoboOrder)
        .filter(
            RoboOrder.thesis_id == t.id,
            RoboOrder.side == "buy",
            RoboOrder.fill_price.isnot(None),
        )
        .order_by(RoboOrder.created_at.asc())
        .first()
    )

    days = 365
    if entry is not None and entry.created_at is not None:
        from datetime import datetime, timezone

        age = datetime.now(timezone.utc).replace(tzinfo=None) - entry.created_at
        days = max(30, min(730, age.days + 14))
    prices = list(reversed(get_prices(session, t.symbol, days=days)))

    latest_close = prices[-1].close if prices else None
    entry_price = num(entry.fill_price) if entry is not None else None
    return_since_entry = None
    if latest_close is not None and entry_price:
        return_since_entry = latest_close / entry_price - 1

    return {
        "symbol": t.symbol,
        "status": t.status,
        "conviction": t.conviction,
        "conviction_history": t.conviction_history,
        "invalidation_conditions": t.invalidation_conditions,
        "entry": {
            "fill_price": entry_price,
            "filled_at": iso(entry.created_at) if entry is not None else None,
        },
        "latest_close": latest_close,
        "return_since_entry": return_since_entry,
        "prices": [
            {"date": iso(p.date), "close": p.close}
            for p in prices
        ],
    }
