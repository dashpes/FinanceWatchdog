"""Ledger: runs, orders with the full gate story, and per-symbol realized P&L."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from investment_monitor.storage.robo_models import RoboOrder, RoboRun

from ..deps import get_read_session
from ._serialize import num, order_dict, run_dict

router = APIRouter(tags=["trades"])


@router.get("/runs")
def runs(
    limit: int = Query(30, ge=1, le=200),
    session: Session = Depends(get_read_session),
) -> dict:
    rows = session.query(RoboRun).order_by(RoboRun.started_at.desc()).limit(limit).all()
    return {"runs": [run_dict(r) for r in rows]}


@router.get("/runs/{run_id}")
def run_detail(run_id: str, session: Session = Depends(get_read_session)) -> dict:
    run = session.query(RoboRun).filter(RoboRun.run_id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    orders = (
        session.query(RoboOrder)
        .filter(RoboOrder.run_id == run_id)
        .order_by(RoboOrder.id.asc())
        .all()
    )
    return {"run": run_dict(run), "orders": [order_dict(o) for o in orders]}


@router.get("/orders")
def orders(
    symbol: str | None = None,
    side: str | None = None,
    placed_only: bool = False,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_read_session),
) -> dict:
    stmt = select(RoboOrder).order_by(RoboOrder.created_at.desc())
    if symbol:
        stmt = stmt.where(RoboOrder.symbol == symbol.upper())
    if side:
        stmt = stmt.where(RoboOrder.side == side.lower())
    if placed_only:
        stmt = stmt.where((RoboOrder.placed.is_(True)) | (RoboOrder.simulated.is_(True)))
    rows = list(session.scalars(stmt.offset(offset).limit(limit)))
    return {"orders": [order_dict(o) for o in rows], "offset": offset, "limit": limit}


@router.get("/pnl")
def pnl(session: Session = Depends(get_read_session)) -> dict:
    from investment_monitor.robo.pnl import realized_pnl, trades_from_fills
    from investment_monitor.storage import get_filled_robo_orders

    rp = realized_pnl(trades_from_fills(get_filled_robo_orders(session)))
    return {
        "total_realized": num(rp.total_realized),
        "per_symbol": [
            {
                "symbol": sp.symbol,
                "realized": num(sp.realized),
                "quantity": num(sp.quantity),
                "avg_cost": num(sp.avg_cost),
            }
            for sp in sorted(
                rp.per_symbol.values(), key=lambda s: s.realized, reverse=True
            )
        ],
    }
