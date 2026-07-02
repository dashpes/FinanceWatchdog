"""Charts: OHLCV series with the robo's own trade markers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from investment_monitor.storage.models import Price
from investment_monitor.storage.robo_models import RoboOrder
from investment_monitor.storage.thesis_models import Thesis

from ..deps import get_read_session
from ._serialize import iso, num

router = APIRouter(tags=["charts"])


@router.get("/charts/symbols")
def symbols(session: Session = Depends(get_read_session)) -> dict:
    """Symbols worth charting: anything with a thesis or a robo order."""
    thesis_syms = {s for (s,) in session.query(Thesis.symbol).distinct()}
    order_syms = {s for (s,) in session.query(RoboOrder.symbol).distinct()}
    return {"symbols": sorted(thesis_syms | order_syms)}


@router.get("/charts/price/{symbol}")
def price(
    symbol: str,
    days: int = Query(180, ge=5, le=1500),
    session: Session = Depends(get_read_session),
) -> dict:
    from investment_monitor.storage.operations import get_prices

    symbol = symbol.upper()
    rows = list(reversed(get_prices(session, symbol, days=days)))
    trades = list(
        session.scalars(
            select(RoboOrder)
            .where(
                RoboOrder.symbol == symbol,
                RoboOrder.fill_price.isnot(None),
            )
            .order_by(RoboOrder.created_at.asc())
        )
    )
    return {
        "symbol": symbol,
        "candles": [
            {
                "date": iso(p.date),
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
            for p in rows
        ],
        "trades": [
            {
                "date": iso(o.created_at),
                "side": o.side,
                "fill_price": num(o.fill_price),
                "fill_quantity": num(o.fill_quantity),
                "rationale": o.rationale,
            }
            for o in trades
        ],
    }
