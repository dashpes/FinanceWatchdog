"""Overview: the front page — value, P&L, today's dealings, movers, bot status."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from loguru import logger
from sqlalchemy.orm import Session

from investment_monitor.config import Settings
from investment_monitor.robo import control
from investment_monitor.storage.robo_models import RoboOrder, RoboRun

from ..deps import get_app_settings, get_read_session
from ._serialize import iso, num, order_dict, position_dict, run_dict

router = APIRouter(tags=["overview"])


def _realized(session: Session):
    """Realized P&L from the bot's own fills; None when unavailable."""
    try:
        from investment_monitor.robo.pnl import realized_pnl, trades_from_fills
        from investment_monitor.storage import get_filled_robo_orders

        return realized_pnl(trades_from_fills(get_filled_robo_orders(session)))
    except Exception as exc:  # noqa: BLE001 - the page must degrade, not 500
        logger.warning("realized P&L unavailable: {e}", e=exc)
        return None


def _todays_orders(session: Session) -> list[dict]:
    start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    rows = (
        session.query(RoboOrder)
        .filter(RoboOrder.created_at >= start)
        .order_by(RoboOrder.created_at.asc())
        .all()
    )
    return [order_dict(r) for r in rows if r.placed or r.simulated]


def _latest_run(session: Session):
    return session.query(RoboRun).order_by(RoboRun.started_at.desc()).first()


@router.get("/overview")
async def overview(
    request: Request,
    session: Session = Depends(get_read_session),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    snapshot = await request.app.state.account_cache.get()
    account = snapshot["account"]
    latest = _latest_run(session)

    if account is not None:
        acct = {
            "total_value": num(account.total_value),
            "settled_cash": num(account.settled_cash),
            "positions_value": num(account.positions_value),
            "unrealized_gain": num(account.total_unrealized_gain),
            "cost_basis": num(account.total_cost_basis),
            "positions": [
                position_dict(p, account.total_value) for p in account.positions
            ],
            "stale": snapshot["stale"],
            "as_of": snapshot["as_of"],
            "source": "broker",
        }
    elif latest is not None:
        # Never fetched from the broker: degrade to the last persisted run.
        acct = {
            "total_value": num(latest.total_value),
            "settled_cash": num(latest.settled_cash),
            "positions_value": None,
            "unrealized_gain": num(latest.unrealized_pnl),
            "cost_basis": None,
            "positions": None,
            "stale": True,
            "as_of": iso(latest.started_at),
            "source": "last_run",
        }
    else:
        acct = None

    realized = _realized(session)
    movers = []
    if account is not None:
        with_gain = [p for p in account.positions if p.unrealized_gain is not None]
        with_gain.sort(key=lambda p: abs(p.unrealized_gain), reverse=True)
        movers = [position_dict(p, account.total_value) for p in with_gain[:5]]

    ctl = control.load_control(settings.db_path)
    return {
        "account": acct,
        "realized_total": num(realized.total_realized) if realized else None,
        "todays_orders": _todays_orders(session),
        "movers": movers,
        "bot": {
            "last_run": run_dict(latest) if latest else None,
            "trading_paused": ctl.trading_paused,
            "control_force_dry_run": ctl.force_dry_run,
            "env_force_dry_run": settings.robo_force_dry_run,
        },
    }


@router.get("/overview/equity")
def equity(session: Session = Depends(get_read_session)) -> dict:
    """Equity curve: last known total value per day, from the runs time series."""
    rows = (
        session.query(RoboRun)
        .filter(RoboRun.total_value.isnot(None))
        .order_by(RoboRun.started_at.asc())
        .all()
    )
    by_day: dict[str, dict] = {}
    for r in rows:
        day = r.started_at.date().isoformat()
        by_day[day] = {
            "date": day,
            "total_value": num(r.total_value),
            "settled_cash": num(r.settled_cash),
            "unrealized_pnl": num(r.unrealized_pnl),
        }
    return {"points": list(by_day.values())}
