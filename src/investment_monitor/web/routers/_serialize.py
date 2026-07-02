"""Tiny serialization helpers shared by the routers (Decimal/date -> JSON)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def num(value: Any) -> float | None:
    """Decimal/int/float -> float, None stays None."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def order_dict(o) -> dict:
    """A RoboOrder row as the ledger's JSON shape (full gate story included)."""
    return {
        "id": o.id,
        "run_id": o.run_id,
        "symbol": o.symbol,
        "side": o.side,
        "order_type": o.order_type,
        "quantity": num(o.quantity),
        "notional": num(o.notional),
        "limit_price": num(o.limit_price),
        "source": o.source,
        "reason": o.reason,
        "gate_accepted": o.gate_accepted,
        "gate_code": o.gate_code,
        "gate_reason": o.gate_reason,
        "preflight_ok": o.preflight_ok,
        "preflight_reason": o.preflight_reason,
        "placed": o.placed,
        "simulated": o.simulated,
        "status": o.status,
        "fill_price": num(o.fill_price),
        "fill_quantity": num(o.fill_quantity),
        "fill_status": o.fill_status,
        "thesis_id": getattr(o, "thesis_id", None),
        "rationale": getattr(o, "rationale", None),
        "created_at": iso(getattr(o, "created_at", None)),
    }


def run_dict(r) -> dict:
    return {
        "run_id": r.run_id,
        "started_at": iso(r.started_at),
        "finished_at": iso(r.finished_at),
        "dry_run": r.dry_run,
        "account_id": r.account_id,
        "source": r.source,
        "total_value": num(r.total_value),
        "settled_cash": num(r.settled_cash),
        "unrealized_pnl": num(r.unrealized_pnl),
        "num_proposed": r.num_proposed,
        "num_accepted": r.num_accepted,
        "num_rejected": r.num_rejected,
        "num_placed": r.num_placed,
        "status": r.status,
        "notes": r.notes,
    }


def position_dict(p, total_value=None) -> dict:
    weight = None
    try:
        if total_value:
            weight = float(p.market_value / total_value)
    except Exception:  # noqa: BLE001 - cosmetic
        weight = None
    return {
        "symbol": p.symbol,
        "quantity": num(p.quantity),
        "price": num(p.price),
        "market_value": num(p.market_value),
        "weight": weight,
        "unit_cost": num(p.unit_cost),
        "unrealized_gain": num(p.unrealized_gain),
        "unrealized_return": num(p.unrealized_return),
    }
