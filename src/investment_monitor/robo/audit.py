"""Append-only structured audit log for the robo advisor.

Every model proposal, every gate decision (accept/reject + reason), and every
placed/simulated order is written as a timestamped JSON line to
``logs/robo_audit.jsonl``. The file is append-only and never rewritten, so it is
a durable record of exactly what the advisor considered and did on every run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from investment_monitor.robo.broker import PreflightResult
from investment_monitor.robo.models import GateDecision, ProposedOrder


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _order_dict(order: ProposedOrder) -> dict[str, Any]:
    return {
        "symbol": order.symbol,
        "side": order.side.value if isinstance(order.side, Enum) else order.side,
        "order_type": (
            order.order_type.value if isinstance(order.order_type, Enum) else order.order_type
        ),
        "quantity": order.quantity,
        "notional": order.notional,
        "limit_price": order.limit_price,
        "reason": order.reason,
        "source": order.source,
    }


class AuditLogger:
    """Writes append-only JSONL audit events for a single run."""

    def __init__(self, log_dir: Path, run_id: str, filename: str = "robo_audit.jsonl") -> None:
        self.run_id = run_id
        self.path = Path(log_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **payload,
        }
        line = json.dumps(record, default=_json_default)
        # Append-only; flush each line so a crash mid-run still leaves a record.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.bind(component="robo_audit").debug("{event}: {payload}", event=event, payload=payload)

    # -- event helpers --------------------------------------------------------

    def run_start(self, *, dry_run: bool, account_id: str, total_value: Decimal,
                  settled_cash: Decimal, source: str) -> None:
        self._write("run_start", {
            "dry_run": dry_run,
            "account_id": account_id,
            "total_value": total_value,
            "settled_cash": settled_cash,
            "source": source,
        })

    def safety_check(self, *, passed: bool, detail: str) -> None:
        self._write("safety_check", {"passed": passed, "detail": detail})

    def signals(self, snapshot: Any) -> None:
        """Record the event-signal snapshot that informed this run's proposal."""
        symbols = {
            sym: {
                "score": s.score,
                "has_caution": s.has_caution,
                "summary": s.summary,
                "events": [
                    {
                        "category": e.category,
                        "direction": e.direction,
                        "magnitude": e.magnitude,
                        "age_days": e.age_days,
                        "caution": e.caution,
                        "detail": e.detail,
                    }
                    for e in s.events
                ],
            }
            for sym, s in snapshot.symbols.items()
        }
        self._write("signals", {
            "as_of": snapshot.as_of,
            "lookback_days": snapshot.lookback_days,
            "symbols": symbols,
        })

    def proposal(self, order: ProposedOrder) -> None:
        self._write("proposal", {"order": _order_dict(order)})

    def gate_decision(self, decision: GateDecision) -> None:
        self._write("gate_decision", {
            "accepted": decision.accepted,
            "code": decision.code,
            "reason": decision.reason,
            "order": _order_dict(decision.order),
        })

    def preflight(self, order: ProposedOrder, result: PreflightResult) -> None:
        self._write("preflight", {
            "order": _order_dict(order),
            "ok": result.ok,
            "estimated_cost": result.estimated_cost,
            "total_fees": result.total_fees,
            "margin_requirement": result.margin_requirement,
            "message": result.message,
        })

    def order_result(self, order: ProposedOrder, *, simulated: bool, placed: bool,
                     broker_order_id: str = "", status: str = "", detail: str = "") -> None:
        self._write("order_result", {
            "order": _order_dict(order),
            "simulated": simulated,
            "placed": placed,
            "broker_order_id": broker_order_id,
            "status": status,
            "detail": detail,
        })

    def run_summary(self, *, num_proposed: int, num_accepted: int, num_rejected: int,
                    num_placed: int, dry_run: bool, status: str) -> None:
        self._write("run_summary", {
            "num_proposed": num_proposed,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "num_placed": num_placed,
            "dry_run": dry_run,
            "status": status,
        })
