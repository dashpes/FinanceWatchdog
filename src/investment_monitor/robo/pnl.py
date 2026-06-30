"""Pure realized-P&L accounting over the broker's executed-trade ledger.

Public is the source of truth for *positions* and *unrealized* P&L (we read its
``CostBasis`` directly — see ``broker.account_state_from_raw``). *Realized* P&L is
not exposed per-trade, so it is reconstructed here from the transaction history via
average-cost accounting.

No network and no DB — a pure function over a list of :class:`Trade`, so it is fully
unit-testable and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from investment_monitor.robo.models import OrderSide, Trade


class SymbolPnL(BaseModel):
    """Realized P&L and the reconstructed remaining lot for one symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    realized: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")  # shares still held per the reconstruction
    avg_cost: Decimal = Decimal("0")  # average cost of the remaining shares
    fees: Decimal = Decimal("0")  # total fees seen for this symbol


class RealizedPnL(BaseModel):
    """Account-wide realized P&L plus the per-symbol breakdown."""

    model_config = ConfigDict(frozen=True)

    per_symbol: dict[str, SymbolPnL]
    total_realized: Decimal
    total_fees: Decimal

    def symbol_realized(self, symbol: str) -> Decimal:
        """Realized P&L for ``symbol`` (0 if it has no trade history)."""
        sp = self.per_symbol.get(symbol.upper())
        return sp.realized if sp else Decimal("0")


def _fill_to_decimal(value: Any) -> Decimal | None:
    """Coerce a stored fill field (float/str/None) to Decimal, or None if unusable."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ValueError, ArithmeticError, TypeError):
        return None


def trades_from_fills(orders: Iterable[Any]) -> list[Trade]:
    """Build the bot's OWN executed-trade ledger from its filled order records.

    Realized P&L must reflect only trades the robo itself initiated — never the
    account's pre-existing or manually-entered positions, which share the one
    brokerage account and would otherwise leak in through the account-wide trade
    history (e.g. a personal ETF sale showing up as robo "realized" gains).

    Each input is one of the robo's order rows (duck-typed: ``symbol``, ``side``,
    ``fill_price``, ``fill_quantity``, ``created_at``) carrying the broker-reconciled
    fill. ``gross`` is ``fill_price * fill_quantity``; ``created_at`` orders the
    average-cost reconstruction. Rows without a usable fill (never filled, or
    rejected/cancelled before filling) are skipped. Per-order fees are not stored,
    so realized P&L here is gross of the (sub-cent) per-trade commission.
    """
    out: list[Trade] = []
    for o in orders:
        price = _fill_to_decimal(getattr(o, "fill_price", None))
        qty = _fill_to_decimal(getattr(o, "fill_quantity", None))
        if price is None or qty is None or qty == 0:
            continue
        symbol = str(getattr(o, "symbol", "")).upper()
        side_raw = str(getattr(o, "side", "")).upper()
        if not symbol or side_raw not in ("BUY", "SELL"):
            continue
        out.append(
            Trade(
                symbol=symbol,
                side=OrderSide.BUY if side_raw == "BUY" else OrderSide.SELL,
                quantity=abs(qty),
                gross=abs(price * qty),
                fees=Decimal("0"),
                timestamp=getattr(o, "created_at", None),
            )
        )
    return out


def realized_pnl(trades: list[Trade]) -> RealizedPnL:
    """Average-cost realized P&L from a list of executed trades.

    Long-only average-cost: a buy grows the lot (fees roll into the cost basis); a
    sell realizes proceeds (net of its fees) minus the average cost of the shares
    sold. A sell beyond the reconstructed holding is clamped to what is held — a
    defensive guard that should not arise in a long-only account, but keeps the
    accounting from going negative on a gap in the history window.

    Trades are processed oldest-first. Trades that carry a timestamp are ordered
    chronologically; trades with no timestamp keep their relative input order and
    are pushed to the end (the broker returns history in order). The sort key is
    total-ordered for *any* combination of {None, tz-aware datetime, tz-naive
    datetime}, so it never raises and never silently falls back to an unsorted
    order — getting the chronology wrong would let a SELL be realized before its
    matching BUY and understate the average-cost realized P&L.
    """
    # Every timestamp is normalized to a tz-aware UTC instant before comparison:
    # a naive datetime is assumed UTC (``replace(tzinfo=...)``), so a mix of
    # aware and naive timestamps — which Python otherwise refuses to compare —
    # becomes a single total order. The ``is None`` flag sorts timestamp-less
    # trades after timestamped ones, and ``_epoch`` (also tz-aware) is a harmless
    # placeholder only ever used inside the all-None group, where the flag is
    # constant and ``sorted``'s stability preserves input order.
    _epoch = datetime.min.replace(tzinfo=timezone.utc)

    def _sort_key(t: Trade) -> tuple[bool, datetime]:
        ts = t.timestamp
        if ts is None:
            return (True, _epoch)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (False, ts)

    ordered = sorted(trades, key=_sort_key)

    state: dict[str, SymbolPnL] = {}
    for t in ordered:
        sym = t.symbol.upper()
        prev = state.get(sym)
        qty = prev.quantity if prev else Decimal("0")
        avg = prev.avg_cost if prev else Decimal("0")
        realized = prev.realized if prev else Decimal("0")
        fees_acc = (prev.fees if prev else Decimal("0")) + t.fees

        if t.side is OrderSide.BUY:
            new_qty = qty + t.quantity
            total_cost = avg * qty + t.gross + t.fees  # fees roll into basis
            avg = (total_cost / new_qty) if new_qty > 0 else Decimal("0")
            qty = new_qty
        else:  # SELL — realize against average cost
            sell_qty = min(t.quantity, qty) if qty > 0 else Decimal("0")
            # Only the matched portion realizes P&L; an unbacked excess (a buy that
            # fell outside the history window) is ignored rather than booked as a
            # phantom gain with zero cost.
            matched_frac = (sell_qty / t.quantity) if t.quantity > 0 else Decimal("0")
            proceeds = (t.gross - t.fees) * matched_frac
            realized += proceeds - avg * sell_qty
            qty -= sell_qty
            if qty <= 0:
                qty = Decimal("0")
                avg = Decimal("0")

        state[sym] = SymbolPnL(
            symbol=sym, realized=realized, quantity=qty, avg_cost=avg, fees=fees_acc
        )

    total_realized = sum((s.realized for s in state.values()), Decimal("0"))
    total_fees = sum((s.fees for s in state.values()), Decimal("0"))
    return RealizedPnL(
        per_symbol=state, total_realized=total_realized, total_fees=total_fees
    )
