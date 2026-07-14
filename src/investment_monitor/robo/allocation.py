"""Deterministic allocation / drift math and candidate-order generation.

Pure functions (no I/O). Given a normalized account snapshot and the target
allocation, this computes how far each holding has drifted and produces a set of
candidate orders that move the portfolio toward target. The orders are sized to
respect the configured caps so the resulting set is affordable and passes the
gate — but the gate is still the authority that accepts or rejects them.

This layer is also what runs when ``use_llm`` is False: the rebalance is computed
entirely here, and the LLM (when enabled) only proposes an alternative set that is
re-checked by the same gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.models import (
    CASH_SYMBOL,
    AccountState,
    OrderSide,
    OrderType,
    ProposedOrder,
)

_CENTS = Decimal("0.01")
# Skip dust orders below this notional to avoid churn/fees on trivial drift.
_MIN_ORDER_NOTIONAL = Decimal("1.00")
# A target WEIGHT at/below this is treated as "thesis gone" -> full exit. Keyed on weight
# (not a dollar floor) so a valid small target on a tiny account isn't force-liquidated:
# a dropped/invalidated thesis targets exactly 0, while a kept one targets >= ~1.75%.
_EXIT_WEIGHT_EPS = 0.005


@dataclass
class AllocationRow:
    """Current vs. target for one symbol (or CASH)."""

    symbol: str
    current_value: Decimal
    current_weight: float
    target_weight: float
    drift: float  # current_weight - target_weight

    @property
    def action(self) -> str:
        if self.symbol == CASH_SYMBOL:
            return "—"
        if self.drift > 0:
            return "trim" if self.current_value > 0 else "—"
        if self.drift < 0:
            return "add"
        return "hold"


def _round_cents(value: Decimal) -> Decimal:
    """Round a notional down to whole cents (down, so we never overspend)."""
    return value.quantize(_CENTS, rounding=ROUND_DOWN)


def compute_allocation(
    account_state: AccountState,
    config: RoboConfig,
) -> list[AllocationRow]:
    """Return current-vs-target rows for every target symbol plus CASH."""
    total = account_state.total_value
    rows: list[AllocationRow] = []
    if total <= 0:
        return rows

    symbols = list(config.tradeable_symbols)
    for symbol in symbols:
        pos = account_state.get_position(symbol)
        current_value = pos.market_value if pos else Decimal("0")
        current_weight = float(current_value / total)
        target_weight = config.target_allocation.get(symbol, 0.0)
        rows.append(
            AllocationRow(
                symbol=symbol,
                current_value=current_value,
                current_weight=current_weight,
                target_weight=target_weight,
                drift=current_weight - target_weight,
            )
        )

    # CASH row (residual).
    cash_weight = float(account_state.settled_cash / total)
    rows.append(
        AllocationRow(
            symbol=CASH_SYMBOL,
            current_value=account_state.settled_cash,
            current_weight=cash_weight,
            target_weight=config.cash_target_weight,
            drift=cash_weight - config.cash_target_weight,
        )
    )
    return rows


def generate_candidate_orders(
    account_state: AccountState,
    config: RoboConfig,
) -> list[ProposedOrder]:
    """Generate notional candidate orders to move the portfolio toward target.

    * Only symbols whose absolute drift exceeds ``rebalance_threshold`` are traded.
    * Each order's notional is capped at ``max_order_pct`` of portfolio value.
    * Buys are additionally capped at the remaining settled cash (with fee buffer);
      sale proceeds are treated as unsettled and never reused within the run.
    * At most ``max_orders_per_run`` orders, prioritizing the largest drift.
    """
    total = account_state.total_value
    if total <= 0:
        return []

    threshold = Decimal(str(config.rebalance_threshold))
    max_notional = Decimal(str(config.caps.max_order_pct)) * total
    fee_multiplier = Decimal("1") + Decimal(str(config.caps.fee_buffer))

    rows = [r for r in compute_allocation(account_state, config) if r.symbol != CASH_SYMBOL]

    def _held_qty(symbol: str) -> Decimal:
        pos = account_state.get_position(symbol)
        return pos.quantity if pos else Decimal("0")

    def _is_full_exit(row: AllocationRow) -> bool:
        # A dropped/invalidated thesis targets ~0; its held shares must ALWAYS be
        # liquidated — even a sub-band dust stub (e.g. a $0.20 leftover) that the
        # rebalance band would otherwise skip and strand forever (FLUT). Keyed on
        # target_weight (not a dollar floor) so a valid small target on a tiny account
        # isn't force-sold: a dropped thesis targets exactly 0, a kept one >= ~1.75%.
        return row.target_weight <= _EXIT_WEIGHT_EPS and _held_qty(row.symbol) > 0

    # Full exits first (they derisk and must always complete regardless of the per-run
    # cap), then largest absolute drift so the most-out-of-balance holdings trade first.
    rows.sort(key=lambda r: (_is_full_exit(r), abs(r.drift)), reverse=True)

    available_cash = account_state.settled_cash
    orders: list[ProposedOrder] = []

    for row in rows:
        if len(orders) >= config.caps.max_orders_per_run:
            break
        full_exit = _is_full_exit(row)
        # Skip trivial drift — but NEVER skip a full exit, even if it is now sub-band dust.
        if abs(Decimal(str(row.drift))) <= threshold and not full_exit:
            continue
        # Target dollar value vs current dollar value.
        drift_value = Decimal(str(row.drift)) * total  # >0 overweight, <0 underweight

        if drift_value > 0:
            # Full exit: sell the ENTIRE holding as a share-QUANTITY order. Public rejects
            # a market-VALUE sell whose notional ~= the whole position ("use a quantity
            # order instead"). The quantity path is uncapped by max_order_pct AND bypasses
            # the dust floor — exiting a broken thesis derisks and must always complete.
            if full_exit:
                held_qty = _held_qty(row.symbol)
                orders.append(
                    ProposedOrder(
                        symbol=row.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=held_qty,
                        reason=(
                            f"exit {row.current_weight:.1%} -> target {row.target_weight:.1%}; "
                            f"sell all {held_qty} shares"
                        ),
                    )
                )
                continue
            # Otherwise a partial trim of the excess (never more than we hold).
            notional = min(drift_value, row.current_value, max_notional)
            notional = _round_cents(notional)
            if notional < _MIN_ORDER_NOTIONAL:
                continue
            orders.append(
                ProposedOrder(
                    symbol=row.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    notional=notional,
                    reason=(
                        f"overweight {row.current_weight:.1%} vs target "
                        f"{row.target_weight:.1%}; trim ${notional}"
                    ),
                )
            )
        else:
            # Underweight -> buy the deficit, capped by caps and available cash.
            deficit = -drift_value
            affordable = available_cash / fee_multiplier  # most we can spend incl. fees
            notional = _round_cents(min(deficit, max_notional, affordable))
            if notional < _MIN_ORDER_NOTIONAL:
                continue
            orders.append(
                ProposedOrder(
                    symbol=row.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    notional=notional,
                    reason=(
                        f"underweight {row.current_weight:.1%} vs target "
                        f"{row.target_weight:.1%}; add ${notional}"
                    ),
                )
            )
            # Reserve the spent cash (incl. fee buffer) so later buys stay affordable.
            available_cash -= notional * fee_multiplier

    return orders
