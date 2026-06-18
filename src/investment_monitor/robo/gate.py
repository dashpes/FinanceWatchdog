"""The guardrail gate — deterministic, heavily tested order validation.

This is the heart of the safety model. The LLM (or deterministic allocator)
*proposes* orders; the gate *disposes* of them. Every proposed order must pass
``validate`` before it is ever preflighted or placed. The gate is intentionally
pure (no I/O, no SDK) so its rules can be exhaustively unit-tested.

Rejection rules (each returns a stable ``code``):
  * not_cash_account     — account isn't cash / reports margin capability
  * forbidden_field      — an options/crypto/margin/leverage/short field is present
  * bad_side             — side not in {buy, sell}
  * bad_order_type       — order_type not in {market, limit}
  * missing_limit_price  — limit order without a limit price
  * symbol_not_allowed   — symbol not on the allowlist
  * no_price             — no usable reference price to size/validate the order
  * sell_exceeds_holdings — sell quantity/notional exceeds shares held (no shorting)
  * insufficient_cash    — buy cost (incl. fee buffer) exceeds available settled cash
  * exceeds_max_order_pct — order notional exceeds max_order_pct of portfolio value
  * max_orders_per_run   — would exceed the per-run order cap
  * max_orders_per_day   — would exceed the per-day order cap

Autonomous-mode guards (Phase 4; additive — each only fires when its parameter/cap
is enabled, so rebalance mode is unaffected). Buys are restricted; SELLS are always
allowed so positions can be exited:
  * drawdown_breaker     — portfolio drawdown breaker active -> no new buys
  * no_active_thesis     — buy of a symbol with no live thesis (autonomous mode)
  * open_order_exists    — symbol already has an in-flight order at the broker
  * exceeds_per_name_cap — post-buy position value exceeds max_per_name_weight
  * max_positions        — buy of a new name would exceed the distinct-position cap
  * exceeds_turnover     — order would exceed the run's gross turnover budget

Conservative-by-construction: sale proceeds are treated as *unsettled* and are
never credited as available cash within the same run, so no sequence of accepted
orders can drive settled cash negative.
"""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.models import (
    AccountState,
    GateDecision,
    OrderSide,
    OrderType,
    ProposedOrder,
    RunCounters,
)

# Substrings in any extra/unexpected order field that mean "not a plain long equity order".
_FORBIDDEN_FIELD_TOKENS = (
    "option",
    "margin",
    "leverage",
    "short",
    "crypto",
    "borrow",
    "stop",  # stop / stop-limit orders are outside the {market, limit} allowlist
    "naked",
)

# Tolerance (in shares) when comparing a sell against held quantity, to absorb
# Decimal rounding on notional->share conversions.
_SHARE_EPSILON = Decimal("0.000001")


def _forbidden_field(order: ProposedOrder) -> str | None:
    """Return the name of the first forbidden field present, else None."""
    for key in order.extra_fields:
        lowered = key.lower()
        for token in _FORBIDDEN_FIELD_TOKENS:
            if token in lowered:
                return key
    return None


def validate(
    order: ProposedOrder,
    account_state: AccountState,
    config: RoboConfig,
    counters: RunCounters | None = None,
    *,
    price: Decimal | None,
    available_cash: Decimal | None = None,
    held_quantity: Decimal | None = None,
    active_symbols: set[str] | None = None,
    halt_buys: bool = False,
    extra_positions: int = 0,
    turnover_remaining: Decimal | None = None,
) -> GateDecision:
    """Validate a single proposed order. Pure; returns a :class:`GateDecision`.

    Args:
        order: the proposed order.
        account_state: current normalized account snapshot.
        config: robo configuration (allowlist + caps).
        counters: order counts so far (this run / today). Defaults to zero.
        price: reference last price for the symbol (used to size quantity orders).
        available_cash: cash available for THIS order; defaults to settled cash.
            The batch validator decrements this as earlier buys are accepted.
        held_quantity: shares available to sell for this symbol; defaults to the
            held quantity. The batch validator decrements this across the run.
    """
    counters = counters or RunCounters()
    if available_cash is None:
        available_cash = account_state.settled_cash
    if held_quantity is None:
        held_quantity = account_state.held_quantity(order.symbol)

    # 1. Structural account guarantee: cash account, no margin capability.
    if not account_state.is_cash_account or account_state.has_margin:
        return GateDecision.reject(
            order, "not_cash_account",
            "account is not a cash account or reports margin/borrowing capability",
        )

    # 1b. Drawdown circuit-breaker (structural): when tripped, halt all new BUYs.
    # Sells are still allowed so the portfolio can de-risk.
    if halt_buys and order.side is OrderSide.BUY:
        return GateDecision.reject(
            order, "drawdown_breaker", "portfolio drawdown breaker active: new buys halted",
        )

    # 2. No forbidden order shapes (options / crypto / margin / leverage / short / stop).
    bad_field = _forbidden_field(order)
    if bad_field is not None:
        return GateDecision.reject(
            order, "forbidden_field", f"forbidden field present on order: {bad_field!r}",
        )

    # 3. Side must be buy or sell (long-only universe).
    if order.side not in (OrderSide.BUY, OrderSide.SELL):
        return GateDecision.reject(order, "bad_side", f"invalid side: {order.side!r}")

    # 4. Order type allowlist: market or limit only.
    if order.order_type not in (OrderType.MARKET, OrderType.LIMIT):
        return GateDecision.reject(
            order, "bad_order_type", f"order_type not allowed: {order.order_type!r}",
        )
    if order.order_type is OrderType.LIMIT and order.limit_price is None:
        return GateDecision.reject(
            order, "missing_limit_price", "limit order is missing a limit price",
        )

    # 5. Symbol allowlist.
    if order.symbol not in config.allowlist:
        return GateDecision.reject(
            order, "symbol_not_allowed", f"{order.symbol} is not on the allowlist",
        )

    # 5b. Autonomous mode: a BUY requires a live thesis for the symbol. ``active_symbols``
    # is None in rebalance mode (check skipped). Sells are always allowed (exit path).
    if (
        order.side is OrderSide.BUY
        and active_symbols is not None
        and order.symbol not in active_symbols
    ):
        return GateDecision.reject(
            order, "no_active_thesis",
            f"no active thesis for {order.symbol}; buys require one in autonomous mode",
        )

    # 5c. Don't stack a new order on a symbol that already has an in-flight order at
    # the broker — prevents duplicate/queued trades across back-to-back runs. Applies
    # to both sides; inert when open_order_symbols is empty (e.g. dry-run/tests).
    if order.symbol in account_state.open_order_symbols:
        return GateDecision.reject(
            order, "open_order_exists",
            f"{order.symbol} already has an open order at the broker; skipping to avoid a duplicate",
        )

    # Need a usable reference price to size/validate quantity orders. For notional
    # orders the notional itself is the cost, so a price is not strictly required —
    # but it is still required for sell share-equivalent checks.
    has_price = price is not None and price > 0
    needs_price = order.quantity is not None or order.side is OrderSide.SELL
    if needs_price and not has_price:
        return GateDecision.reject(
            order, "no_price", f"no usable reference price for {order.symbol}",
        )

    gross_cost = order.estimated_cost(price or Decimal("0"))
    if gross_cost <= 0:
        return GateDecision.reject(order, "no_price", "could not compute a positive order notional")

    # Structural guarantees first (long-only, can't overspend), then the softer
    # size/rate caps — so the most important violation is the one reported.
    if order.side is OrderSide.SELL:
        # 7a. No shorting: cannot sell more than currently held.
        if held_quantity <= 0:
            return GateDecision.reject(
                order, "sell_exceeds_holdings", f"no shares of {order.symbol} held to sell",
            )
        sell_shares = (
            order.quantity
            if order.quantity is not None
            else (order.notional or Decimal("0")) / price  # type: ignore[operator]
        )
        if sell_shares > held_quantity + _SHARE_EPSILON:
            return GateDecision.reject(
                order, "sell_exceeds_holdings",
                f"sell of {sell_shares} {order.symbol} exceeds held {held_quantity}",
            )
    else:
        # 7b. Affordability: buy cost incl. fee buffer must fit in available cash.
        fee_multiplier = Decimal("1") + Decimal(str(config.caps.fee_buffer))
        cost_with_fees = gross_cost * fee_multiplier
        if cost_with_fees > available_cash:
            return GateDecision.reject(
                order, "insufficient_cash",
                f"buy cost {cost_with_fees} (incl. fee buffer) exceeds available cash "
                f"{available_cash}",
            )

    # 8. Max order size as a fraction of total portfolio value.
    max_notional = Decimal(str(config.caps.max_order_pct)) * account_state.total_value
    if gross_cost > max_notional:
        return GateDecision.reject(
            order, "exceeds_max_order_pct",
            f"order notional {gross_cost} exceeds {config.caps.max_order_pct:.0%} "
            f"of portfolio ({max_notional})",
        )

    # 8b. Concentration + position-count guards on BUYS (disabled by permissive defaults).
    if order.side is OrderSide.BUY:
        if config.caps.max_per_name_weight < 1.0:
            pos = account_state.get_position(order.symbol)
            held_value = pos.market_value if pos else Decimal("0")
            cap_value = Decimal(str(config.caps.max_per_name_weight)) * account_state.total_value
            if held_value + gross_cost > cap_value:
                return GateDecision.reject(
                    order, "exceeds_per_name_cap",
                    f"{order.symbol} post-buy value {held_value + gross_cost} exceeds "
                    f"{config.caps.max_per_name_weight:.0%} cap ({cap_value})",
                )
        if (
            config.caps.max_positions > 0
            and account_state.get_position(order.symbol) is None
            and len(account_state.positions) + extra_positions >= config.caps.max_positions
        ):
            return GateDecision.reject(
                order, "max_positions",
                f"buying a new name would exceed the position cap "
                f"({config.caps.max_positions})",
            )

    # 8c. Gross turnover budget for the run. BUY-only (like the other guards): a SELL
    # must never be blocked, so the portfolio can always de-risk/exit.
    if (
        order.side is OrderSide.BUY
        and turnover_remaining is not None
        and gross_cost > turnover_remaining
    ):
        return GateDecision.reject(
            order, "exceeds_turnover",
            f"order notional {gross_cost} exceeds remaining turnover budget "
            f"{turnover_remaining}",
        )

    # 9. Per-run and per-day rate caps.
    if counters.orders_this_run >= config.caps.max_orders_per_run:
        return GateDecision.reject(
            order, "max_orders_per_run",
            f"per-run order cap reached ({config.caps.max_orders_per_run})",
        )
    if counters.orders_today >= config.caps.max_orders_per_day:
        return GateDecision.reject(
            order, "max_orders_per_day",
            f"per-day order cap reached ({config.caps.max_orders_per_day})",
        )

    return GateDecision.accept(order)


def validate_orders(
    orders: list[ProposedOrder],
    account_state: AccountState,
    config: RoboConfig,
    prices: dict[str, Decimal],
    *,
    orders_today: int = 0,
    active_symbols: set[str] | None = None,
    halt_buys: bool = False,
) -> list[GateDecision]:
    """Validate a batch of orders for one run, threading shared limits safely.

    Accepted *buys* decrement the cash available to later orders (sale proceeds are
    treated as unsettled and never re-credited). Accepted *sells* decrement the
    shares available to later sells of the same symbol. Per-run/per-day caps and the
    gross-turnover budget advance only on acceptance; new distinct positions opened
    earlier in the run count against the position cap for later orders.

    ``active_symbols`` (autonomous mode) and ``halt_buys`` (drawdown breaker) are
    None/False in rebalance mode, leaving the additive guards inert.
    """
    decisions: list[GateDecision] = []
    available_cash = account_state.settled_cash
    held: dict[str, Decimal] = {p.symbol: p.quantity for p in account_state.positions}
    held_symbols = set(held)
    counters = RunCounters(orders_this_run=0, orders_today=orders_today)
    turnover_remaining = (
        Decimal(str(config.caps.max_turnover_pct)) * account_state.total_value
        if config.caps.max_turnover_pct > 0
        else None
    )
    new_names: set[str] = set()

    for order in orders:
        price = prices.get(order.symbol)
        decision = validate(
            order,
            account_state,
            config,
            counters,
            price=price,
            available_cash=available_cash,
            held_quantity=held.get(order.symbol, Decimal("0")),
            active_symbols=active_symbols,
            halt_buys=halt_buys,
            extra_positions=len(new_names),
            turnover_remaining=turnover_remaining,
        )
        decisions.append(decision)
        if decision.accepted:
            counters = RunCounters(
                orders_this_run=counters.orders_this_run + 1,
                orders_today=counters.orders_today + 1,
            )
            gross = order.estimated_cost(price or Decimal("0"))
            if order.side is OrderSide.BUY:
                # Only buys consume the turnover budget (sells are never blocked but
                # still reduce nothing the buyer relies on).
                if turnover_remaining is not None:
                    turnover_remaining -= gross
                fee_multiplier = Decimal("1") + Decimal(str(config.caps.fee_buffer))
                available_cash -= gross * fee_multiplier
                if order.symbol not in held_symbols:
                    new_names.add(order.symbol)
            else:
                sell_shares = (
                    order.quantity
                    if order.quantity is not None
                    else (order.notional or Decimal("0")) / (price or Decimal("1"))
                )
                held[order.symbol] = held.get(order.symbol, Decimal("0")) - sell_shares

    return decisions
