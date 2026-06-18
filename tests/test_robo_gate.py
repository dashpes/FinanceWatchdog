"""Exhaustive tests for the robo advisor guardrail gate.

These are the most important tests in the feature: they prove that no proposed
order can bypass the hard safety requirements. Every rejection rule and the
batch cash/holdings threading are covered.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from investment_monitor.robo.config import RoboCaps, RoboConfig
from investment_monitor.robo.gate import validate, validate_orders
from investment_monitor.robo.models import (
    AccountState,
    OrderSide,
    OrderType,
    Position,
    ProposedOrder,
    RunCounters,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #

def make_account(
    *,
    settled_cash: str = "100",
    is_cash: bool = True,
    has_margin: bool = False,
    positions: list[Position] | None = None,
) -> AccountState:
    if positions is None:
        # VOO: 0.1 share @ $500 = $50 of positions value; total value = $150.
        positions = [Position(symbol="VOO", quantity=Decimal("0.1"), price=Decimal("500"))]
    return AccountState(
        account_id="ABC123",
        account_type="BROKERAGE",
        is_cash_account=is_cash,
        has_margin=has_margin,
        settled_cash=Decimal(settled_cash),
        positions=positions,
    )


def make_config(
    *,
    allowlist: list[str] | None = None,
    max_order_pct: float = 0.25,
    max_orders_per_run: int = 5,
    max_orders_per_day: int = 10,
    fee_buffer: float = 0.01,
) -> RoboConfig:
    return RoboConfig(
        target_allocation={"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2},
        allowlist=allowlist if allowlist is not None else ["VOO", "SCHD"],
        caps=RoboCaps(
            max_order_pct=max_order_pct,
            max_orders_per_run=max_orders_per_run,
            max_orders_per_day=max_orders_per_day,
            fee_buffer=fee_buffer,
        ),
    )


def buy(symbol="VOO", *, quantity=None, notional=None, order_type=OrderType.MARKET, limit_price=None):
    return ProposedOrder(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=order_type,
        quantity=Decimal(str(quantity)) if quantity is not None else None,
        notional=Decimal(str(notional)) if notional is not None else None,
        limit_price=Decimal(str(limit_price)) if limit_price is not None else None,
    )


def sell(symbol="VOO", *, quantity=None, notional=None):
    return ProposedOrder(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal(str(quantity)) if quantity is not None else None,
        notional=Decimal(str(notional)) if notional is not None else None,
    )


PRICE = {"VOO": Decimal("500"), "SCHD": Decimal("80")}


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #

def test_accepts_valid_quantity_buy():
    d = validate(buy(quantity="0.05"), make_account(), make_config(), price=Decimal("500"))
    assert d.accepted, d.reason
    assert d.code == "accepted"


def test_accepts_valid_notional_buy_without_price():
    # Notional buys do not require a reference price.
    d = validate(buy(notional="20"), make_account(), make_config(), price=None)
    assert d.accepted, d.reason


def test_accepts_valid_sell_within_holdings():
    d = validate(sell(quantity="0.05"), make_account(), make_config(), price=Decimal("500"))
    assert d.accepted, d.reason


def test_accepts_buy_cost_exactly_equal_to_cash():
    # cost_with_fees == available cash is allowed (not strictly greater).
    acct = make_account(settled_cash="20", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("1000")),
    ])
    cfg = make_config(fee_buffer=0.0)
    d = validate(buy(notional="20"), acct, cfg, price=Decimal("500"))
    assert d.accepted, d.reason


# --------------------------------------------------------------------------- #
# Rejection rules
# --------------------------------------------------------------------------- #

def test_rejects_non_cash_account():
    d = validate(buy(notional="10"), make_account(is_cash=False), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "not_cash_account"


def test_rejects_margin_capability():
    d = validate(buy(notional="10"), make_account(has_margin=True), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "not_cash_account"


@pytest.mark.parametrize("field", ["option_type", "marginUsed", "leverage", "short_sale", "crypto", "stop_price"])
def test_rejects_forbidden_fields(field):
    order = ProposedOrder(symbol="VOO", side=OrderSide.BUY, notional=Decimal("10"),
                          extra_fields={field: "x"})
    d = validate(order, make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "forbidden_field"


def test_rejects_symbol_not_on_allowlist():
    d = validate(buy(symbol="TSLA", notional="10"), make_account(), make_config(), price=Decimal("250"))
    assert not d.accepted
    assert d.code == "symbol_not_allowed"


def test_rejects_bad_order_type_defense_in_depth():
    # The enum normally prevents this; construct via model_construct to exercise the gate branch.
    order = ProposedOrder.model_construct(
        symbol="VOO", side=OrderSide.BUY, order_type="stop", quantity=Decimal("0.01"),
        notional=None, limit_price=None, reason="", source="test", extra_fields={},
    )
    d = validate(order, make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "bad_order_type"


def test_rejects_limit_order_missing_price_defense_in_depth():
    order = ProposedOrder.model_construct(
        symbol="VOO", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("0.01"),
        notional=None, limit_price=None, reason="", source="test", extra_fields={},
    )
    d = validate(order, make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "missing_limit_price"


def test_rejects_no_price_for_quantity_buy():
    d = validate(buy(quantity="0.01"), make_account(), make_config(), price=None)
    assert not d.accepted
    assert d.code == "no_price"


def test_rejects_sell_with_no_holdings():
    d = validate(sell(symbol="SCHD", quantity="1"), make_account(), make_config(), price=PRICE["SCHD"])
    assert not d.accepted
    assert d.code == "sell_exceeds_holdings"


def test_rejects_sell_exceeding_holdings():
    # Hold 0.1 VOO, try to sell 1.0 (would be a short).
    d = validate(sell(quantity="1"), make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "sell_exceeds_holdings"


def test_rejects_notional_sell_exceeding_holdings():
    # Hold 0.1 VOO ($50 worth); try to sell $90 notional.
    d = validate(sell(notional="90"), make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "sell_exceeds_holdings"


def test_rejects_insufficient_cash():
    # Plenty of headroom under max_order_pct, but not enough settled cash.
    acct = make_account(settled_cash="10", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500")),
    ])
    d = validate(buy(notional="20"), acct, make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "insufficient_cash"


def test_fee_buffer_makes_marginal_buy_unaffordable():
    acct = make_account(settled_cash="20", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("1000")),
    ])
    # notional 20 with 1% fee buffer = 20.20 > 20 available.
    d = validate(buy(notional="20"), acct, make_config(fee_buffer=0.01), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "insufficient_cash"


def test_rejects_exceeds_max_order_pct():
    # total_value = 150, max_order_pct 0.25 -> max notional 37.5; 50 exceeds it.
    d = validate(buy(notional="50"), make_account(), make_config(), price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "exceeds_max_order_pct"


def test_rejects_max_orders_per_run():
    counters = RunCounters(orders_this_run=5, orders_today=5)
    d = validate(buy(notional="10"), make_account(), make_config(), counters, price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "max_orders_per_run"


def test_rejects_max_orders_per_day():
    counters = RunCounters(orders_this_run=0, orders_today=10)
    d = validate(buy(notional="10"), make_account(), make_config(), counters, price=PRICE["VOO"])
    assert not d.accepted
    assert d.code == "max_orders_per_day"


# --------------------------------------------------------------------------- #
# Batch validation: shared cash / holdings threading
# --------------------------------------------------------------------------- #

def test_batch_threads_available_cash_across_buys():
    acct = make_account(settled_cash="100", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("1000")),
    ])
    cfg = make_config(max_order_pct=0.5)  # max notional = 0.5 * 1100 = 550 (not the binding limit)
    orders = [buy(notional="60"), buy(notional="60")]
    decisions = validate_orders(orders, acct, cfg, PRICE)
    assert decisions[0].accepted
    # First buy consumes 60.6; only 39.4 remains, so the second is unaffordable.
    assert not decisions[1].accepted
    assert decisions[1].code == "insufficient_cash"


def test_batch_does_not_credit_unsettled_sale_proceeds():
    # Selling does not free cash for buys within the same run.
    acct = make_account(settled_cash="10", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500")),
    ])
    cfg = make_config(max_order_pct=1.0)
    orders = [sell(quantity="1"), buy(symbol="SCHD", notional="100")]
    decisions = validate_orders(orders, acct, cfg, PRICE)
    assert decisions[0].accepted  # sell is fine
    assert not decisions[1].accepted  # buy still constrained to the $10 settled cash
    assert decisions[1].code == "insufficient_cash"


def test_batch_threads_holdings_across_sells():
    acct = make_account(positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500")),
    ])
    cfg = make_config(max_order_pct=1.0)  # isolate holdings threading from the size cap
    orders = [sell(quantity="0.6"), sell(quantity="0.6")]
    decisions = validate_orders(orders, acct, cfg, PRICE)
    assert decisions[0].accepted
    assert not decisions[1].accepted  # only 0.4 left after the first sell
    assert decisions[1].code == "sell_exceeds_holdings"


def test_batch_respects_per_run_cap():
    cfg = make_config(max_orders_per_run=2, max_order_pct=1.0)
    acct = make_account(settled_cash="1000")
    orders = [buy(notional="10"), buy(notional="10"), buy(notional="10")]
    decisions = validate_orders(orders, acct, cfg, PRICE)
    assert [d.accepted for d in decisions] == [True, True, False]
    assert decisions[2].code == "max_orders_per_run"


def test_no_sequence_can_drive_cash_negative():
    """Property: accepted buys never sum past settled cash (the core invariant)."""
    settled = Decimal("100")
    acct = make_account(settled_cash="100", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("1000")),
    ])
    cfg = make_config(max_order_pct=1.0, max_orders_per_run=100, max_orders_per_day=100)
    # Twenty $10 buys = far more than $100 of intent.
    orders = [buy(notional="10") for _ in range(20)]
    decisions = validate_orders(orders, acct, cfg, PRICE)
    accepted_buy_cost = sum(
        (d.order.estimated_cost(PRICE["VOO"]) * Decimal("1.01") for d in decisions if d.accepted),
        Decimal("0"),
    )
    assert accepted_buy_cost <= settled
    assert any(not d.accepted for d in decisions)
