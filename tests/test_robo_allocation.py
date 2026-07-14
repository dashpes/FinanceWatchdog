"""Tests for deterministic allocation / drift math and candidate-order generation."""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.allocation import (
    compute_allocation,
    generate_candidate_orders,
)
from investment_monitor.robo.config import RoboCaps, RoboConfig
from investment_monitor.robo.gate import validate_orders
from investment_monitor.robo.models import (
    CASH_SYMBOL,
    AccountState,
    OrderSide,
    Position,
)


def make_account(*, settled_cash="100", positions=None):
    return AccountState(
        account_id="A",
        is_cash_account=True,
        has_margin=False,
        settled_cash=Decimal(settled_cash),
        positions=positions or [],
    )


def make_config(*, threshold=0.05, max_order_pct=0.25, max_orders_per_run=5):
    return RoboConfig(
        target_allocation={"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2},
        allowlist=["VOO", "SCHD"],
        rebalance_threshold=threshold,
        caps=RoboCaps(max_order_pct=max_order_pct, max_orders_per_run=max_orders_per_run),
    )


def test_compute_allocation_weights_and_cash_row():
    acct = make_account(settled_cash="50", positions=[
        Position(symbol="VOO", quantity=Decimal("0.1"), price=Decimal("500")),  # $50
    ])
    rows = {r.symbol: r for r in compute_allocation(acct, make_config())}
    assert set(rows) == {"VOO", "SCHD", CASH_SYMBOL}
    assert rows["VOO"].current_weight == 0.5
    assert rows["VOO"].drift == 0.0
    assert rows["SCHD"].current_weight == 0.0
    assert rows["SCHD"].drift == -0.3
    assert rows[CASH_SYMBOL].current_weight == 0.5
    assert rows[CASH_SYMBOL].target_weight == 0.2


def test_generate_orders_all_cash_produces_capped_buys():
    acct = make_account(settled_cash="100")
    orders = generate_candidate_orders(acct, make_config())
    assert {o.symbol for o in orders} == {"VOO", "SCHD"}
    assert all(o.side is OrderSide.BUY for o in orders)
    # Each buy is capped at max_order_pct (25% of $100 = $25).
    assert all(o.notional == Decimal("25.00") for o in orders)
    # Largest drift (VOO, -0.5) is generated first.
    assert orders[0].symbol == "VOO"


def test_generate_orders_skips_drift_below_threshold():
    # VOO at 48% vs 50% target -> 2% drift, below the 5% threshold.
    acct = make_account(settled_cash="52", positions=[
        Position(symbol="VOO", quantity=Decimal("0.48"), price=Decimal("100")),  # $48
    ])
    orders = generate_candidate_orders(acct, make_config())
    assert all(o.symbol != "VOO" for o in orders)


def test_generate_orders_trims_overweight_position():
    # VOO is 80% of a $100 portfolio; target 50% -> sell the excess (capped at 25%).
    acct = make_account(settled_cash="20", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("80")),  # $80
    ])
    orders = generate_candidate_orders(acct, make_config())
    voo = [o for o in orders if o.symbol == "VOO"][0]
    assert voo.side is OrderSide.SELL
    assert voo.notional == Decimal("25.00")  # min(drift $30, held $80, cap $25)


def test_full_exit_uses_a_share_quantity_sell():
    # A held name whose target is ~0 (thesis broke) must be exited with a QUANTITY sell of
    # the WHOLE position — Public rejects a market-value sell ~= the position value, which
    # used to strand a broken name overweight forever (ADSK/FLUT).
    cfg = make_config().model_copy(update={"target_allocation": {"VOO": 0.0, "CASH": 1.0}})
    acct = make_account(settled_cash="10", positions=[
        Position(symbol="VOO", quantity=Decimal("0.5"), price=Decimal("80")),  # $40 of $50 -> 80%
    ])
    voo = [o for o in generate_candidate_orders(acct, cfg) if o.symbol == "VOO"][0]
    assert voo.side is OrderSide.SELL
    assert voo.quantity == Decimal("0.5")   # sell ALL shares...
    assert voo.notional is None             # ...as a quantity order, not a market-value one
    assert "exit" in voo.reason.lower()


def test_small_but_valid_target_is_trimmed_not_liquidated():
    # Regression: a small (2%) BUT non-zero target on a tiny account must NOT be misread as
    # "thesis broke" and force-liquidated — only target ~0 triggers a full quantity exit.
    cfg = make_config().model_copy(update={"target_allocation": {"VOO": 0.02, "CASH": 0.98}})
    acct = make_account(settled_cash="10", positions=[
        Position(symbol="VOO", quantity=Decimal("0.5"), price=Decimal("80")),  # $40 of $50 -> 80%
    ])
    voo = [o for o in generate_candidate_orders(acct, cfg) if o.symbol == "VOO"][0]
    assert voo.side is OrderSide.SELL
    assert voo.notional is not None and voo.quantity is None  # a trim, not a full exit


def test_sub_band_dust_with_zero_target_is_still_fully_exited():
    # Regression (FLUT): a held name whose target is ~0 must be liquidated even when the
    # leftover is a tiny sub-band stub. The rebalance-band gate used to skip it BEFORE the
    # full-exit path ever ran, stranding e.g. a $0.20 position forever.
    cfg = make_config().model_copy(update={"target_allocation": {"FLUT": 0.0, "CASH": 1.0}})
    acct = make_account(settled_cash="100", positions=[
        Position(symbol="FLUT", quantity=Decimal("0.01"), price=Decimal("20")),  # $0.20 of ~$100 -> 0.2%
    ])
    flut = [o for o in generate_candidate_orders(acct, cfg) if o.symbol == "FLUT"]
    assert flut, "sub-band dust with target 0 must still produce an exit order"
    assert flut[0].side is OrderSide.SELL
    assert flut[0].quantity == Decimal("0.01")  # sell ALL shares, as a quantity order
    assert flut[0].notional is None


def test_full_exit_beats_the_per_run_cap():
    # A broken-thesis dust exit must not be starved by the per-run cap: full exits sort
    # ahead of ordinary buys/trims so derisking always completes.
    cfg = make_config().model_copy(update={
        "target_allocation": {"FLUT": 0.0, "VOO": 0.5, "SCHD": 0.5},
    })
    acct = make_account(settled_cash="100", positions=[
        Position(symbol="FLUT", quantity=Decimal("0.01"), price=Decimal("20")),  # $0.20 dust, target 0
    ])
    orders = generate_candidate_orders(acct, cfg.model_copy(update={
        "caps": RoboCaps(max_order_pct=0.25, max_orders_per_run=1),
    }))
    assert len(orders) == 1
    assert orders[0].symbol == "FLUT" and orders[0].side is OrderSide.SELL


def test_partial_trim_still_uses_a_notional_order():
    # A meaningful remaining target -> unchanged notional trim (not a full exit).
    cfg = make_config().model_copy(update={"target_allocation": {"VOO": 0.3, "CASH": 0.7}})
    acct = make_account(settled_cash="20", positions=[
        Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("80")),  # $80 of $100 -> 80%
    ])
    voo = [o for o in generate_candidate_orders(acct, cfg) if o.symbol == "VOO"][0]
    assert voo.side is OrderSide.SELL and voo.notional is not None and voo.quantity is None


def test_generate_orders_respects_per_run_cap():
    acct = make_account(settled_cash="100")
    orders = generate_candidate_orders(acct, make_config(max_orders_per_run=1))
    assert len(orders) == 1
    assert orders[0].symbol == "VOO"  # largest drift wins the single slot


def test_no_value_account_yields_no_orders():
    acct = make_account(settled_cash="0")
    assert compute_allocation(acct, make_config()) == []
    assert generate_candidate_orders(acct, make_config()) == []


def test_generated_orders_all_pass_the_gate():
    """Acceptance: the deterministic order set is affordable and gate-clean."""
    acct = make_account(settled_cash="100")
    cfg = make_config()
    orders = generate_candidate_orders(acct, cfg)
    decisions = validate_orders(orders, acct, cfg, prices={})
    assert orders, "expected some orders for an all-cash account"
    assert all(d.accepted for d in decisions), [d.reason for d in decisions if not d.accepted]
    # Total committed cash stays within the settled balance.
    total_spent = sum((o.notional for o in orders if o.side is OrderSide.BUY), Decimal("0"))
    assert total_spent <= acct.settled_cash
