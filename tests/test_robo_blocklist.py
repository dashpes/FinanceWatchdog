"""Tests for the smart blocklist module and its enforcement in the gate."""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo import blocklist as bl
from investment_monitor.robo.config import RoboCaps, RoboConfig
from investment_monitor.robo.gate import validate, validate_orders
from investment_monitor.robo.models import (
    AccountState,
    OrderSide,
    OrderType,
    Position,
    ProposedOrder,
)


# --------------------------------------------------------------------------- #
# blocklist module
# --------------------------------------------------------------------------- #

def _db(tmp_path):
    return str(tmp_path / "portfolio.db")  # file need not exist


def test_load_learned_empty_when_no_file(tmp_path):
    assert bl.load_learned(_db(tmp_path)) == set()


def test_add_and_load_roundtrip(tmp_path):
    db = _db(tmp_path)
    assert bl.add_learned(db, "cbkm", "only available when closing") is True
    assert bl.load_learned(db) == {"CBKM"}  # uppercased
    # file lands next to the DB
    assert (tmp_path / "robo_blocklist.json").exists()


def test_add_is_idempotent(tmp_path):
    db = _db(tmp_path)
    assert bl.add_learned(db, "CBKM") is True
    assert bl.add_learned(db, "cbkm") is False  # already present
    assert bl.load_learned(db) == {"CBKM"}


def test_add_accumulates_multiple(tmp_path):
    db = _db(tmp_path)
    bl.add_learned(db, "CBKM")
    bl.add_learned(db, "FOO")
    assert bl.load_learned(db) == {"CBKM", "FOO"}


def test_add_blank_symbol_is_noop(tmp_path):
    db = _db(tmp_path)
    assert bl.add_learned(db, "   ") is False
    assert bl.load_learned(db) == set()


def test_is_unbuyable_message():
    assert bl.is_unbuyable_message(
        "API Error 400: This asset is currently only available for trading when "
        "closing an existing position on Public."
    )
    assert bl.is_unbuyable_message("Security is HALTED")
    assert bl.is_unbuyable_message("pending delisting")
    assert not bl.is_unbuyable_message("insufficient buying power")
    assert not bl.is_unbuyable_message(None)
    assert not bl.is_unbuyable_message("")


def test_load_failopen_on_garbage_file(tmp_path):
    db = _db(tmp_path)
    (tmp_path / "robo_blocklist.json").write_text("{ not json")
    assert bl.load_learned(db) == set()  # logged + ignored, never raises


# --------------------------------------------------------------------------- #
# gate enforcement
# --------------------------------------------------------------------------- #

def _account(positions=None, cash="100"):
    if positions is None:
        positions = [Position(symbol="VOO", quantity=Decimal("0.1"), price=Decimal("500"))]
    return AccountState(
        account_id="ABC", account_type="BROKERAGE", is_cash_account=True,
        has_margin=False, settled_cash=Decimal(cash), positions=positions,
    )


def _config(*, allowlist, max_positions=0):
    return RoboConfig(
        target_allocation={"VOO": 0.8, "CASH": 0.2},
        allowlist=allowlist,
        caps=RoboCaps(max_order_pct=0.9, max_positions=max_positions),
    )


def _buy(symbol, notional="10"):
    return ProposedOrder(symbol=symbol, side=OrderSide.BUY,
                         order_type=OrderType.MARKET, notional=Decimal(notional))


def _sell(symbol, quantity="0.05"):
    return ProposedOrder(symbol=symbol, side=OrderSide.SELL,
                         order_type=OrderType.MARKET, quantity=Decimal(quantity))


def test_blocklisted_buy_rejected():
    d = validate(
        _buy("CBKM"), _account(), _config(allowlist=["VOO", "CBKM"]),
        price=Decimal("5"), blocklist={"CBKM"},
    )
    assert not d.accepted
    assert d.code == "blocklisted"


def test_blocklisted_sell_is_allowed():
    # A blocklisted name already held can always be exited.
    d = validate(
        _sell("VOO"), _account(), _config(allowlist=["VOO"]),
        price=Decimal("500"), blocklist={"VOO"},
    )
    assert d.code != "blocklisted"
    assert d.accepted, d.reason


def test_blocklist_none_has_no_effect():
    d = validate(
        _buy("CBKM"), _account(), _config(allowlist=["VOO", "CBKM"]),
        price=Decimal("5"), blocklist=None,
    )
    assert d.code != "blocklisted"


def test_blocklisted_buy_does_not_consume_a_position_slot():
    """The slot-waste fix: a blocklisted higher-priority name must not steal the
    one open slot from a buyable lower-priority name."""
    account = _account()  # holds VOO -> 1 position
    config = _config(allowlist=["VOO", "CBKM", "GOOD"], max_positions=2)  # 1 slot free
    orders = [_buy("CBKM"), _buy("GOOD")]  # CBKM ranked first, but blocklisted

    decisions = validate_orders(orders, account, config, prices={}, blocklist={"CBKM"})
    assert decisions[0].code == "blocklisted"
    assert decisions[1].accepted, decisions[1].reason  # GOOD still gets the slot

    # Sanity: without the blocklist, CBKM would take the slot and GOOD be rejected.
    no_bl = validate_orders(orders, account, config, prices={}, blocklist=None)
    assert no_bl[0].accepted
    assert no_bl[1].code == "max_positions"
