"""Tests for the broker's raw-payload -> AccountState mapping.

These use plain dicts shaped exactly like ``model_dump(mode="json")`` of the real
SDK models (verified against publicdotcom-py 0.1.16), so they protect the
safety-critical cash-vs-margin detection without importing the SDK.
"""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.broker import account_state_from_raw


def _portfolio(cash="100.00", positions=None):
    return {
        "account_id": "ACC1",
        "account_type": "BROKERAGE",
        "buying_power": {
            "cash_only_buying_power": cash,
            "buying_power": cash,
            "options_buying_power": "0",
        },
        "equity": [],
        "orders": [],
        "positions": positions if positions is not None else [],
    }


def _position(symbol="VOO", quantity="0.1", current_value="50.00", last_price="500.00"):
    pos = {
        "instrument": {"symbol": symbol, "name": symbol, "type": "EQUITY"},
        "quantity": quantity,
        "current_value": current_value,
    }
    if last_price is not None:
        pos["last_price"] = {"last_price": last_price}
    return pos


def test_detects_cash_account():
    state = account_state_from_raw(
        {"account_id": "ACC1", "brokerage_account_type": "CASH"}, _portfolio()
    )
    assert state.is_cash_account is True
    assert state.has_margin is False
    assert state.settled_cash == Decimal("100.00")


def test_detects_margin_account():
    state = account_state_from_raw(
        {"account_id": "ACC2", "brokerage_account_type": "MARGIN"}, _portfolio()
    )
    assert state.is_cash_account is False
    assert state.has_margin is True


def test_unknown_brokerage_type_is_fail_safe_noncash():
    # Missing/unknown type must NOT be treated as cash (refuse rather than assume safe).
    state = account_state_from_raw({"account_id": "ACC3"}, _portfolio())
    assert state.is_cash_account is False
    assert state.has_margin is False


def test_settled_cash_from_buying_power():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"}, _portfolio(cash="42.50")
    )
    assert state.settled_cash == Decimal("42.50")


def test_position_price_from_last_price():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[_position(quantity="2", current_value="1000", last_price="500")]),
    )
    pos = state.get_position("VOO")
    assert pos is not None
    assert pos.price == Decimal("500")
    assert pos.quantity == Decimal("2")


def test_position_price_derived_from_current_value_when_no_last_price():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[_position(quantity="4", current_value="200", last_price=None)]),
    )
    pos = state.get_position("VOO")
    assert pos is not None
    assert pos.price == Decimal("50")  # 200 / 4


def test_total_value_is_cash_plus_positions():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(cash="100", positions=[_position(quantity="0.1", current_value="50", last_price="500")]),
    )
    assert state.total_value == Decimal("150")


def _order(symbol="VOO", status="NEW"):
    return {"order_id": "o1", "instrument": {"symbol": symbol, "type": "EQUITY"}, "status": status}


def test_open_order_symbols_extracted_only_for_working_orders():
    portfolio = _portfolio()
    portfolio["orders"] = [
        _order("VOO", "NEW"),                # open
        _order("MSFT", "PARTIALLY_FILLED"),  # open
        _order("AAPL", "FILLED"),            # terminal -> excluded
        _order("GS", "CANCELLED"),           # terminal -> excluded
        _order("HD", "REJECTED"),            # terminal -> excluded
    ]
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"}, portfolio
    )
    assert state.open_order_symbols == ["MSFT", "VOO"]  # sorted, working orders only


def test_no_open_orders_default_empty():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"}, _portfolio()
    )
    assert state.open_order_symbols == []
