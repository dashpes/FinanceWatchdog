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


def _position_cb(symbol="VOO", quantity="2", last_price="500",
                 unit_cost="450", gain_value="100.00", gain_percentage="11.11"):
    """A position carrying Public's CostBasis sub-object."""
    pos = {
        "instrument": {"symbol": symbol, "name": symbol, "type": "EQUITY"},
        "quantity": quantity,
        "last_price": {"last_price": last_price},
        "cost_basis": {
            "unit_cost": unit_cost,
            "total_cost": str(Decimal(unit_cost) * Decimal(quantity)),
            "gain_value": gain_value,
            "gain_percentage": gain_percentage,
        },
    }
    return pos


def test_cost_basis_parsed_into_position():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[_position_cb(quantity="2", last_price="500",
                                           unit_cost="450", gain_value="100.00")]),
    )
    pos = state.get_position("VOO")
    assert pos.unit_cost == Decimal("450")
    assert pos.unrealized_gain == Decimal("100.00")
    assert pos.cost_basis_value == Decimal("900")
    assert pos.unrealized_return == Decimal("500") / Decimal("450") - 1


def test_negative_unit_cost_payload_does_not_block_snapshot():
    # Finding #4: a quirky cost-basis payload (negative unitCost) must NOT raise a
    # ValidationError inside account_state_from_raw — that would fail the whole account
    # snapshot and refuse trading for the run. It is sanitized to None ("unknown"),
    # while every other field is preserved so the snapshot (and trading) proceeds.
    pos_raw = {
        "instrument": {"symbol": "VOO", "type": "EQUITY"},
        "quantity": "2",
        "last_price": {"last_price": "500"},
        "cost_basis": {"unit_cost": "-450", "total_cost": "-900"},  # bad/odd payload
    }
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[pos_raw]),
    )
    pos = state.get_position("VOO")
    assert pos is not None                      # snapshot built, run not refused
    assert pos.unit_cost is None                # bad basis treated as unknown
    assert pos.cost_basis_value is None
    assert pos.unrealized_return is None
    assert pos.price == Decimal("500")          # other fields intact
    assert pos.quantity == Decimal("2")
    assert state.is_cash_account is True


def test_negative_derived_unit_cost_from_total_cost_is_unknown():
    # Even when unit_cost is derived from a negative totalCost it must clamp to None
    # rather than produce a negative basis that distorts P&L / return math.
    pos_raw = {
        "instrument": {"symbol": "MSFT", "type": "EQUITY"},
        "quantity": "4",
        "last_price": {"last_price": "100"},
        "cost_basis": {"total_cost": "-320"},  # no unitCost; derived would be -80
    }
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[pos_raw]),
    )
    assert state.get_position("MSFT").unit_cost is None


def test_position_model_sanitizes_negative_unit_cost_without_raising():
    # Belt-and-suspenders at the model layer: constructing a Position directly with a
    # negative unit_cost sanitizes it to None instead of raising.
    from investment_monitor.robo.models import Position

    p = Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500"),
                 unit_cost=Decimal("-3"))
    assert p.unit_cost is None
    # A valid (>= 0) basis is preserved unchanged.
    p2 = Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500"),
                  unit_cost=Decimal("450"))
    assert p2.unit_cost == Decimal("450")


def test_unit_cost_derived_from_total_cost_when_unit_absent():
    pos_raw = {
        "instrument": {"symbol": "MSFT", "type": "EQUITY"},
        "quantity": "4",
        "last_price": {"last_price": "100"},
        "cost_basis": {"total_cost": "320", "gain_value": "80"},  # no unitCost
    }
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[pos_raw]),
    )
    assert state.get_position("MSFT").unit_cost == Decimal("80")  # 320 / 4


def test_position_without_cost_basis_is_none_not_zero():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[_position(quantity="2", current_value="1000", last_price="500")]),
    )
    pos = state.get_position("VOO")
    assert pos.unit_cost is None
    assert pos.unrealized_gain is None
    assert pos.unrealized_return is None
    assert pos.cost_basis_value is None


def test_total_unrealized_gain_aggregates_and_signs():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[
            _position_cb(symbol="VOO", gain_value="100.00"),
            _position_cb(symbol="MSFT", gain_value="-30.00"),
        ]),
    )
    assert state.total_unrealized_gain == Decimal("70.00")


def test_total_unrealized_gain_none_when_no_basis_reported():
    state = account_state_from_raw(
        {"account_id": "A", "brokerage_account_type": "CASH"},
        _portfolio(positions=[_position(quantity="2", current_value="1000", last_price="500")]),
    )
    assert state.total_unrealized_gain is None
    assert state.total_cost_basis is None


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
