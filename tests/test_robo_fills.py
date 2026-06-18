"""Tests for order-fill lifecycle capture (parse, apply, backfill)."""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.broker import fill_from_order_raw
from investment_monitor.robo.rebalance import _apply_fill, _reconcile_order_fills
from investment_monitor.storage import (
    RoboOrder,
    get_robo_orders_for_run,
    get_session,
    get_unfilled_placed_orders,
    init_db,
    save_robo_order,
)


# --------------------------------------------------------------------------- #
# Parsing a get_order payload
# --------------------------------------------------------------------------- #
def test_fill_from_order_raw_filled():
    info = fill_from_order_raw({"status": "FILLED", "filledQuantity": "10", "averagePrice": "123.45"})
    assert info["average_price"] == Decimal("123.45")
    assert info["filled_quantity"] == Decimal("10")
    assert info["terminal"] is True


def test_fill_from_order_raw_rejected_terminal_without_price():
    info = fill_from_order_raw({"status": "REJECTED"})
    assert info["average_price"] is None and info["terminal"] is True


def test_fill_from_order_raw_working_not_terminal():
    info = fill_from_order_raw({"status": "NEW"})
    assert info["terminal"] is False and info["average_price"] is None


# --------------------------------------------------------------------------- #
# Applying a fill onto a row
# --------------------------------------------------------------------------- #
class _Row:
    fill_price = None
    fill_quantity = None
    fill_status = None


def test_apply_fill_filled_sets_fields():
    r = _Row()
    terminal = _apply_fill(r, {"average_price": Decimal("100"), "filled_quantity": Decimal("2"),
                               "status": "FILLED", "terminal": True})
    assert terminal is True
    assert r.fill_price == 100.0 and r.fill_quantity == 2.0 and r.fill_status == "FILLED"


def test_apply_fill_rejected_no_price():
    r = _Row()
    terminal = _apply_fill(r, {"average_price": None, "status": "REJECTED", "terminal": True})
    assert terminal is True and r.fill_price is None and r.fill_quantity == 0.0
    assert r.fill_status == "REJECTED"


def test_apply_fill_working_is_not_terminal():
    r = _Row()
    assert _apply_fill(r, {"average_price": None, "status": "NEW", "terminal": False}) is False
    assert r.fill_status is None


# --------------------------------------------------------------------------- #
# Query + backfill reconciliation
# --------------------------------------------------------------------------- #
def _placed_order(run_id="r1", symbol="VOO", oid="o1"):
    return RoboOrder(run_id=run_id, symbol=symbol, side="buy", order_type="market",
                     quantity=1.0, source="deterministic", placed=True,
                     broker_order_id=oid, status="placed")


class _FakeBroker:
    def __init__(self, responses):
        self._responses = responses

    def get_order(self, order_id):
        return self._responses[order_id]


def test_get_unfilled_placed_orders_filters(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_order(s, _placed_order(oid="o1"))            # placed + unreconciled -> yes
        sim = _placed_order(oid=None)
        sim.placed = False
        sim.simulated = True
        save_robo_order(s, sim)                                # simulated -> no
        done = _placed_order(oid="o2")
        done.fill_status = "FILLED"
        save_robo_order(s, done)                               # already reconciled -> no
    with get_session() as s:
        assert [r.broker_order_id for r in get_unfilled_placed_orders(s)] == ["o1"]


def test_reconcile_order_fills_backfills(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_order(s, _placed_order(oid="o1"))
    broker = _FakeBroker({"o1": {"status": "FILLED", "filledQuantity": "1", "averagePrice": "500.25"}})
    with get_session() as s:
        _reconcile_order_fills(s, broker)
    with get_session() as s:
        assert get_unfilled_placed_orders(s) == []  # resolved
        o = get_robo_orders_for_run(s, "r1")[0]
        assert o.fill_price == 500.25 and o.fill_status == "FILLED"


def test_reconcile_order_fills_is_fail_open(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_order(s, _placed_order(oid="o1"))

    class _BadBroker:
        def get_order(self, _order_id):
            raise RuntimeError("network down")

    with get_session() as s:
        _reconcile_order_fills(s, _BadBroker())  # must not raise
    with get_session() as s:
        assert len(get_unfilled_placed_orders(s)) == 1  # left pending for a later retry
