"""Tests for order-fill lifecycle capture (parse, apply, backfill)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from investment_monitor.robo.broker import fill_from_order_raw
from investment_monitor.robo.rebalance import _apply_fill, _reconcile_order_fills
from investment_monitor.storage import (
    RoboOrder,
    get_filled_robo_orders,
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


def test_fill_from_order_raw_partial_with_price_is_not_terminal():
    # Finding #3: a PARTIALLY_FILLED order reports an average_price for the shares
    # filled so far but is STILL WORKING. Terminality must key on the STATUS, not the
    # mere presence of a price — otherwise the unfilled remainder is never reconciled.
    info = fill_from_order_raw(
        {"status": "PARTIALLY_FILLED", "filledQuantity": "3", "averagePrice": "99.50"}
    )
    assert info["average_price"] == Decimal("99.50")
    assert info["filled_quantity"] == Decimal("3")
    assert info["terminal"] is False  # still working despite a price being present


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


def test_apply_fill_partial_records_progress_but_keeps_polling():
    # Finding #3: a partial fill records the shares filled so far (price/qty) but is
    # NOT terminal — fill_status must stay NULL so get_unfilled_placed_orders keeps
    # polling it and the remaining shares are reconciled on a later run.
    r = _Row()
    terminal = _apply_fill(
        r,
        {"average_price": Decimal("99.50"), "filled_quantity": Decimal("3"),
         "status": "PARTIALLY_FILLED", "terminal": False},
    )
    assert terminal is False              # keep polling
    assert r.fill_price == 99.5           # progress captured
    assert r.fill_quantity == 3.0
    assert r.fill_status is None          # NULL sentinel -> still in the unfilled set


def test_partial_fill_stays_unreconciled_then_completes(tmp_path):
    # End-to-end: a PARTIALLY_FILLED poll must leave the order in the unfilled set, and
    # a later FILLED poll latches the FULL filled quantity. Without the fix the partial
    # would latch terminal and the shares that fill later would never be reconciled.
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_order(s, _placed_order(oid="o1"))

    partial = _FakeBroker(
        {"o1": {"status": "PARTIALLY_FILLED", "filledQuantity": "3", "averagePrice": "99.50"}}
    )
    with get_session() as s:
        _reconcile_order_fills(s, partial)
    with get_session() as s:
        still = get_unfilled_placed_orders(s)
        assert [r.broker_order_id for r in still] == ["o1"]  # remainder still polled
        o = get_robo_orders_for_run(s, "r1")[0]
        assert o.fill_quantity == 3.0 and o.fill_status is None  # partial progress only

    full = _FakeBroker(
        {"o1": {"status": "FILLED", "filledQuantity": "10", "averagePrice": "100.00"}}
    )
    with get_session() as s:
        _reconcile_order_fills(s, full)
    with get_session() as s:
        assert get_unfilled_placed_orders(s) == []  # now resolved
        o = get_robo_orders_for_run(s, "r1")[0]
        assert o.fill_quantity == 10.0 and o.fill_status == "FILLED"


def test_reconcile_order_fills_is_skipped_in_dry_run(tmp_path):
    # Finding #5: a DRY-RUN must be fully read-isolated from the live broker. Unfilled
    # rows left by an earlier LIVE run must NOT be polled via broker.get_order during a
    # paper run; they stay unreconciled for the next live run to pick up.
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_order(s, _placed_order(oid="o1"))

    class _SpyBroker:
        dry_run = True

        def __init__(self):
            self.get_order_calls = 0

        def get_order(self, order_id):
            self.get_order_calls += 1
            return {"status": "FILLED", "filledQuantity": "1", "averagePrice": "500.25"}

    broker = _SpyBroker()
    with get_session() as s:
        _reconcile_order_fills(s, broker)
    assert broker.get_order_calls == 0  # never touched the live broker in dry-run
    with get_session() as s:
        assert len(get_unfilled_placed_orders(s)) == 1  # left for the next live run


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


def test_get_unfilled_placed_orders_returns_all_beyond_old_cap(tmp_path):
    # Regression: more than the old hard cap (100) of unfilled orders must ALL be
    # returned (oldest-first), or the newest beyond the cap would never be reconciled.
    init_db(tmp_path / "t.db")
    n = 250  # > old cap of 100
    base = datetime(2026, 6, 22, tzinfo=timezone.utc)
    with get_session() as s:
        for i in range(n):
            o = _placed_order(oid=f"o{i:04d}")
            o.created_at = base + timedelta(seconds=i)  # deterministic oldest-first order
            save_robo_order(s, o)
    with get_session() as s:
        rows = get_unfilled_placed_orders(s)  # default: paginate until exhausted
        oids = [r.broker_order_id for r in rows]
        assert len(oids) == n  # nothing dropped past the old cap
        assert oids == sorted(oids)  # still oldest-first


def test_get_unfilled_placed_orders_respects_explicit_limit(tmp_path):
    # An explicit limit still bounds the page (oldest-first) for callers that want it.
    init_db(tmp_path / "t.db")
    base = datetime(2026, 6, 22, tzinfo=timezone.utc)
    with get_session() as s:
        for i in range(5):
            o = _placed_order(oid=f"o{i}")
            o.created_at = base + timedelta(seconds=i)
            save_robo_order(s, o)
    with get_session() as s:
        rows = get_unfilled_placed_orders(s, limit=3)
        assert [r.broker_order_id for r in rows] == ["o0", "o1", "o2"]


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


def test_get_filled_robo_orders_returns_only_the_bots_filled_trades(tmp_path):
    # Realized-P&L attribution: only the robo's OWN real, filled orders form the
    # ledger — oldest-first. Simulated paper orders and not-yet-reconciled live
    # orders are excluded so the account's manual/pending activity never counts.
    init_db(tmp_path / "t.db")
    base = datetime(2026, 6, 23, tzinfo=timezone.utc)
    with get_session() as s:
        buy = _placed_order(symbol="BORR", oid="b1")
        buy.side, buy.fill_price, buy.fill_quantity = "buy", 4.2683, 1.54332
        buy.fill_status, buy.created_at = "FILLED", base
        save_robo_order(s, buy)

        sell = _placed_order(symbol="BORR", oid="s1")
        sell.side, sell.fill_price, sell.fill_quantity = "sell", 4.3503, 1.15164
        sell.fill_status, sell.created_at = "FILLED", base + timedelta(days=3)
        save_robo_order(s, sell)

        sim = _placed_order(symbol="AAPL", oid=None)  # paper order with a (fake) fill
        sim.placed, sim.simulated = False, True
        sim.fill_price, sim.fill_quantity, sim.fill_status = 100.0, 1.0, "FILLED"
        save_robo_order(s, sim)

        save_robo_order(s, _placed_order(symbol="MSFT", oid="p1"))  # live, unfilled yet

    with get_session() as s:
        rows = get_filled_robo_orders(s)
        assert [(r.symbol, r.side) for r in rows] == [("BORR", "buy"), ("BORR", "sell")]


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
