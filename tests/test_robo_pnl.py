"""Tests for realized-P&L accounting and history normalization.

The accounting is a pure average-cost function over executed trades; the
normalization mirrors ``model_dump(mode="json")`` of the SDK's HistoryTransaction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from investment_monitor.robo.broker import trades_from_raw
from investment_monitor.robo.models import OrderSide, Trade
from investment_monitor.robo.pnl import realized_pnl, trades_from_fills


def _t(symbol, side, quantity, gross, fees="0", timestamp=None):
    return Trade(
        symbol=symbol,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        quantity=Decimal(quantity),
        gross=Decimal(gross),
        fees=Decimal(fees),
        timestamp=timestamp,
    )


# --------------------------------------------------------------------------- #
# Realized P&L accounting (pure, average-cost)
# --------------------------------------------------------------------------- #
def test_buy_then_full_sell_at_profit_includes_fees_in_basis():
    trades = [_t("VOO", "buy", "10", "1000", "1"), _t("VOO", "sell", "10", "1200", "1")]
    rp = realized_pnl(trades)
    # basis = (1000 + 1)/10 = 100.1; proceeds = 1200 - 1 = 1199; realized = 1199 - 1001
    assert rp.symbol_realized("VOO") == Decimal("198")
    assert rp.per_symbol["VOO"].quantity == 0
    assert rp.total_fees == Decimal("2")


def test_partial_sell_leaves_remaining_lot():
    rp = realized_pnl([_t("VOO", "buy", "10", "1000"), _t("VOO", "sell", "4", "600")])
    assert rp.symbol_realized("VOO") == Decimal("200")  # 600 - 100*4
    sp = rp.per_symbol["VOO"]
    assert sp.quantity == Decimal("6") and sp.avg_cost == Decimal("100")


def test_multiple_buys_average_cost():
    trades = [_t("X", "buy", "10", "1000"), _t("X", "buy", "10", "2000"), _t("X", "sell", "5", "1500")]
    rp = realized_pnl(trades)
    # avg = 3000/20 = 150; realized = 1500 - 150*5 = 750
    assert rp.symbol_realized("X") == Decimal("750")
    assert rp.per_symbol["X"].quantity == Decimal("15")
    assert rp.per_symbol["X"].avg_cost == Decimal("150")


def test_open_position_has_zero_realized():
    rp = realized_pnl([_t("AAPL", "buy", "3", "600")])
    assert rp.symbol_realized("AAPL") == Decimal("0")
    assert rp.per_symbol["AAPL"].quantity == Decimal("3")
    assert rp.total_realized == Decimal("0")


def test_sell_more_than_held_realizes_only_matched_portion():
    # Hold 5, sell 10 -> only the 5 we actually hold realize; no phantom gain.
    rp = realized_pnl([_t("V", "buy", "5", "500"), _t("V", "sell", "10", "1200")])
    # matched 5/10 -> proceeds 600; realized = 600 - 100*5 = 100
    assert rp.symbol_realized("V") == Decimal("100")
    assert rp.per_symbol["V"].quantity == 0


def test_sell_with_no_holding_books_nothing():
    rp = realized_pnl([_t("GS", "sell", "10", "1200")])
    assert rp.symbol_realized("GS") == Decimal("0")
    assert rp.per_symbol["GS"].quantity == 0


def test_totals_across_symbols():
    rp = realized_pnl([
        _t("A", "buy", "10", "1000"), _t("A", "sell", "10", "1100"),  # +100
        _t("B", "buy", "10", "1000"), _t("B", "sell", "10", "900"),   # -100
    ])
    assert rp.total_realized == Decimal("0")
    assert rp.symbol_realized("A") == Decimal("100")
    assert rp.symbol_realized("B") == Decimal("-100")


# --------------------------------------------------------------------------- #
# Chronological ordering (mixed / absent timestamps must not crash the sort)
# --------------------------------------------------------------------------- #
def test_sort_does_not_crash_on_mixed_timestamps_and_keeps_chronology():
    # Input order is reverse-chronological for the timestamped trades: the SELL is
    # listed BEFORE its matching BUY. The buggy fallback (TypeError -> raw order)
    # would realize the SELL against an empty holding and book nothing; the correct
    # behavior sorts by timestamp so the BUY is processed first.
    sell = _t("VOO", "sell", "10", "1200", timestamp=datetime(2026, 6, 2))
    buy = _t("VOO", "buy", "10", "1000", timestamp=datetime(2026, 6, 1))
    no_ts = _t("AAPL", "buy", "1", "100")  # mixes a None timestamp into the list
    rp = realized_pnl([sell, no_ts, buy])
    # BUY before SELL: realized = 1200 - 100*10 = 200, position flat.
    assert rp.symbol_realized("VOO") == Decimal("200")
    assert rp.per_symbol["VOO"].quantity == 0
    # The timestamp-less trade is still accounted for (pushed to the end, not dropped).
    assert rp.per_symbol["AAPL"].quantity == Decimal("1")


def test_sort_handles_mixed_tz_aware_and_naive_timestamps():
    # Live trigger: the broker history mixes tz-aware and tz-naive datetimes.
    # Python forbids comparing aware vs naive, so the prior sort key raised
    # TypeError -> the caller's broad except silently dropped realized P&L.
    # The BUY (naive) must be ordered before the SELL (aware) so the SELL
    # realizes against the reconstructed lot rather than an empty holding.
    buy_naive = _t("VOO", "buy", "10", "1000", timestamp=datetime(2026, 6, 1))
    sell_aware = _t(
        "VOO", "sell", "10", "1200", timestamp=datetime(2026, 6, 2, tzinfo=timezone.utc)
    )
    no_ts = _t("AAPL", "buy", "1", "100")  # also mix in a None timestamp
    # Feed reverse-chronological so the only way to get the right answer is a
    # correct sort (not preserved input order).
    rp = realized_pnl([sell_aware, no_ts, buy_naive])
    assert rp.symbol_realized("VOO") == Decimal("200")  # 1200 - 100*10
    assert rp.per_symbol["VOO"].quantity == 0
    assert rp.per_symbol["AAPL"].quantity == Decimal("1")


def test_sort_handles_aware_naive_across_the_naive_utc_boundary():
    # A naive timestamp is assumed UTC; verify the chronology is correct when an
    # aware and a naive timestamp straddle midnight UTC so ordering actually
    # depends on the normalization (not just the calendar date).
    buy_aware = _t(
        "X", "buy", "10", "1000", timestamp=datetime(2026, 6, 1, 23, 0, tzinfo=timezone.utc)
    )
    sell_naive = _t("X", "sell", "10", "1300", timestamp=datetime(2026, 6, 2, 1, 0))
    rp = realized_pnl([sell_naive, buy_aware])  # reversed on input
    assert rp.symbol_realized("X") == Decimal("300")  # 1300 - 100*10
    assert rp.per_symbol["X"].quantity == 0


def test_all_timestamps_present_orders_chronologically():
    trades = [
        _t("X", "sell", "5", "1500", timestamp=datetime(2026, 6, 3)),
        _t("X", "buy", "10", "2000", timestamp=datetime(2026, 6, 2)),
        _t("X", "buy", "10", "1000", timestamp=datetime(2026, 6, 1)),
    ]
    rp = realized_pnl(trades)
    # Two buys average to 3000/20 = 150; sell 5 -> realized = 1500 - 150*5 = 750.
    assert rp.symbol_realized("X") == Decimal("750")
    assert rp.per_symbol["X"].quantity == Decimal("15")


def test_all_timestamps_absent_trusts_input_order():
    # No timestamps anywhere: input order must be preserved (BUY then SELL here).
    rp = realized_pnl([_t("V", "buy", "5", "500"), _t("V", "sell", "5", "600")])
    assert rp.symbol_realized("V") == Decimal("100")
    assert rp.per_symbol["V"].quantity == 0


# --------------------------------------------------------------------------- #
# History normalization
# --------------------------------------------------------------------------- #
def test_trades_from_raw_keeps_trades_only():
    raw = [
        {"type": "TRADE", "side": "BUY", "symbol": "VOO", "quantity": "2",
         "principal_amount": "1000", "fees": "1.50"},
        {"type": "MONEY_MOVEMENT", "sub_type": "DEPOSIT", "net_amount": "50"},
        {"type": "TRADE", "sub_type": "DIVIDEND", "symbol": "VOO", "net_amount": "3"},  # no side
    ]
    trades = trades_from_raw(raw)
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "VOO" and t.side is OrderSide.BUY
    assert t.quantity == Decimal("2") and t.gross == Decimal("1000") and t.fees == Decimal("1.50")


def test_trades_from_raw_camelcase_and_skips_incomplete():
    raw = [
        # A real Public SELL: quantity is NEGATIVE, principal POSITIVE (cash in).
        {"type": "TRADE", "side": "SELL", "symbol": "MSFT", "quantity": "-1", "principalAmount": "400"},
        {"type": "TRADE", "side": "BUY", "symbol": "", "quantity": "1", "principalAmount": "100"},  # no symbol
        {"type": "TRADE", "side": "BUY", "symbol": "AAPL", "principalAmount": "100"},  # no qty
    ]
    trades = trades_from_raw(raw)
    assert len(trades) == 1 and trades[0].symbol == "MSFT" and trades[0].side is OrderSide.SELL


def test_trades_from_raw_keeps_negative_quantity_sells():
    # Regression: Public reports SELL quantity as negative and BUY principal as
    # negative (it signs by cash flow, not magnitude). A ``qty <= 0`` guard dropped
    # every sell, so realized P&L was permanently $0 on a live account that had
    # actually sold. The side carries the direction; magnitudes must be abs'd.
    raw = [
        {"type": "TRADE", "side": "BUY", "symbol": "BORR", "quantity": "1.54332",
         "principal_amount": "-6.5899764", "fees": "0", "timestamp": "2026-06-23T14:02:18Z"},
        {"type": "TRADE", "side": "SELL", "symbol": "BORR", "quantity": "-1.15164",
         "principal_amount": "5.009979492", "fees": "0.02", "timestamp": "2026-06-26T14:02:45Z"},
    ]
    trades = trades_from_raw(raw)
    assert [t.side for t in trades] == [OrderSide.BUY, OrderSide.SELL]
    sell = trades[1]
    assert sell.quantity == Decimal("1.15164")  # abs'd to a positive magnitude
    assert sell.gross == Decimal("5.009979492")
    assert sell.fees == Decimal("0.02")
    # End-to-end: the sell now realizes P&L instead of being silently dropped.
    rp = realized_pnl(trades)
    assert rp.symbol_realized("BORR") != Decimal("0")


def test_trades_from_raw_handles_empty():
    assert trades_from_raw(None) == []
    assert trades_from_raw([]) == []


# --------------------------------------------------------------------------- #
# Own-ledger attribution: realized P&L from the bot's OWN filled orders
# --------------------------------------------------------------------------- #
class _FilledOrder:
    """Minimal stand-in for a RoboOrder row carrying a broker-reconciled fill."""

    def __init__(self, symbol, side, fill_price, fill_quantity, created_at=None):
        self.symbol = symbol
        self.side = side
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.created_at = created_at


def test_trades_from_fills_maps_side_and_computes_gross():
    orders = [
        _FilledOrder("AAPL", "buy", 298.26, 0.01274),
        _FilledOrder("AAPL", "sell", 294.855, 0.00932),
    ]
    buy, sell = trades_from_fills(orders)
    assert buy.side is OrderSide.BUY and sell.side is OrderSide.SELL
    assert buy.symbol == "AAPL" and buy.quantity == Decimal("0.01274")
    assert buy.gross == Decimal("298.26") * Decimal("0.01274")  # gross = fill_price * qty
    assert sell.gross == Decimal("294.855") * Decimal("0.00932")
    assert buy.fees == Decimal("0") and sell.fees == Decimal("0")  # fees not stored per order


def test_trades_from_fills_skips_rows_without_a_usable_fill():
    orders = [
        _FilledOrder("MSFT", "buy", None, None),    # never filled
        _FilledOrder("MSFT", "buy", 375.0, None),   # half-missing
        _FilledOrder("MSFT", "buy", 375.0, 0.0),    # zero quantity
        _FilledOrder("", "buy", 375.0, 0.01),       # no symbol
        _FilledOrder("MSFT", "buy", 375.0, 0.01),   # the one valid row
    ]
    trades = trades_from_fills(orders)
    assert len(trades) == 1 and trades[0].symbol == "MSFT"


def test_trades_from_fills_realized_only_sees_bot_symbols():
    # The ledger holds ONLY the bot's own orders, so a manual holding the bot never
    # traded (e.g. MSTY) is simply absent and contributes nothing to realized P&L.
    orders = [
        _FilledOrder("BORR", "buy", 4.2683, 1.54332, datetime(2026, 6, 23)),
        _FilledOrder("BORR", "sell", 4.3503, 1.15164, datetime(2026, 6, 26)),
    ]
    rp = realized_pnl(trades_from_fills(orders))
    assert "MSTY" not in rp.per_symbol
    assert rp.symbol_realized("BORR") != Decimal("0")
