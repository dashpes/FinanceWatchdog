"""Tests for realized-P&L accounting and history normalization.

The accounting is a pure average-cost function over executed trades; the
normalization mirrors ``model_dump(mode="json")`` of the SDK's HistoryTransaction.
"""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.broker import trades_from_raw
from investment_monitor.robo.models import OrderSide, Trade
from investment_monitor.robo.pnl import realized_pnl


def _t(symbol, side, quantity, gross, fees="0"):
    return Trade(
        symbol=symbol,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        quantity=Decimal(quantity),
        gross=Decimal(gross),
        fees=Decimal(fees),
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
        {"type": "TRADE", "side": "SELL", "symbol": "MSFT", "quantity": "1", "principalAmount": "400"},
        {"type": "TRADE", "side": "BUY", "symbol": "", "quantity": "1", "principalAmount": "100"},  # no symbol
        {"type": "TRADE", "side": "BUY", "symbol": "AAPL", "principalAmount": "100"},  # no qty
    ]
    trades = trades_from_raw(raw)
    assert len(trades) == 1 and trades[0].symbol == "MSFT" and trades[0].side is OrderSide.SELL


def test_trades_from_raw_handles_empty():
    assert trades_from_raw(None) == []
    assert trades_from_raw([]) == []
