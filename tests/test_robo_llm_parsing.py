"""Tests for robust parsing of LLM rebalance proposals."""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.robo.llm import parse_orders
from investment_monitor.robo.models import OrderSide, OrderType


def test_parses_clean_json_array():
    text = '[{"symbol": "VOO", "side": "buy", "notional": 25, "reason": "underweight"}]'
    orders = parse_orders(text)
    assert len(orders) == 1
    o = orders[0]
    assert o.symbol == "VOO"
    assert o.side is OrderSide.BUY
    assert o.notional == Decimal("25")
    assert o.source == "llm"


def test_parses_markdown_fenced_json():
    text = '```json\n[{"symbol": "schd", "side": "SELL", "quantity": "0.5"}]\n```'
    orders = parse_orders(text)
    assert len(orders) == 1
    assert orders[0].symbol == "SCHD"  # upper-cased
    assert orders[0].side is OrderSide.SELL  # case-insensitive
    assert orders[0].quantity == Decimal("0.5")


def test_extracts_array_from_noisy_text():
    text = 'Sure! Here are my proposals:\n[{"symbol": "VOO", "side": "buy", "amount": 10}]\nHope that helps.'
    orders = parse_orders(text)
    assert len(orders) == 1
    assert orders[0].notional == Decimal("10")  # "amount" alias


def test_parses_single_object_not_array():
    text = '{"symbol": "VOO", "side": "buy", "notional": 5}'
    orders = parse_orders(text)
    assert len(orders) == 1


def test_strips_currency_and_commas_in_numbers():
    text = '[{"symbol": "VOO", "side": "buy", "notional": "$1,250.50"}]'
    orders = parse_orders(text)
    assert orders[0].notional == Decimal("1250.50")


def test_limit_order_with_price():
    text = '[{"symbol": "VOO", "side": "buy", "quantity": 1, "order_type": "limit", "limit_price": 500}]'
    orders = parse_orders(text)
    assert orders[0].order_type is OrderType.LIMIT
    assert orders[0].limit_price == Decimal("500")


def test_forbidden_field_preserved_for_gate():
    # An options field must survive parsing so the gate can reject it.
    text = '[{"symbol": "VOO", "side": "buy", "notional": 10, "option_type": "call"}]'
    orders = parse_orders(text)
    assert len(orders) == 1
    assert orders[0].extra_fields.get("option_type") == "call"


def test_skips_malformed_entries():
    text = """[
        {"symbol": "VOO", "side": "buy", "notional": 10, "quantity": 1},
        {"symbol": "SCHD", "side": "buy"},
        {"side": "buy", "notional": 5},
        {"symbol": "VOO", "side": "hodl", "notional": 5},
        {"symbol": "VOO", "side": "sell", "quantity": 2}
    ]"""
    orders = parse_orders(text)
    # Only the last (well-formed) entry survives.
    assert len(orders) == 1
    assert orders[0].side is OrderSide.SELL
    assert orders[0].quantity == Decimal("2")


def test_limit_order_missing_price_is_skipped():
    text = '[{"symbol": "VOO", "side": "buy", "quantity": 1, "order_type": "limit"}]'
    assert parse_orders(text) == []


def test_empty_and_garbage_input():
    assert parse_orders("") == []
    assert parse_orders("no json here at all") == []
    assert parse_orders("[]") == []
