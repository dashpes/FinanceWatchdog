"""Tests for the discovery scorer's derived inputs (momentum / insider / news)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from investment_monitor.research.discovery import (
    compute_momentum_inputs,
    summarize_insider_activity,
    summarize_recent_news,
)

TODAY = date.today()


@dataclass
class _P:
    date: date
    close: float | None


@dataclass
class _Txn:
    raw_code: str
    total_value: float | None


@dataclass
class _News:
    headline: str
    relevance_score: float | None


def _series(days: int, start: float, end: float) -> list[_P]:
    """Newest-first linear price series covering ``days`` calendar days."""
    out = []
    for i in range(days):
        frac = i / max(1, days - 1)
        out.append(_P(date=TODAY - timedelta(days=i), close=end - (end - start) * frac))
    return out


def test_momentum_full_history():
    prices = _series(400, start=10.0, end=20.0)  # steadily rising
    m = compute_momentum_inputs(prices, today=TODAY)
    assert m["price_change_1m"] > 0
    assert m["price_change_1y"] > 0
    assert m["price_change_1y"] > m["price_change_1m"]
    assert m["rsi"] == 100.0                      # monotonic gains
    assert abs(m["vs_52w_high"]) < 1e-9           # at the high
    assert m["vs_52w_low"] > 0                    # well above the low


def test_momentum_short_history_degrades_to_none():
    prices = _series(35, start=10.0, end=11.0)  # ~1 month of data
    m = compute_momentum_inputs(prices, today=TODAY)
    assert m["price_change_1m"] is not None
    assert m["price_change_3m"] is None
    assert m["price_change_1y"] is None
    assert m["vs_52w_high"] is None and m["vs_52w_low"] is None  # young series can't fake 52w
    assert m["rsi"] is not None


def test_momentum_empty_and_priceless():
    assert compute_momentum_inputs([], today=TODAY)["price_change_1m"] is None
    m = compute_momentum_inputs([_P(TODAY, None)], today=TODAY)
    assert all(v is None for v in m.values())


def test_insider_summary_net_direction():
    txns = [_Txn("P", 400_000.0), _Txn("P", 100_000.0), _Txn("S", 50_000.0)]
    s = summarize_insider_activity(txns)
    assert "2 open-market buys ($500,000)" in s and "net buying" in s
    assert summarize_insider_activity([]) == "No insider activity data"
    assert "net selling" in summarize_insider_activity([_Txn("S", 1_000_000.0)])


def test_news_summary_ranks_by_relevance():
    items = [
        _News("boring recap", 0.1),
        _News("CEO resigns amid probe", 0.9),
        _News("earnings beat", 0.5),
    ]
    s = summarize_recent_news(items, max_headlines=2)
    assert s.startswith("3 headlines")
    assert "CEO resigns" in s and "earnings beat" in s and "boring recap" not in s
    assert summarize_recent_news([]) == "No recent news available"
