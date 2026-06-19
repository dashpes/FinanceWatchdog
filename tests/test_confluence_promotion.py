"""Tests for promoting confluence findings into tradeable theses (the loop closure)."""

from __future__ import annotations

from datetime import date

from investment_monitor.robo.confluence_promotion import (
    _conviction_from_score,
    promote_confluence_findings,
)
from investment_monitor.storage import (
    ConfluenceFinding,
    Price,
    get_active_theses,
    get_session,
    init_db,
)

TODAY = date(2026, 6, 18)


def _seed_finding(s, ticker, score, kind="insider_cluster"):
    s.add(ConfluenceFinding(
        ticker=ticker, kind=kind, score=score, window_days=30, n_sources=1,
        n_actors=4, total_value=100000.0, evidence=[],
        narrative=f"{ticker}: 4 insiders bought.", as_of_date=TODAY,
    ))


def _seed_price(s, ticker, close=10.0):
    s.add(Price(ticker=ticker, date=TODAY, open=close, high=close, low=close,
                close=close, volume=100000))


def test_conviction_band():
    assert _conviction_from_score(8.0) == 0.8
    assert _conviction_from_score(100) == 0.85   # capped
    assert _conviction_from_score(0) == 0.4      # 0.4 floor region


def test_promote_creates_active_theses_above_floor(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        for t, sc in [("AAA", 8.0), ("BBB", 6.0), ("CCC", 3.0)]:  # CCC below min_score
            _seed_finding(s, t, sc)
            _seed_price(s, t)
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0, max_promotions=5)
    assert set(promoted) == {"AAA", "BBB"}
    with get_session() as s:
        assert {t.symbol for t in get_active_theses(s)} == {"AAA", "BBB"}


def test_promote_skips_untradeable(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "NOPX", 8.0)              # no price -> untradeable
        _seed_finding(s, "OK", 8.0); _seed_price(s, "OK")
    with get_session() as s:
        assert promote_confluence_findings(s, min_score=4.0) == ["OK"]


def test_promote_caps_and_does_not_duplicate(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        for i in range(6):
            _seed_finding(s, f"T{i}", 8.0 - i * 0.1)
            _seed_price(s, f"T{i}")
    with get_session() as s:
        first = promote_confluence_findings(s, min_score=4.0, max_promotions=3)
    assert len(first) == 3
    with get_session() as s:
        second = promote_confluence_findings(s, min_score=4.0, max_promotions=3)
    # Already-promoted names are skipped; the next batch gets promoted instead.
    assert len(second) == 3 and set(second).isdisjoint(set(first))
