"""Tests for the confluence / insight engine."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import func, select

from investment_monitor.analysis.confluence import (
    ConfluenceConfig,
    Evidence,
    detect_confluence,
    gather_insider_evidence,
    score_confluence,
)
from investment_monitor.storage import (
    ConfluenceFinding,
    InsiderTransaction,
    get_session,
    init_db,
)

TODAY = date(2026, 6, 18)


def _ev(ticker, source, actor, days_ago=1, value=10000.0):
    return Evidence(ticker=ticker, source=source, actor=actor,
                    date=TODAY - timedelta(days=days_ago), value=value, detail="x")


# --------------------------------------------------------------------------- #
# Pure scorer
# --------------------------------------------------------------------------- #
def test_score_rewards_more_distinct_actors():
    one = score_confluence([_ev("X", "insider", "A")], today=TODAY)
    three = score_confluence(
        [_ev("X", "insider", "A"), _ev("X", "insider", "B"), _ev("X", "insider", "C")],
        today=TODAY,
    )
    assert three["n_actors"] == 3 and three["n_sources"] == 1
    assert three["score"] > one["score"]


def test_score_is_super_additive_across_sources():
    # Same actor count (2) — but TWO distinct sources must score higher than one,
    # the opposite of the old weighted-average that diluted corroboration.
    one_source = score_confluence(
        [_ev("X", "insider", "A"), _ev("X", "insider", "B")], today=TODAY)
    two_source = score_confluence(
        [_ev("X", "insider", "A"), _ev("X", "congress", "Rep1")], today=TODAY)
    assert one_source["n_sources"] == 1 and two_source["n_sources"] == 2
    assert two_source["score"] > one_source["score"]


def test_score_decays_with_staleness():
    fresh = score_confluence([_ev("X", "insider", "A", days_ago=0)], today=TODAY)
    stale = score_confluence([_ev("X", "insider", "A", days_ago=60)], today=TODAY)
    assert fresh["score"] > stale["score"]


def test_score_empty_is_zero():
    assert score_confluence([], today=TODAY)["score"] == 0.0


# --------------------------------------------------------------------------- #
# Insider gatherer + detection (DB)
# --------------------------------------------------------------------------- #
def _seed(db, rows):
    """rows: (ticker, owner, raw_code, days_ago, value)."""
    init_db(db)
    with get_session() as s:
        for i, (ticker, owner, code, days_ago, value) in enumerate(rows):
            ttype = "P" if code in ("P", "A", "M") else "S"
            s.add(InsiderTransaction(
                ticker=ticker, filing_date=TODAY - timedelta(days=days_ago),
                trade_date=TODAY - timedelta(days=days_ago), owner_name=owner,
                owner_title="Director", transaction_type=ttype, raw_code=code,
                shares=100, price_per_share=value / 100, total_value=value,
                sec_url=f"u/{ticker}/{owner}/{code}/{days_ago}/{i}",
            ))


def test_gather_keeps_only_real_buys_and_real_tickers(tmp_path):
    db = tmp_path / "i.db"
    _seed(db, [
        ("AAA", "Alice", "P", 1, 50000),    # genuine open-market buy
        ("AAA", "Bob", "A", 1, 999999),     # grant -> excluded
        ("BBB", "Carol", "S", 1, 50000),    # sale -> excluded
        ("NONE", "Dan", "P", 1, 50000),     # junk ticker -> excluded
    ])
    with get_session() as s:
        ev = gather_insider_evidence(s, 30, TODAY)
    assert {e.ticker for e in ev} == {"AAA"}
    assert all(e.source == "insider" for e in ev)


def test_detect_emits_only_real_clusters(tmp_path):
    db = tmp_path / "i.db"
    rows = [("CLST", n, "P", 2, 40000) for n in ("Ann", "Bob", "Cy", "Dee")]
    rows.append(("SOLO", "Eve", "P", 2, 40000))  # single insider -> not a cluster
    _seed(db, rows)
    with get_session() as s:
        findings = detect_confluence(s, ConfluenceConfig(min_actors=3), today=TODAY)
        assert {f.ticker for f in findings} == {"CLST"}
        f = findings[0]
        assert f.n_actors == 4 and f.n_sources == 1
        assert f.kind == "insider_cluster"
        assert "4 insiders" in f.narrative


def test_detect_dedups_within_a_day(tmp_path):
    db = tmp_path / "i.db"
    _seed(db, [("CLST", n, "P", 2, 40000) for n in ("Ann", "Bob", "Cy", "Dee")])
    with get_session() as s:
        detect_confluence(s, ConfluenceConfig(min_actors=3), today=TODAY)
    with get_session() as s:
        again = detect_confluence(s, ConfluenceConfig(min_actors=3), today=TODAY)
        assert again == []
        assert s.scalar(select(func.count()).select_from(ConfluenceFinding)) == 1
