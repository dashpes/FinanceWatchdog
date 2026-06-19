"""Tests for the confluence / insight engine."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import func, select

from investment_monitor.analysis.confluence import (
    ConfluenceConfig,
    Evidence,
    detect_confluence,
    gather_insider_evidence,
    gather_volume_evidence,
    score_confluence,
)
from investment_monitor.storage import (
    ConfluenceFinding,
    InsiderTransaction,
    Price,
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


# --------------------------------------------------------------------------- #
# v2: data-quality + concentration-aware scoring
# --------------------------------------------------------------------------- #
def test_entity_owners_excluded(tmp_path):
    # Fund/entity 10%-holders are not individual-insider conviction — exclude them.
    db = tmp_path / "i.db"
    _seed(db, [
        ("BIO", "Alice Smith", "P", 2, 60000),
        ("BIO", "Bob Jones", "P", 2, 60000),
        ("BIO", "RA Capital Management, L.P.", "P", 2, 5_000_000),
    ])
    with get_session() as s:
        actors = {e.actor for e in gather_insider_evidence(s, 30, TODAY)}
    assert "Alice Smith" in actors and "Bob Jones" in actors
    assert not any("Capital" in a for a in actors)


def test_duplicate_physical_transaction_deduped(tmp_path):
    # Same Form 4 indexed under issuer + owner CIK lands as two rows; the engine must
    # collapse it so dollars aren't double-counted.
    db = tmp_path / "i.db"
    _seed(db, [
        ("DUP", "Alice", "P", 2, 100000),
        ("DUP", "Alice", "P", 2, 100000),   # identical physical txn, second copy
        ("DUP", "Bob", "P", 2, 100000),
        ("DUP", "Cy", "P", 2, 100000),
    ])
    with get_session() as s:
        ev = gather_insider_evidence(s, 30, TODAY)
    assert len(ev) == 3                                  # Alice counted once
    assert sum(e.value for e in ev) == 300000            # not 400000


def test_dollar_floor_drops_trivial_clusters(tmp_path):
    db = tmp_path / "i.db"
    _seed(db, [("TINY", n, "P", 2, 2000) for n in ("A", "B", "C", "D")])  # $8k < floor
    with get_session() as s:
        assert detect_confluence(s, ConfluenceConfig(), today=TODAY) == []


def test_concentration_beats_routine_breadth(tmp_path):
    # MEGA: 10 insiders, one day, token buys (routine board-wide event).
    # CONV:  5 insiders, spread over days, real money (genuine conviction cluster).
    db = tmp_path / "i.db"
    rows = [("MEGA", f"Filer{i}", "P", 5, 3000) for i in range(10)]
    rows += [("CONV", f"Exec{i}", "P", 1 + i, 80000) for i in range(5)]
    _seed(db, rows)
    with get_session() as s:
        scores = {f.ticker: f.score for f in detect_confluence(s, ConfluenceConfig(), today=TODAY)}
    assert "CONV" in scores and "MEGA" in scores
    assert scores["CONV"] > scores["MEGA"]   # conviction ranks above raw headcount


# --------------------------------------------------------------------------- #
# v3: volume-spike second source + price context
# --------------------------------------------------------------------------- #
def _seed_prices(db, ticker, *, days=22, base_vol=100000, spike_vol=None,
                 base_close=10.0, latest_close=None):
    init_db(db)
    with get_session() as s:
        for i in range(days):
            d = TODAY - timedelta(days=i)
            vol = spike_vol if (i == 0 and spike_vol) else base_vol
            close = latest_close if (i == 0 and latest_close) else base_close
            s.add(Price(ticker=ticker, date=d, open=close, high=close, low=close,
                        close=close, volume=vol))


def test_volume_evidence_detects_spike(tmp_path):
    db = tmp_path / "v.db"
    _seed_prices(db, "SPK", spike_vol=300000)            # 3x the 100k baseline
    with get_session() as s:
        ev = gather_volume_evidence(s, {"SPK"}, TODAY)
    assert len(ev) == 1 and ev[0].source == "volume" and ev[0].ticker == "SPK"


def test_no_volume_evidence_without_spike(tmp_path):
    db = tmp_path / "v.db"
    _seed_prices(db, "FLAT")                              # flat volume, no spike
    with get_session() as s:
        assert gather_volume_evidence(s, {"FLAT"}, TODAY) == []


def test_cross_source_insider_plus_volume(tmp_path):
    # Only 2 insiders (below the 3-cluster floor) — but a corroborating volume spike
    # makes it a cross-SOURCE finding.
    db = tmp_path / "x.db"
    _seed(db, [("XS", "Alice", "P", 2, 60000), ("XS", "Bob", "P", 3, 60000)])
    _seed_prices(db, "XS", spike_vol=300000)
    with get_session() as s:
        findings = detect_confluence(s, ConfluenceConfig(), today=TODAY)
        xs = [f for f in findings if f.ticker == "XS"]
        assert xs, "expected a cross-source finding"
        assert xs[0].n_sources == 2 and xs[0].kind == "multi_source"
        assert "unusual volume" in xs[0].narrative


def test_price_context_set_on_finding(tmp_path):
    db = tmp_path / "p.db"
    _seed(db, [("RISE", n, "P", 5, 60000) for n in ("A", "B", "C")])
    _seed_prices(db, "RISE", base_close=10.0, latest_close=12.0)   # +20% since the buys
    with get_session() as s:
        findings = detect_confluence(s, ConfluenceConfig(), today=TODAY)
        r = [f for f in findings if f.ticker == "RISE"][0]
        assert r.price_change_pct is not None and r.price_change_pct > 15
        assert "% since buys" in r.narrative


def test_news_evidence_requires_min_items(tmp_path):
    from datetime import datetime

    from investment_monitor.analysis.confluence import gather_news_evidence
    from investment_monitor.storage import NewsItem

    init_db(tmp_path / "n.db")
    with get_session() as s:
        for i in range(3):
            s.add(NewsItem(ticker="NWS", headline=f"h{i}", source="x",
                           url=f"http://n/{i}", published_at=datetime(2026, 6, 17, 12, 0)))
        s.add(NewsItem(ticker="ONE", headline="h", source="x", url="http://one",
                       published_at=datetime(2026, 6, 17, 12, 0)))
    with get_session() as s:
        ev = gather_news_evidence(s, {"NWS", "ONE"}, TODAY, window_days=30, min_items=2)
    tickers = {e.ticker for e in ev}
    assert "NWS" in tickers       # 3 headlines >= min_items
    assert "ONE" not in tickers   # 1 headline < min_items
    assert all(e.source == "news" for e in ev)
