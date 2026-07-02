"""Tests for the shadow ledger — tracking the theses the robo did NOT trade."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select

from investment_monitor.robo.confluence_promotion import promote_confluence_findings
from investment_monitor.robo.shadow import (
    evaluate_shadow_entries,
    maintain_shadow_ledger,
    record_discovery_shadows,
    shadow_report,
    sync_gate_reject_shadows,
)
from investment_monitor.storage import (
    ConfluenceFinding,
    LEARNING_KIND_SHADOW_OUTCOME,
    LearningEvent,
    Price,
    RoboOrder,
    SHADOW_SOURCE_CONFLUENCE,
    SHADOW_SOURCE_DISCOVERY,
    SHADOW_SOURCE_GATE,
    SHADOW_STATUS_CLOSED,
    SHADOW_STATUS_OPEN,
    StockCandidate,
    accuracy_stats_for_symbol,
    get_session,
    get_shadow_entries,
    init_db,
    record_shadow_entry,
)

TODAY = date.today()


def _seed_finding(s, ticker, score, price_change_pct=None, as_of=TODAY):
    s.add(ConfluenceFinding(
        ticker=ticker, kind="insider_cluster", score=score, window_days=30, n_sources=1,
        n_actors=4, total_value=100000.0, evidence=[], price_change_pct=price_change_pct,
        narrative=f"{ticker}: 4 insiders bought.", as_of_date=as_of,
    ))


def _seed_price(s, ticker, close=10.0, volume=100000, days=20):
    for i in range(days):
        s.add(Price(ticker=ticker, date=TODAY - timedelta(days=i), open=close, high=close,
                    low=close, close=close, volume=volume))


def _entries(s, source=None):
    return get_shadow_entries(s, source=source, limit=100)


def test_promotion_records_skip_reasons(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "LOWS", 3.0)                          # below promotion floor (4.0)
        _seed_price(s, "LOWS")
        _seed_finding(s, "RUNUP", 8.0, price_change_pct=55.0)  # already ran
        _seed_price(s, "RUNUP")
        _seed_finding(s, "THIN", 8.0)                          # no price -> illiquid
        _seed_finding(s, "OK", 9.0)
        _seed_price(s, "OK")
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0, max_promotions=5)
    assert promoted == ["OK"]
    with get_session() as s:
        by_symbol = {
            e.symbol: (e.skip_reason, e.entry_price)
            for e in _entries(s, SHADOW_SOURCE_CONFLUENCE)
        }
    assert by_symbol["LOWS"] == ("below_score_floor", 10.0)  # price snapshot for the counterfactual
    assert by_symbol["RUNUP"][0] == "run_up"
    assert by_symbol["THIN"] == ("illiquid", None)           # unpriceable, kept for the record
    assert "OK" not in by_symbol                             # promoted names are never shadowed


def test_promotion_cap_overflow_is_shadowed(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        for i in range(4):
            _seed_finding(s, f"T{i}", 9.0 - i * 0.1)
            _seed_price(s, f"T{i}")
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0, max_promotions=2)
    assert len(promoted) == 2
    with get_session() as s:
        reasons = {e.symbol: e.skip_reason for e in _entries(s, SHADOW_SOURCE_CONFLUENCE)}
    assert set(reasons.values()) == {"cap_overflow"} and len(reasons) == 2


def test_open_entry_not_duplicated_across_runs(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "LOWS", 3.0)
        _seed_price(s, "LOWS")
    for _ in range(3):  # the research loop re-runs promotion many times a day
        with get_session() as s:
            promote_confluence_findings(s, min_score=4.0)
    with get_session() as s:
        assert len(_entries(s, SHADOW_SOURCE_CONFLUENCE)) == 1


def test_evaluate_marks_open_and_closes_at_horizon(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_price(s, "YNG", close=12.0)
        _seed_price(s, "OLD", close=15.0)
        record_shadow_entry(
            s, symbol="YNG", source=SHADOW_SOURCE_CONFLUENCE, skip_reason="run_up",
            entry_date=TODAY - timedelta(days=10), entry_price=10.0, conviction=0.6,
        )
        record_shadow_entry(
            s, symbol="OLD", source=SHADOW_SOURCE_CONFLUENCE, skip_reason="illiquid",
            entry_date=TODAY - timedelta(days=95), entry_price=10.0, conviction=0.6,
        )
    with get_session() as s:
        out = evaluate_shadow_entries(s, horizon_days=90)
    assert out == {"marked": 1, "closed": 1}
    with get_session() as s:
        by_symbol = {e.symbol: e for e in _entries(s)}
        assert by_symbol["YNG"].status == SHADOW_STATUS_OPEN
        assert abs(by_symbol["YNG"].realized_return - 0.2) < 1e-9
        assert by_symbol["OLD"].status == SHADOW_STATUS_CLOSED
        assert abs(by_symbol["OLD"].realized_return - 0.5) < 1e-9
        events = list(s.scalars(select(LearningEvent).where(
            LearningEvent.kind == LEARNING_KIND_SHADOW_OUTCOME)))
        assert len(events) == 1 and events[0].symbol == "OLD"
        assert events[0].direction_correct == 1
        # Hypothetical outcomes must NOT leak into real-money accuracy stats.
        assert accuracy_stats_for_symbol(s, "OLD")["n"] == 0


def test_gate_rejects_sync_once(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_price(s, "REJ", close=20.0)
        s.add(RoboOrder(run_id="r1", symbol="REJ", side="buy", order_type="market",
                        notional=100.0, gate_accepted=False, gate_code="max_positions"))
        s.add(RoboOrder(run_id="r1", symbol="SLD", side="sell", order_type="market",
                        notional=100.0, gate_accepted=False, gate_code="blocklist"))
    with get_session() as s:
        assert sync_gate_reject_shadows(s) == 1  # buys only; sells aren't counterfactual longs
        assert sync_gate_reject_shadows(s) == 0  # ref_id dedup: never re-shadowed
        (entry,) = _entries(s, SHADOW_SOURCE_GATE)
        assert entry.symbol == "REJ" and entry.skip_reason == "gate:max_positions"
        assert entry.entry_price == 20.0


def test_discovery_near_miss_band(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_price(s, "NEAR", close=30.0)
        _seed_price(s, "FAR", close=30.0)
        s.add(StockCandidate(ticker="NEAR", composite_score=48.0))  # floor 55, band 15
        s.add(StockCandidate(ticker="FAR", composite_score=20.0))   # far below: noise
        s.add(StockCandidate(ticker="ABOVE", composite_score=60.0))
    with get_session() as s:
        assert record_discovery_shadows(s, score_floor=55.0) == 1
        (entry,) = _entries(s, SHADOW_SOURCE_DISCOVERY)
        assert entry.symbol == "NEAR" and entry.skip_reason == "below_score_floor"
        assert abs(entry.conviction - 0.48) < 1e-9


def test_maintain_and_report_roundtrip(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_price(s, "AAA", close=11.0)
        record_shadow_entry(
            s, symbol="AAA", source=SHADOW_SOURCE_CONFLUENCE, skip_reason="cap_overflow",
            entry_date=TODAY - timedelta(days=100), entry_price=10.0, conviction=0.7,
        )
    with get_session() as s:
        maintain_shadow_ledger(s, score_floor=55.0)
        report = shadow_report(s)
    stats = report["shadow"][SHADOW_SOURCE_CONFLUENCE]
    assert stats["closed"] == 1 and stats["hit_rate"] == 1.0
    assert abs(stats["avg_return"] - 0.1) < 1e-9
    assert report["real"]["n"] == 0
