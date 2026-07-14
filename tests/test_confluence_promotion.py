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

# Seed relative to the real today: the bridge applies recency (max_age_days) and
# liquidity windows against date.today().
TODAY = date.today()


def _seed_finding(s, ticker, score, kind="insider_cluster", price_change_pct=None, as_of=TODAY):
    s.add(ConfluenceFinding(
        ticker=ticker, kind=kind, score=score, window_days=30, n_sources=1,
        n_actors=4, total_value=100000.0, evidence=[], price_change_pct=price_change_pct,
        narrative=f"{ticker}: 4 insiders bought.", as_of_date=as_of,
    ))


def _seed_price(s, ticker, close=10.0, volume=100000):
    # 20 days of liquid history so the dollar-volume floor passes.
    from datetime import timedelta
    for i in range(20):
        s.add(Price(ticker=ticker, date=TODAY - timedelta(days=i), open=close, high=close,
                    low=close, close=close, volume=volume))


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


def test_illiquid_penny_stock_skipped(tmp_path):
    init_path = tmp_path / "t.db"
    init_db(init_path)
    with get_session() as s:
        _seed_finding(s, "PENNY", 8.0)
        _seed_price(s, "PENNY", close=0.50)          # below the $3 floor
        _seed_finding(s, "THIN", 8.0)
        _seed_price(s, "THIN", close=10.0, volume=100)  # ~$1k/day dollar-volume, too thin
        _seed_finding(s, "GOOD", 8.0)
        _seed_price(s, "GOOD", close=10.0, volume=100000)
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0)
    assert promoted == ["GOOD"]   # penny + thin are filtered out


def test_already_run_up_skipped(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "RAN", 8.0, price_change_pct=135.0)   # already +135%
        _seed_price(s, "RAN")
        _seed_finding(s, "FRESH", 8.0, price_change_pct=5.0)
        _seed_price(s, "FRESH")
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0, max_run_pct=40.0)
    assert promoted == ["FRESH"]


def test_stale_finding_not_promoted(tmp_path):
    from datetime import timedelta
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "OLD", 8.0, as_of=TODAY - timedelta(days=10))  # outside 3-day window
        _seed_price(s, "OLD")
    with get_session() as s:
        promoted = promote_confluence_findings(s, min_score=4.0, max_age_days=3)
    assert promoted == []


def test_promoted_thesis_carries_horizon_exit(tmp_path):
    # A confluence bet is time-boxed: the promotion stamps the backtest-validated
    # 90d horizon; profit target / trailing stop come from the config defaults.
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "AAA", 8.0)
        _seed_price(s, "AAA")
    with get_session() as s:
        assert promote_confluence_findings(s, min_score=4.0) == ["AAA"]
        assert get_active_theses(s)[0].exit_conditions == {"max_hold_days": 90}


def test_exited_name_not_repromoted_from_same_finding(tmp_path):
    # After a take-profit/horizon exit, the SAME finding must not churn the sell
    # straight back into a buy; only a genuinely NEW finding may re-enter.
    from datetime import datetime

    from investment_monitor.storage import (
        Thesis, ThesisStatus, get_recent_findings, save_thesis,
    )
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "TOOK", 8.0)
        _seed_price(s, "TOOK")
    with get_session() as s:
        f = get_recent_findings(s, min_score=4.0, max_age_days=3)[0]
        save_thesis(s, Thesis(
            symbol="TOOK", narrative="x", conviction=0.0,
            status=ThesisStatus.EXITED.value,
            evidence_refs={"confluence_finding_id": f.id},
            last_evaluated_at=datetime.combine(TODAY, datetime.min.time()),
        ))
    with get_session() as s:
        assert promote_confluence_findings(s, min_score=4.0) == []


def test_falling_knife_not_repromoted_from_same_finding(tmp_path):
    # A self-invalidated name must NOT be re-bought from the SAME stale finding.
    from datetime import datetime

    from investment_monitor.storage import (
        Thesis, ThesisStatus, get_recent_findings, save_thesis,
    )
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_finding(s, "KNIFE", 8.0)
        _seed_price(s, "KNIFE")
    with get_session() as s:
        f = get_recent_findings(s, min_score=4.0, max_age_days=3)[0]
        save_thesis(s, Thesis(
            symbol="KNIFE", narrative="x", conviction=0.0,
            status=ThesisStatus.INVALIDATED.value,
            evidence_refs={"confluence_finding_id": f.id},
            last_evaluated_at=datetime.combine(TODAY, datetime.min.time()),
        ))
    with get_session() as s:
        assert promote_confluence_findings(s, min_score=4.0) == []


def test_same_day_fresh_finding_repromotes_invalidated_name(tmp_path):
    # A self-invalidated name MUST be re-promoted on a genuinely NEW (different) finding
    # produced on the SAME calendar day a thesis was last evaluated — a same-day fresh,
    # stronger cross-source signal is a real trade, not a falling knife. Only the SAME
    # (or strictly older) finding is the falling knife we must refuse.
    from datetime import datetime

    from investment_monitor.storage import (
        Thesis, ThesisStatus, get_recent_findings, save_thesis,
    )
    init_db(tmp_path / "t.db")
    with get_session() as s:
        # Original (weaker) finding that drove the first buy, then a brand-new stronger
        # finding produced the very same day. Both dated TODAY.
        _seed_finding(s, "REBORN", 5.0)
        _seed_finding(s, "REBORN", 9.0)
        _seed_price(s, "REBORN")
    with get_session() as s:
        # The finding recorded on the invalidated thesis is the ORIGINAL (weaker) one,
        # NOT the new strongest one promotion will now consider.
        findings = get_recent_findings(s, min_score=4.0, max_age_days=3)
        original = min(findings, key=lambda f: f.score)
        save_thesis(s, Thesis(
            symbol="REBORN", narrative="x", conviction=0.0,
            status=ThesisStatus.INVALIDATED.value,
            evidence_refs={"confluence_finding_id": original.id},
            # Last evaluated TODAY (same calendar day as the fresh finding's as_of_date).
            last_evaluated_at=datetime.combine(TODAY, datetime.min.time()),
        ))
    with get_session() as s:
        # The new, stronger, same-day finding re-promotes the name.
        assert promote_confluence_findings(s, min_score=4.0) == ["REBORN"]
        assert {t.symbol for t in get_active_theses(s)} == {"REBORN"}
