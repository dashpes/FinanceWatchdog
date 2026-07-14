"""Tests for the walk-forward confluence backtest (no look-ahead, real scoring)."""

from __future__ import annotations

from datetime import date, timedelta

from investment_monitor.simulation.backtest import run_confluence_backtest
from investment_monitor.storage import InsiderTransaction, Price, get_session, init_db

# Fixed, deterministic window (backtest queries are all as-of bounded, so the
# real clock never matters).
START = date(2026, 1, 5)   # a Monday
END = date(2026, 5, 29)


def _seed_insiders(s, ticker, on: date, *, n=4, value=60_000):
    # Spread filings over a few days: one-day mass events are deliberately
    # score-penalized by the dispersion factor.
    for i in range(n):
        d = on + timedelta(days=i % 3)
        s.add(InsiderTransaction(
            ticker=ticker, filing_date=d, trade_date=d,
            owner_name=f"{ticker} Insider {i}", owner_title="Director",
            transaction_type="P", raw_code="P", shares=1000,
            price_per_share=value / 1000, total_value=value,
            sec_url=f"u/{ticker}/{d}/{i}",
        ))


def _seed_prices(s, ticker, *, start, days, price_fn, volume=200_000):
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        px = price_fn(i)
        s.add(Price(ticker=ticker, date=d, open=px, high=px, low=px,
                    close=px, volume=volume))


def test_winner_closes_at_horizon_with_gain(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)
    with get_session() as s:
        _seed_prices(s, "WIN", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 10.0 + i * 0.05)  # steady riser
        _seed_insiders(s, "WIN", cluster_day)
    with get_session() as s:
        result = run_confluence_backtest(
            s, start=START, end=END, step_days=5, horizon_days=60,
        )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.ticker == "WIN" and t.entry_date >= cluster_day
    assert t.exit_reason == "horizon" and t.ret > 0


def test_loser_exits_on_drawdown(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)
    with get_session() as s:
        # Flat until the cluster, then a steady slide well past -25%.
        _seed_prices(s, "LOSE", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 20.0 if i < 45 else max(2.0, 20.0 - (i - 45) * 0.4))
        _seed_insiders(s, "LOSE", cluster_day)
    with get_session() as s:
        result = run_confluence_backtest(
            s, start=START, end=END, step_days=5, horizon_days=365,
        )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "drawdown" and t.ret < -0.2


def test_no_lookahead_entry_before_cluster(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = END - timedelta(days=10)  # cluster near the very end
    with get_session() as s:
        _seed_prices(s, "LATE", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 15.0)
        _seed_insiders(s, "LATE", cluster_day)
    with get_session() as s:
        result = run_confluence_backtest(s, start=START, end=END, step_days=5)
    assert len(result.trades) == 1
    # The entry can only exist ON/AFTER the cluster filed — never before.
    assert result.trades[0].entry_date >= cluster_day
    assert result.trades[0].exit_reason == "end_of_data"


def test_illiquid_and_small_clusters_skipped(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)
    with get_session() as s:
        # Penny stock: fails the min_price floor.
        _seed_prices(s, "PENNY", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 1.5)
        _seed_insiders(s, "PENNY", cluster_day)
        # Two-insider cluster with no volume spike: below min_actors.
        _seed_prices(s, "SMALL", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 15.0)
        _seed_insiders(s, "SMALL", cluster_day, n=2)
    with get_session() as s:
        result = run_confluence_backtest(s, start=START, end=END, step_days=5)
    assert result.trades == []


def test_profit_target_exit_captures_gain(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)
    with get_session() as s:
        # Flat through entry, then a steady climb well past +25%.
        _seed_prices(s, "WIN", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 10.0 if i < 50 else 10.0 + (i - 50) * 0.15)
        _seed_insiders(s, "WIN", cluster_day)
    with get_session() as s:
        result = run_confluence_backtest(
            s, start=START, end=END, step_days=5, horizon_days=365,
            profit_target_pct=25.0,
        )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "profit_target" and t.ret >= 0.25


def test_trailing_stop_protects_gain(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)

    def px(i: int) -> float:
        if i < 50:
            return 10.0                              # flat through entry
        if i < 70:
            return 10.0 + (i - 50) * 0.15            # ride to 13.0 (+30%, arms the stop)
        return max(8.0, 13.0 - (i - 70) * 0.1)       # then give it back

    with get_session() as s:
        _seed_prices(s, "RIDE", start=START - timedelta(days=30), days=200, price_fn=px)
        _seed_insiders(s, "RIDE", cluster_day)
    with get_session() as s:
        result = run_confluence_backtest(
            s, start=START, end=END, step_days=5, horizon_days=365,
            trailing_stop_pct=15.0, trailing_arm_pct=10.0,
        )
    assert len(result.trades) == 1
    t = result.trades[0]
    # Exits ~15% below the 13.0 peak — still a protected GAIN, long before the
    # entry-based -25% drawdown would ever have fired.
    assert t.exit_reason == "trailing_stop" and t.ret > 0


def test_summary_bands(tmp_path):
    init_db(tmp_path / "t.db")
    cluster_day = START + timedelta(days=14)
    with get_session() as s:
        _seed_prices(s, "WIN", start=START - timedelta(days=30), days=200,
                     price_fn=lambda i: 10.0 + i * 0.05)
        _seed_insiders(s, "WIN", cluster_day)
    with get_session() as s:
        summary = run_confluence_backtest(
            s, start=START, end=END, step_days=5, horizon_days=60
        ).summary()
    assert summary["n_trades"] == 1 and summary["n_closed"] == 1
    assert summary["overall"]["hit_rate"] == 1.0
    assert sum(b["n"] for b in summary["by_score_band"].values()) == 1
