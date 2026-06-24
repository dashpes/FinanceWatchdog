"""Tests for broad, confluence-relevant price/volume collection.

PriceCollector.collect_all() prices ONLY the names insiders recently traded
(the DISTINCT tickers in insider_transactions within window_days) — not the
whole ~8000-ticker market. It reuses the per-ticker fetch, dedups by Price's
(ticker, date) unique constraint, commits once, and fails open per ticker.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import func, select

from investment_monitor.collectors.prices import PriceCollector
from investment_monitor.config import Settings
from investment_monitor.storage import (
    InsiderTransaction,
    Price,
    get_session,
    init_db,
)


def _settings(tmp_path, db):
    return Settings(public_api_token="t", config_dir=tmp_path, data_dir=tmp_path,
                    log_dir=tmp_path, db_path=db)


def _seed_insider(session, ticker: str, trade_date: date) -> None:
    """One minimal insider row — collect_all derives its universe from these."""
    session.add(InsiderTransaction(
        ticker=ticker,
        filing_date=trade_date,
        trade_date=trade_date,
        owner_name="Insider",
        owner_title="CEO",
        transaction_type="P",
        raw_code="P",
        shares=1000,
        price_per_share=10.0,
        total_value=10_000.0,
        sec_url=f"http://sec/{ticker}/{trade_date}",
    ))


def _ohlcv_frame(tickers: list[str], periods: int = 3) -> pd.DataFrame:
    """A yf.download() style frame mirroring the collector's two code paths:

    - one ticker  -> flat OHLCV columns (single-ticker branch in _batch_fetch)
    - many tickers -> (ticker, field) MultiIndex columns so data[ticker] is a sub-frame
    """
    dates = pd.date_range(end=date.today(), periods=periods, freq="D")
    if len(tickers) == 1:
        return pd.DataFrame({
            "Open": [100.0 + n for n in range(periods)],
            "High": [105.0 + n for n in range(periods)],
            "Low": [99.0 + n for n in range(periods)],
            "Close": [104.0 + n for n in range(periods)],
            "Volume": [1_000_000 + n for n in range(periods)],
        }, index=dates)

    cols = {}
    for i, t in enumerate(tickers):
        base = 100.0 + i
        cols[(t, "Open")] = [base + n for n in range(periods)]
        cols[(t, "High")] = [base + 5 + n for n in range(periods)]
        cols[(t, "Low")] = [base - 1 + n for n in range(periods)]
        cols[(t, "Close")] = [base + 4 + n for n in range(periods)]
        cols[(t, "Volume")] = [1_000_000 + n for n in range(periods)]
    frame = pd.DataFrame(cols, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _collector(tmp_path, db, *, days_to_fetch: int = 30) -> PriceCollector:
    """A price collector wired to a fresh session (collect_all commits itself)."""
    init_db(db)
    session = get_session().__enter__()
    return PriceCollector(session, _settings(tmp_path, db), days_to_fetch=days_to_fetch)


def _price_tickers(db):
    """DISTINCT tickers actually persisted to the prices table (read in-session
    so ORM instances aren't accessed after the session closes)."""
    init_db(db)
    with get_session() as s:
        return {p.ticker for p in s.scalars(select(Price))}


def _price_count(db):
    init_db(db)
    with get_session() as s:
        return s.scalar(select(func.count()).select_from(Price))


@pytest.mark.asyncio
async def test_collect_all_prices_insider_tickers(tmp_path):
    """It fetches prices for exactly the tickers insiders recently traded."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)
    _seed_insider(c.session, "NVDA", date.today() - timedelta(days=2))
    _seed_insider(c.session, "SMCI", date.today() - timedelta(days=5))
    c.session.commit()

    with patch("investment_monitor.collectors.prices.yf.download",
               return_value=_ohlcv_frame(["NVDA", "SMCI"])) as mock_dl:
        result = await c.collect_all()

    assert result.success
    # 2 tickers x 3 days = 6 OHLCV rows, all for the insider names.
    assert result.records_collected == 6
    assert _price_tickers(db) == {"NVDA", "SMCI"}
    # Passed the relevant universe to yfinance — not the whole market.
    called_tickers = set(mock_dl.call_args.kwargs["tickers"])
    assert called_tickers == {"NVDA", "SMCI"}


@pytest.mark.asyncio
async def test_collect_all_dedups_on_second_run(tmp_path):
    """Re-running with the same data adds nothing (Price (ticker,date) dedup)."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)
    _seed_insider(c.session, "NVDA", date.today() - timedelta(days=1))
    c.session.commit()

    frame = _ohlcv_frame(["NVDA"])
    with patch("investment_monitor.collectors.prices.yf.download", return_value=frame):
        first = await c.collect_all()
    assert first.records_collected == 3

    c2 = _collector(tmp_path, db)
    with patch("investment_monitor.collectors.prices.yf.download", return_value=frame):
        second = await c2.collect_all()  # same prices again
    assert second.records_collected == 0
    assert _price_count(db) == 3  # nothing duplicated


@pytest.mark.asyncio
async def test_collect_all_window_excludes_stale_insiders(tmp_path):
    """Tickers whose only insider trade predates window_days are not priced."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)
    _seed_insider(c.session, "NVDA", date.today() - timedelta(days=2))    # fresh
    _seed_insider(c.session, "OLDX", date.today() - timedelta(days=120))  # stale
    c.session.commit()

    with patch("investment_monitor.collectors.prices.yf.download",
               return_value=_ohlcv_frame(["NVDA"])) as mock_dl:
        result = await c.collect_all(window_days=30)

    assert result.success
    assert set(mock_dl.call_args.kwargs["tickers"]) == {"NVDA"}
    assert _price_tickers(db) == {"NVDA"}


@pytest.mark.asyncio
async def test_collect_all_no_insiders_is_noop(tmp_path):
    """No recent insider activity -> empty universe -> no network call, success."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)

    with patch("investment_monitor.collectors.prices.yf.download") as mock_dl:
        result = await c.collect_all()

    assert result.success and result.records_collected == 0
    assert not mock_dl.called  # never reached out to yfinance


@pytest.mark.asyncio
async def test_collect_all_max_tickers_caps_universe(tmp_path):
    """max_tickers bounds the run, keeping the most-recently-active names."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)
    _seed_insider(c.session, "NEWR", date.today() - timedelta(days=1))   # newest
    _seed_insider(c.session, "MIDDL", date.today() - timedelta(days=10))
    _seed_insider(c.session, "OLDER", date.today() - timedelta(days=20))
    c.session.commit()

    with patch("investment_monitor.collectors.prices.yf.download",
               return_value=_ohlcv_frame(["NEWR"])) as mock_dl:
        result = await c.collect_all(max_tickers=1)

    assert result.success
    # Only the single most-recent insider ticker was fetched.
    assert set(mock_dl.call_args.kwargs["tickers"]) == {"NEWR"}


@pytest.mark.asyncio
async def test_relevant_tickers_helper_returns_distinct_recent(tmp_path):
    """The helper returns DISTINCT in-window tickers and excludes stale ones."""
    db = tmp_path / "p.db"
    c = _collector(tmp_path, db)
    _seed_insider(c.session, "AAA", date.today() - timedelta(days=1))
    _seed_insider(c.session, "AAA", date.today() - timedelta(days=3))  # dup ticker
    _seed_insider(c.session, "BBB", date.today() - timedelta(days=4))
    _seed_insider(c.session, "ZZZ", date.today() - timedelta(days=99))  # stale
    c.session.commit()

    tickers = c._relevant_tickers(window_days=30)
    assert set(tickers) == {"AAA", "BBB"}  # distinct, stale ZZZ excluded
    assert len(tickers) == 2  # AAA appears once despite two rows
