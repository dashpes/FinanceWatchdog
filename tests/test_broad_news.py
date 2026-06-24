"""Tests for broad, universe-independent news collection (market-wide)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import func, select

from investment_monitor.collectors.news import NewsCollector
from investment_monitor.config import Settings
from investment_monitor.storage import NewsItem, get_session, init_db

# A broad, market-wide feed naming issuers via cashtags — NONE of these need to
# belong to any configured universe; that is the whole point of "broad".
ENTRIES = [
    {"title": "$NVDA pops on AI demand", "summary": "chips", "link": "http://x/nvda"},
    {"title": "Earnings beat", "summary": "$SMCI guides higher", "link": "http://x/smci"},
    {"title": "$TSLA recalls cars", "summary": "", "link": "http://x/tsla"},
    # No resolvable ticker -> must be skipped (broad keeps items WITH a ticker).
    {"title": "Market wrap: stocks mixed", "summary": "no symbols here", "link": "http://x/wrap"},
]


def _settings(tmp_path, db):
    return Settings(public_api_token="t", config_dir=tmp_path, data_dir=tmp_path,
                    log_dir=tmp_path, db_path=db)


def _collector(tmp_path, db, *, entries=ENTRIES):
    """A news collector on a fresh session with the feed fetch mocked (no network)."""
    init_db(db)
    session = get_session().__enter__()  # plain session; collect_all commits itself
    c = NewsCollector(session, _settings(tmp_path, db))
    # Reuse the existing RSS parse seam; collect_all only fetches the broad feeds.
    c._parse_feed = MagicMock(return_value=entries)
    return c


def _count(db):
    init_db(db)
    with get_session() as s:
        return s.scalar(select(func.count()).select_from(NewsItem))


def _tickers(db):
    init_db(db)
    with get_session() as s:
        return {n.ticker for n in s.scalars(select(NewsItem))}


@pytest.mark.asyncio
async def test_collect_all_retains_market_wide(tmp_path):
    db = tmp_path / "n.db"
    result = await _collector(tmp_path, db).collect_all()
    assert result.success and result.records_collected == 3
    # All three resolvable tickers retained; the no-ticker wrap is dropped.
    assert _tickers(db) == {"NVDA", "SMCI", "TSLA"}


@pytest.mark.asyncio
async def test_collect_all_only_fetches_broad_feeds(tmp_path):
    db = tmp_path / "n.db"
    c = _collector(tmp_path, db)
    await c.collect_all()
    # DEFAULT_FEEDS has one per_ticker feed (Yahoo) + one broad feed (Seeking Alpha).
    # Broad collection must skip per_ticker templates -> exactly one parse call.
    assert c._parse_feed.call_count == 1


@pytest.mark.asyncio
async def test_collect_all_dedups_on_url_within_run(tmp_path):
    db = tmp_path / "n.db"
    dupes = [
        {"title": "$NVDA news", "summary": "", "link": "http://x/dup"},
        {"title": "$NVDA news (mirror)", "summary": "", "link": "http://x/dup"},
    ]
    result = await _collector(tmp_path, db, entries=dupes).collect_all()
    assert result.records_collected == 1  # second item url-deduped within the run
    assert _count(db) == 1


@pytest.mark.asyncio
async def test_collect_all_dedups_across_runs(tmp_path):
    db = tmp_path / "n.db"
    await _collector(tmp_path, db).collect_all()
    result = await _collector(tmp_path, db).collect_all()  # same feed again
    assert result.records_collected == 0
    assert _count(db) == 3


@pytest.mark.asyncio
async def test_collect_all_multi_ticker_item_retained_per_ticker(tmp_path):
    db = tmp_path / "n.db"
    multi = [{"title": "$AAPL vs $MSFT", "summary": "rivalry", "link": "http://x/duo"}]
    result = await _collector(tmp_path, db, entries=multi).collect_all()
    # One article naming two issuers is retained once per distinct ticker.
    assert result.records_collected == 2
    assert _tickers(db) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_collect_all_survives_feed_failure(tmp_path):
    db = tmp_path / "n.db"
    c = _collector(tmp_path, db)
    c._parse_feed = MagicMock(side_effect=RuntimeError("feed down"))
    result = await c.collect_all()
    assert not result.success and result.errors  # failure surfaced
    assert _count(db) == 0  # nothing retained, but no crash (fail-open)


# --------------------------------------------------------------------------- #
# #11 junk cashtags must not become NewsItem.ticker rows
# --------------------------------------------------------------------------- #
def test_extract_tickers_drops_junk_cashtags(tmp_path):
    db = tmp_path / "n.db"
    c = _collector(tmp_path, db)
    # $NONE/$NA/$USD/$CEO/$WSJ are non-issuer noise; only $NVDA is a real symbol.
    text = "$NONE $NA $USD $CEO $WSJ $NVDA up on demand"
    assert c._extract_tickers(text) == ["NVDA"]


@pytest.mark.asyncio
async def test_collect_all_skips_junk_cashtags(tmp_path):
    db = tmp_path / "n.db"
    entries = [
        {"title": "$USD weakens vs euro", "summary": "macro", "link": "http://x/usd"},
        {"title": "$CEO comments on $WSJ piece", "summary": "", "link": "http://x/ceo"},
        {"title": "$NVDA pops", "summary": "$NONE here", "link": "http://x/nvda"},
    ]
    result = await _collector(tmp_path, db, entries=entries).collect_all()
    # Only the genuine NVDA cashtag is retained; all junk cashtags are dropped, and the
    # all-junk items resolve to no ticker so they are skipped entirely.
    assert _tickers(db) == {"NVDA"}
    assert result.records_collected == 1


# --------------------------------------------------------------------------- #
# #8 published_at must be parsed as UTC, not shifted by the host's local offset
# --------------------------------------------------------------------------- #
def test_published_date_parsed_as_utc(tmp_path):
    import time
    from datetime import datetime

    import feedparser

    db = tmp_path / "n.db"
    c = _collector(tmp_path, db)
    # feedparser exposes published_parsed as a UTC struct_time (attribute access).
    # 2026-06-17 12:00:00 UTC.
    entry = feedparser.FeedParserDict()
    entry["published_parsed"] = time.struct_time((2026, 6, 17, 12, 0, 0, 2, 168, 0))
    parsed = c._parse_published_date(entry)
    # Must equal the UTC wall-clock regardless of the host's timezone (no offset shift).
    assert parsed == datetime(2026, 6, 17, 12, 0, 0)
