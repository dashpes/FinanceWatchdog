"""Tests for batch sentiment labeling and its confluence integration."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from investment_monitor.analysis.confluence import gather_news_evidence
from investment_monitor.analysis.news_processor import classify_unscored_sentiment
from investment_monitor.storage import (
    NewsItem,
    get_recent_news,
    get_session,
    get_unclassified_news,
    init_db,
)

TODAY = date.today()


def _news(ticker, headline, *, sentiment=None, days_ago=1, url_suffix=""):
    published = datetime.now() - timedelta(days=days_ago)
    return NewsItem(
        ticker=ticker, headline=headline, source="test",
        url=f"https://example.com/{ticker}/{headline[:12]}{url_suffix}",
        published_at=published, sentiment=sentiment,
    )


def test_unclassified_query_skips_tickerless(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        s.add(_news("AAA", "up big"))
        s.add(_news(None, "market wrap"))
        s.add(_news("BBB", "already done", sentiment="bullish"))
    with get_session() as s:
        assert [i.ticker for i in get_unclassified_news(s)] == ["AAA"]


def test_classify_batch_persists_and_retries_unknown(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        s.add(_news("AAA", "record earnings"))
        s.add(_news("BBB", "mystery item"))
    llm = MagicMock()
    llm.is_available.return_value = True
    llm.classify_sentiment = AsyncMock(side_effect=["bullish", "unknown"])
    with get_session() as s:
        assert asyncio.run(classify_unscored_sentiment(s, llm)) == 1
    with get_session() as s:
        by_headline = {i.headline: i.sentiment for i in get_recent_news(s, hours=48)}
        assert by_headline["record earnings"] == "bullish"
        assert by_headline["mystery item"] is None      # left NULL -> retried next pass
        # And it IS retried: still in the unclassified queue.
        assert [i.headline for i in get_unclassified_news(s)] == ["mystery item"]


def test_classify_noop_without_llm(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        s.add(_news("AAA", "something"))
    llm = MagicMock()
    llm.is_available.return_value = False
    with get_session() as s:
        assert asyncio.run(classify_unscored_sentiment(s, llm)) == 0


def test_confluence_news_excludes_bearish(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        # BAD: 3 headlines but 2 bearish -> only 1 kept -> below min_items(2).
        s.add(_news("BAD", "probe widens", sentiment="bearish", url_suffix="1"))
        s.add(_news("BAD", "lawsuit filed", sentiment="bearish", url_suffix="2"))
        s.add(_news("BAD", "neutral note", url_suffix="3"))
        # GOOD: bullish + unlabeled both count.
        s.add(_news("GOOD", "contract win", sentiment="bullish", url_suffix="4"))
        s.add(_news("GOOD", "analyst day", url_suffix="5"))
    with get_session() as s:
        ev = gather_news_evidence(s, {"BAD", "GOOD"}, TODAY, min_items=2)
    assert [e.ticker for e in ev] == ["GOOD"]
    assert "1 bullish" in ev[0].detail
