"""Broad, universe-independent multi-source collection.

The portfolio collectors (``main.py``) are scoped to holdings + watchlist tickers, so
the market-wide event stream never lands in the DB. The insight engine needs the
opposite: retain what is happening across the WHOLE market — e.g. every congressional
trade, not just trades in names we already hold — so cross-source confluence can later
be mined from it.

This runner invokes collectors in their broad ``collect_all`` mode, decoupled from the
configured universe. It is the foundation the original "aggregate → unique insight"
vision was missing (see the insight-engine-gap analysis).
"""

from __future__ import annotations

import asyncio
from datetime import date

from loguru import logger

from investment_monitor.collectors.base import CollectorResult
from investment_monitor.collectors.insider import InsiderCollector
from investment_monitor.collectors.material_events import MaterialEventsCollector
from investment_monitor.collectors.news import NewsCollector
from investment_monitor.collectors.prices import PriceCollector
from investment_monitor.config import Settings, get_settings
from investment_monitor.storage import get_session, init_db

# Price collection is bounded to the insider-active universe (the names confluence
# cares about) and capped so the daily run stays fast.
_PRICE_WINDOW_DAYS = 30
_PRICE_MAX_TICKERS = 500


async def run_broad_collection(
    settings: Settings | None = None, *, days_back: int = 1, limit: int | None = None
) -> list[CollectorResult]:
    """Run every broad (universe-independent) collector, returning their results.

    Order matters: insider first (it defines the relevant universe), then prices for
    that universe (volume-spike confluence + price context), then market-wide news.

    Args:
        settings: app settings (defaults to ``get_settings()``).
        days_back: recent business days of SEC daily indexes to ingest.
        limit: cap on filings parsed per source this run (safety/testing bound).
    """
    settings = settings or get_settings()
    init_db(settings.db_path)

    results: list[CollectorResult] = []
    with get_session() as session:
        # 1. SEC Form 4 insider transactions, retained market-wide (free, authoritative).
        insider = InsiderCollector(session, settings)
        results.append(await insider.collect_all(days_back=days_back, limit=limit))

        # 1.5. SEC 8-K material corporate events, market-wide (same daily indexes,
        #      header-only fetches). The raw event stream for event-driven theses.
        events = MaterialEventsCollector(session, settings)
        results.append(await events.collect_all(days_back=days_back, limit=limit))

        # 2. Daily OHLCV for the insider-active universe (foundation for volume-spike
        #    confluence + price context). No-op on a fresh DB with no insider rows.
        prices = PriceCollector(session, settings)
        results.append(await prices.collect_all(
            window_days=_PRICE_WINDOW_DAYS, max_tickers=_PRICE_MAX_TICKERS,
        ))

        # 3. Market-wide news (non-directional context).
        news = NewsCollector(session, settings)
        results.append(await news.collect_all())

        # Congress is DEFERRED: the free House/Senate Stock Watcher feed is dead. The
        # broad collect_all() + tests exist in CongressTradesCollector, ready to repoint
        # at a live source (House Clerk PTR PDFs or a paid API).

    for r in results:
        logger.info(f"broad-collect: {r}")
    return results


def run_broad_collection_sync(
    settings: Settings | None = None, *, days_back: int = 1, limit: int | None = None
) -> list[CollectorResult]:
    """Synchronous wrapper around :func:`run_broad_collection` for CLI/cron use."""
    return asyncio.run(run_broad_collection(settings, days_back=days_back, limit=limit))
