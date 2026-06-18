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
from investment_monitor.config import Settings, get_settings
from investment_monitor.storage import get_session, init_db


async def run_broad_collection(
    settings: Settings | None = None, *, days_back: int = 1, limit: int | None = None
) -> list[CollectorResult]:
    """Run every broad (universe-independent) collector, returning their results.

    Args:
        settings: app settings (defaults to ``get_settings()``).
        days_back: recent business days of SEC daily indexes to ingest.
        limit: cap on filings parsed per source this run (safety/testing bound).
    """
    settings = settings or get_settings()
    init_db(settings.db_path)

    results: list[CollectorResult] = []
    with get_session() as session:
        # SEC Form 4 insider transactions, retained market-wide (free, authoritative).
        insider = InsiderCollector(session, settings)
        results.append(await insider.collect_all(days_back=days_back, limit=limit))
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
