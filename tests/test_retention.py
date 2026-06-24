"""Tests for the retention / cleanup policy (storage/retention.py).

Mirrors the broad-collection test style: tmp_path SQLite, no network, seed rows
directly, then assert the prune window deletes the OLD rows and keeps the recent ones,
and that the default (all-zero) config is a strict NO-OP.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select

from investment_monitor.storage import (
    ConfluenceFinding,
    InsiderTransaction,
    NewsItem,
    Price,
    get_session,
    init_db,
)
from investment_monitor.storage.retention import RetentionConfig, prune_old_data

# "old" = ~400 days back (outside any reasonable window); "recent" = today-ish.
OLD_D = date.today() - timedelta(days=400)
RECENT_D = date.today() - timedelta(days=5)
_UTCNOW = datetime.now(timezone.utc).replace(tzinfo=None)
OLD_DT = _UTCNOW - timedelta(days=400)
RECENT_DT = _UTCNOW - timedelta(days=5)


def _seed(db) -> None:
    """One OLD + one RECENT row in each prunable table."""
    init_db(db)
    with get_session() as s:
        # insider_transactions — aged by trade_date
        for d, tag in ((OLD_D, "old"), (RECENT_D, "new")):
            s.add(InsiderTransaction(
                ticker="NVDA", filing_date=d, trade_date=d, owner_name=f"O-{tag}",
                transaction_type="P", shares=1, sec_url=f"http://x/ins/{tag}",
            ))
        # news_items — aged by published_at
        for dt, tag in ((OLD_DT, "old"), (RECENT_DT, "new")):
            s.add(NewsItem(
                headline=f"H-{tag}", source="src", url=f"http://x/news/{tag}",
                published_at=dt,
            ))
        # prices — aged by date
        for d in (OLD_D, RECENT_D):
            s.add(Price(ticker="NVDA", date=d, close=1.0))
        # confluence_findings — aged by as_of_date
        for d in (OLD_D, RECENT_D):
            s.add(ConfluenceFinding(
                ticker="NVDA", kind="insider_cluster", as_of_date=d,
            ))


def _counts(db) -> dict[str, int]:
    init_db(db)
    with get_session() as s:
        return {
            "insider_transactions": s.scalar(select(func.count()).select_from(InsiderTransaction)),
            "news_items": s.scalar(select(func.count()).select_from(NewsItem)),
            "prices": s.scalar(select(func.count()).select_from(Price)),
            "confluence_findings": s.scalar(select(func.count()).select_from(ConfluenceFinding)),
        }


def _remaining_insider_owners(db) -> set[str]:
    init_db(db)
    with get_session() as s:
        return {t.owner_name for t in s.scalars(select(InsiderTransaction))}


def test_default_config_is_strict_noop(tmp_path):
    db = tmp_path / "r.db"
    _seed(db)
    init_db(db)
    with get_session() as s:
        deleted = prune_old_data(s, RetentionConfig())
    assert deleted == {}  # nothing reported deleted
    # Every row survives — keep-everything is the default.
    assert _counts(db) == {
        "insider_transactions": 2, "news_items": 2, "prices": 2, "confluence_findings": 2,
    }


def test_window_deletes_old_keeps_recent_all_sources(tmp_path):
    db = tmp_path / "r.db"
    _seed(db)
    # A 30-day window on every source: OLD (400d) goes, RECENT (5d) stays.
    cfg = RetentionConfig(insider_days=30, news_days=30, price_days=30, findings_days=30)
    init_db(db)
    with get_session() as s:
        deleted = prune_old_data(s, cfg)
    assert deleted == {
        "insider_transactions": 1, "news_items": 1, "prices": 1, "confluence_findings": 1,
    }
    assert _counts(db) == {
        "insider_transactions": 1, "news_items": 1, "prices": 1, "confluence_findings": 1,
    }
    # The surviving insider row is the RECENT one, not the old one.
    assert _remaining_insider_owners(db) == {"O-new"}


def test_disabled_source_is_skipped_when_others_enabled(tmp_path):
    db = tmp_path / "r.db"
    _seed(db)
    # Only news is enabled; insider/price/findings windows stay 0 (untouched).
    cfg = RetentionConfig(news_days=30)
    init_db(db)
    with get_session() as s:
        deleted = prune_old_data(s, cfg)
    assert deleted == {"news_items": 1}  # only the enabled source reports a count
    counts = _counts(db)
    assert counts["news_items"] == 1               # old news pruned
    assert counts["insider_transactions"] == 2     # disabled — both kept
    assert counts["prices"] == 2                    # disabled — both kept
    assert counts["confluence_findings"] == 2       # disabled — both kept


def test_window_wide_enough_keeps_everything(tmp_path):
    db = tmp_path / "r.db"
    _seed(db)
    # 10_000-day window: even the 400-day-old rows are inside it.
    cfg = RetentionConfig(insider_days=10_000, news_days=10_000,
                          price_days=10_000, findings_days=10_000)
    init_db(db)
    with get_session() as s:
        deleted = prune_old_data(s, cfg)
    assert deleted == {
        "insider_transactions": 0, "news_items": 0, "prices": 0, "confluence_findings": 0,
    }
    assert _counts(db) == {
        "insider_transactions": 2, "news_items": 2, "prices": 2, "confluence_findings": 2,
    }
