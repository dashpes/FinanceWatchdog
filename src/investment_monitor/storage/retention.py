"""Retention / cleanup policy for the broad, market-wide data store.

Broad collection (congress / insider / news / prices) is universe-INDEPENDENT and
keeps everything, so the SQLite file grows without bound. This module is the bounded
counterweight: a per-source day-window prune that deletes rows older than a cutoff and
reclaims space with ``VACUUM``.

The whole policy is OPT-IN and fail-safe: every window defaults to ``0``, which means
"disabled / keep everything". With the default :class:`RetentionConfig`, ``prune_old_data``
is a strict NO-OP — it touches no rows and does not even VACUUM. Only the windows you
explicitly set to a positive number are enforced; each source is independent, so you can
prune chatty news daily while keeping years of insider history.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from . import ConfluenceFinding, InsiderTransaction, NewsItem, Price


def _utcnow() -> datetime:
    """Naive-UTC now, matching the storage layer's convention (DB cols are tz-naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RetentionConfig(BaseModel):
    """Per-source retention windows, in days. ``0`` = disabled (keep everything).

    Each window is independent and matches one table's natural time column:
    ``insider_days`` -> ``insider_transactions.trade_date``,
    ``news_days``    -> ``news_items.published_at``,
    ``price_days``   -> ``prices.date``,
    ``findings_days``-> ``confluence_findings.as_of_date``.
    """

    insider_days: int = Field(default=0, ge=0)
    news_days: int = Field(default=0, ge=0)
    price_days: int = Field(default=0, ge=0)
    findings_days: int = Field(default=0, ge=0)

    def any_enabled(self) -> bool:
        """True if at least one window is set (i.e. there is real work to do)."""
        return any(
            (self.insider_days, self.news_days, self.price_days, self.findings_days)
        )


def prune_old_data(session: Session, config: RetentionConfig) -> dict[str, int]:
    """Delete rows older than each ENABLED window and ``VACUUM`` to reclaim space.

    For every window > 0, rows whose time column is strictly older than
    ``now - window`` are deleted. Disabled windows (the default ``0``) are skipped
    entirely, so the default :class:`RetentionConfig` is a strict NO-OP that deletes
    nothing and does not VACUUM.

    Returns a per-table mapping of how many rows were deleted (tables left untouched by
    a disabled window are simply absent from the mapping).
    """
    deleted: dict[str, int] = {}

    # If nothing is enabled, do not even open a write transaction / VACUUM — strict NO-OP.
    if not config.any_enabled():
        return deleted

    now = _utcnow()

    # (table label, day-window, model, time column) — one entry per prunable source.
    plan = [
        ("insider_transactions", config.insider_days, InsiderTransaction, InsiderTransaction.trade_date),
        ("news_items", config.news_days, NewsItem, NewsItem.published_at),
        ("prices", config.price_days, Price, Price.date),
        ("confluence_findings", config.findings_days, ConfluenceFinding, ConfluenceFinding.as_of_date),
    ]

    for label, window_days, model, time_col in plan:
        if window_days <= 0:
            continue  # disabled window — keep everything for this source.
        cutoff = now - timedelta(days=window_days)
        # ``trade_date`` / ``date`` / ``as_of_date`` are DATE columns; ``published_at`` is
        # DATETIME — SQLite compares both fine against a datetime cutoff. Rows with a NULL
        # time column are left alone (NULL < cutoff is never true), which is intentional:
        # we never delete a row we cannot age.
        result = session.execute(delete(model).where(time_col < cutoff))
        n = result.rowcount or 0
        deleted[label] = n
        logger.info(
            "retention: pruned {n} rows from {t} older than {d}d (cutoff {c})",
            n=n, t=label, d=window_days, c=cutoff.date(),
        )

    session.commit()

    # VACUUM cannot run inside a transaction; commit above closed it. Reclaim file space.
    try:
        session.execute(text("VACUUM"))
    except Exception as exc:  # noqa: BLE001 - reclaiming space must never fail the prune.
        logger.warning("retention: VACUUM failed (continuing): {e}", e=exc)

    return deleted
