"""CRUD + aggregation for the shadow ledger (considered-but-not-traded theses).

``record_shadow_entry`` is the single write funnel; it enforces one OPEN entry per
symbol/source so a candidate skipped on ten consecutive runs yields one tracked
counterfactual, not ten autocorrelated copies. ``shadow_summary`` reduces the ledger
to compact per-source aggregates for the CLI report and (later) the re-eval prompt.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .shadow_models import (
    SHADOW_STATUS_CLOSED,
    SHADOW_STATUS_OPEN,
    ShadowEntry,
)


def _utcnow() -> datetime:
    # Naive UTC, matching the schema's DateTime columns (server_default=func.now()).
    return datetime.now(timezone.utc).replace(tzinfo=None)


def has_open_shadow(
    session: Session, symbol: str, source: str, *, account_id: str | None = None
) -> bool:
    """True if an OPEN shadow entry already tracks this symbol/source."""
    stmt = select(ShadowEntry.id).where(
        ShadowEntry.symbol == symbol,
        ShadowEntry.source == source,
        ShadowEntry.status == SHADOW_STATUS_OPEN,
    )
    if account_id:
        stmt = stmt.where(ShadowEntry.account_id == account_id)
    return session.scalar(stmt.limit(1)) is not None


def shadow_ref_ids(session: Session, source: str) -> set[int]:
    """All ref_ids ever shadowed for a source (dedup key for scan-based recorders)."""
    stmt = select(ShadowEntry.ref_id).where(
        ShadowEntry.source == source, ShadowEntry.ref_id.is_not(None)
    )
    return {r for r in session.scalars(stmt)}


def record_shadow_entry(
    session: Session,
    *,
    symbol: str,
    source: str,
    skip_reason: str,
    entry_date: date,
    entry_price: float | None,
    account_id: str | None = None,
    ref_id: int | None = None,
    detail: str | None = None,
    score: float | None = None,
    conviction: float | None = None,
    payload: dict | None = None,
) -> int | None:
    """Open a shadow entry (the single write funnel), returning its id.

    Returns None (no write) when an OPEN entry already tracks the symbol/source or
    an entry already exists for the same symbol/source/day — idempotent under the
    research loop's many-runs-per-day cadence.
    """
    if has_open_shadow(session, symbol, source, account_id=account_id):
        return None
    dup = session.scalar(
        select(ShadowEntry.id).where(
            ShadowEntry.symbol == symbol,
            ShadowEntry.source == source,
            ShadowEntry.entry_date == entry_date,
        ).limit(1)
    )
    if dup is not None:
        return None
    entry = ShadowEntry(
        symbol=symbol,
        source=source,
        skip_reason=skip_reason,
        entry_date=entry_date,
        entry_price=entry_price,
        account_id=account_id,
        ref_id=ref_id,
        detail=detail,
        score=score,
        conviction=conviction,
        payload=payload or {},
    )
    session.add(entry)
    session.flush()
    return entry.id


def get_open_shadow_entries(session: Session, *, limit: int = 1000) -> list[ShadowEntry]:
    """All OPEN entries, oldest first (so horizon closes process in order)."""
    stmt = (
        select(ShadowEntry)
        .where(ShadowEntry.status == SHADOW_STATUS_OPEN)
        .order_by(ShadowEntry.entry_date.asc())
        .limit(max(1, limit))
    )
    return list(session.scalars(stmt))


def get_shadow_entries(
    session: Session, *, status: str | None = None, source: str | None = None, limit: int = 200
) -> list[ShadowEntry]:
    """Entries filtered by status/source, newest first (display/report)."""
    stmt = select(ShadowEntry)
    if status:
        stmt = stmt.where(ShadowEntry.status == status)
    if source:
        stmt = stmt.where(ShadowEntry.source == source)
    stmt = stmt.order_by(ShadowEntry.entry_date.desc(), ShadowEntry.id.desc()).limit(max(1, limit))
    return list(session.scalars(stmt))


def mark_shadow_entry(session: Session, entry: ShadowEntry, *, realized_return: float) -> None:
    """Refresh an open entry's running counterfactual mark."""
    entry.realized_return = float(realized_return)
    entry.last_evaluated_at = _utcnow()
    session.flush()


def close_shadow_entry(
    session: Session,
    entry: ShadowEntry,
    *,
    exit_date: date,
    exit_price: float,
    realized_return: float,
) -> None:
    """Freeze an entry at its horizon: final mark + closed status."""
    entry.status = SHADOW_STATUS_CLOSED
    entry.exit_date = exit_date
    entry.exit_price = float(exit_price)
    entry.realized_return = float(realized_return)
    entry.last_evaluated_at = _utcnow()
    session.flush()


def shadow_summary(session: Session) -> dict:
    """Compact per-source aggregates: {source: {open, closed, hit_rate, avg_return}}.

    hit_rate/avg_return are over CLOSED entries only (frozen outcomes); ``open_mark``
    is the mean running mark of still-open entries (best-effort context).
    """
    out: dict[str, dict] = {}
    for e in get_shadow_entries(session, limit=10_000):
        s = out.setdefault(
            e.source,
            {"open": 0, "closed": 0, "hit_rate": None, "avg_return": None,
             "_closed_rets": [], "_open_rets": [], "open_mark": None},
        )
        if e.status == SHADOW_STATUS_CLOSED and e.realized_return is not None:
            s["closed"] += 1
            s["_closed_rets"].append(float(e.realized_return))
        elif e.status == SHADOW_STATUS_OPEN:
            s["open"] += 1
            if e.realized_return is not None:
                s["_open_rets"].append(float(e.realized_return))
    for s in out.values():
        closed = s.pop("_closed_rets")
        open_rets = s.pop("_open_rets")
        if closed:
            s["hit_rate"] = sum(1 for r in closed if r > 0) / len(closed)
            s["avg_return"] = sum(closed) / len(closed)
        if open_rets:
            s["open_mark"] = sum(open_rets) / len(open_rets)
    return out
