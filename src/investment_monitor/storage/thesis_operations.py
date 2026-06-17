"""CRUD operations for the thesis store (Phase 3)."""

from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .thesis_models import LIVE_THESIS_STATUSES, Thesis, ThesisStatus


def _utcnow() -> datetime:
    # Naive UTC, matching the schema's DateTime columns (server_default=func.now()).
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_active_theses(session: Session, account_id: str | None = None) -> list[Thesis]:
    """Theses that currently drive the target allocation (ACTIVE or WATCH).

    Account-less (global) theses apply to every account; passing ``account_id``
    additionally includes that account's scoped theses.
    """
    stmt = select(Thesis).where(Thesis.status.in_(LIVE_THESIS_STATUSES))
    if account_id:
        stmt = stmt.where(or_(Thesis.account_id == account_id, Thesis.account_id.is_(None)))
    stmt = stmt.order_by(Thesis.symbol)
    return list(session.scalars(stmt))


def get_all_theses(session: Session, account_id: str | None = None) -> list[Thesis]:
    """All theses regardless of status (newest first)."""
    stmt = select(Thesis)
    if account_id:
        stmt = stmt.where(Thesis.account_id == account_id)
    stmt = stmt.order_by(Thesis.updated_at.desc())
    return list(session.scalars(stmt))


def get_thesis(session: Session, symbol: str, account_id: str | None = None) -> Thesis | None:
    """The most-recently-updated non-EXITED thesis for a symbol, if any."""
    stmt = select(Thesis).where(
        Thesis.symbol == symbol, Thesis.status != ThesisStatus.EXITED.value
    )
    if account_id:
        stmt = stmt.where(Thesis.account_id == account_id)
    stmt = stmt.order_by(Thesis.updated_at.desc()).limit(1)
    return session.scalar(stmt)


def save_thesis(session: Session, thesis: Thesis) -> int:
    """Persist a thesis, returning its id."""
    session.add(thesis)
    session.flush()
    return thesis.id


def record_conviction_update(
    session: Session, thesis: Thesis, new_conviction: float, trigger: str
) -> None:
    """Set conviction (clamped 0-1), stamp the re-eval time, append to the audit history."""
    clamped = max(0.0, min(1.0, float(new_conviction)))
    # Reassign (not in-place mutate) so SQLAlchemy flags the JSON column dirty.
    history = list(thesis.conviction_history or [])
    history.append({"ts": _utcnow().isoformat(), "conviction": clamped, "trigger": trigger})
    thesis.conviction_history = history
    thesis.conviction = clamped
    thesis.last_evaluated_at = _utcnow()
    session.flush()


def set_target_weight(session: Session, thesis: Thesis, weight: float) -> None:
    """Cache the sized target weight on the thesis."""
    thesis.target_weight = max(0.0, float(weight))
    session.flush()


def invalidate_thesis(session: Session, thesis: Thesis, reason: str) -> None:
    """Trip a thesis: status INVALIDATED, conviction -> 0 (forces a sell-toward-zero)."""
    history = list(thesis.conviction_history or [])
    history.append({"ts": _utcnow().isoformat(), "conviction": 0.0, "trigger": f"invalidated: {reason}"})
    thesis.conviction_history = history
    thesis.conviction = 0.0
    thesis.target_weight = 0.0
    thesis.status = ThesisStatus.INVALIDATED.value
    thesis.last_evaluated_at = _utcnow()
    session.flush()


def get_active_symbols(session: Session, account_id: str | None = None) -> set[str]:
    """Set of symbols with a live (ACTIVE/WATCH) thesis — the dynamic allowlist."""
    return {t.symbol for t in get_active_theses(session, account_id)}
