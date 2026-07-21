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
    session: Session, thesis: Thesis, new_conviction: float, trigger: str,
    *, evidence_hash: str | None = None,
) -> None:
    """Set conviction (clamped 0-1), stamp the re-eval time, append to the audit history.

    ``evidence_hash`` (LLM re-evals only) fingerprints the evidence the model saw, so
    the next re-eval can skip the call when nothing changed. Callers withhold it when
    the update was rate-limit clamped — the un-absorbed evidence must trigger a re-run.
    """
    clamped = max(0.0, min(1.0, float(new_conviction)))
    # Reassign (not in-place mutate) so SQLAlchemy flags the JSON column dirty.
    history = list(thesis.conviction_history or [])
    entry: dict = {"ts": _utcnow().isoformat(), "conviction": clamped, "trigger": trigger}
    if evidence_hash:
        entry["evidence_hash"] = evidence_hash
    history.append(entry)
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


def exit_thesis(session: Session, thesis: Thesis, reason: str) -> None:
    """Conclude a thesis that PLAYED OUT (take-profit/horizon): status EXITED, conviction 0.

    Same sell mechanics as invalidation — the symbol drops out of the live thesis set,
    so the next rebalance trims the held position to a 0 target via the full-exit path.
    EXITED (unlike INVALIDATED) also vanishes from ``get_thesis``, so a fresh signal can
    re-promote the name immediately; callers guard against same-signal re-entry.
    """
    history = list(thesis.conviction_history or [])
    history.append({"ts": _utcnow().isoformat(), "conviction": 0.0, "trigger": f"exited: {reason}"})
    thesis.conviction_history = history
    thesis.conviction = 0.0
    thesis.target_weight = 0.0
    thesis.status = ThesisStatus.EXITED.value
    thesis.last_evaluated_at = _utcnow()
    session.flush()


def bench_thesis(session: Session, thesis: Thesis, reason: str) -> None:
    """Demote a thesis to WATCH (benched): keep it and its history, stop sizing it.

    NOT an exit — conviction is preserved as the revival signal. A benched name gets
    one LLM re-look per ``autonomy.bench_reeval_days`` and returns to ACTIVE when its
    conviction recovers over the floor (or a fresh confluence finding revives it).
    """
    history = list(thesis.conviction_history or [])
    history.append({"ts": _utcnow().isoformat(), "conviction": thesis.conviction,
                    "trigger": f"benched: {reason}"})
    thesis.conviction_history = history
    thesis.target_weight = 0.0
    thesis.status = ThesisStatus.WATCH.value
    session.flush()


def update_high_water(session: Session, thesis: Thesis, price: float | None) -> None:
    """Monotonically raise the thesis's high-water mark (no-op on None/lower prices)."""
    if price is None or price <= 0:
        return
    current = thesis.high_water_mark
    if current is None or price > current:
        thesis.high_water_mark = float(price)
        session.flush()


def get_last_exited_thesis(
    session: Session, symbol: str, account_id: str | None = None
) -> Thesis | None:
    """The most-recently-updated EXITED thesis for a symbol (re-entry-guard lookup)."""
    stmt = select(Thesis).where(
        Thesis.symbol == symbol, Thesis.status == ThesisStatus.EXITED.value
    )
    if account_id:
        stmt = stmt.where(Thesis.account_id == account_id)
    stmt = stmt.order_by(Thesis.updated_at.desc()).limit(1)
    return session.scalar(stmt)


def get_active_symbols(session: Session, account_id: str | None = None) -> set[str]:
    """Set of symbols with a live (ACTIVE/WATCH) thesis — the dynamic allowlist."""
    return {t.symbol for t in get_active_theses(session, account_id)}
