"""CRUD + aggregation for the learning/feedback ledger (Phase 6).

The single funnel for the robo advisor's feedback loop. ``record_learning_event``
is the only writer (mirroring ``record_conviction_update`` being the sole
conviction_history writer); ``accuracy_stats_for_symbol`` is the compact, EWMA-
smoothed read used by sizing and the re-eval prompt — it reduces the full event
history to four numbers so nothing bloats the LLM context.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .learning_models import LEARNING_KIND_OUTCOME, LearningEvent


def _utcnow() -> datetime:
    # Naive UTC, matching the schema's DateTime columns (server_default=func.now()).
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utctoday() -> date:
    return _utcnow().date()


def outcome_metrics(realized_return: float, conviction: float) -> tuple[int, float]:
    """Map a realized return + held conviction to ``(direction_correct, brier)`` (pure).

    * ``direction_correct``: 1 if the (long-only) thesis made money, else 0 — SIGN
      ONLY, deliberately robust to the magnitude/look-ahead imprecision in stored
      cumulative returns.
    * ``brier``: ``(conviction - outcome)^2``; lower is better, penalizing a confident
      thesis that was wrong the most.
    """
    outcome = 1 if float(realized_return) > 0 else 0
    conv = max(0.0, min(1.0, float(conviction)))
    return outcome, (conv - outcome) ** 2


def record_learning_event(
    session: Session,
    *,
    kind: str,
    symbol: str | None = None,
    account_id: str | None = None,
    thesis_id: int | None = None,
    conviction: float | None = None,
    realized_return: float | None = None,
    direction_correct: int | None = None,
    brier: float | None = None,
    applied: bool = False,
    before_value: float | None = None,
    after_value: float | None = None,
    as_of_date: date | None = None,
    payload: dict | None = None,
    note: str | None = None,
) -> int:
    """Append one event to the ledger, returning its id (the single write funnel)."""
    event = LearningEvent(
        kind=kind,
        symbol=symbol,
        account_id=account_id,
        thesis_id=thesis_id,
        conviction=conviction,
        realized_return=realized_return,
        direction_correct=direction_correct,
        brier=brier,
        applied=applied,
        before_value=before_value,
        after_value=after_value,
        as_of_date=as_of_date,
        payload=payload or {},
        note=note,
    )
    session.add(event)
    session.flush()
    return event.id


def record_thesis_outcome(
    session: Session,
    *,
    symbol: str,
    conviction_at_eval: float,
    realized_return: float,
    account_id: str | None = None,
    thesis_id: int | None = None,
    as_of_date: date | None = None,
    payload: dict | None = None,
) -> int:
    """Record a realized thesis outcome (kind=thesis_outcome) with derived metrics.

    ``as_of_date`` defaults to today (UTC) and is the de-dup key for the production
    path (see ``outcome_exists_for_date``).
    """
    direction_correct, brier = outcome_metrics(realized_return, conviction_at_eval)
    return record_learning_event(
        session,
        kind=LEARNING_KIND_OUTCOME,
        symbol=symbol,
        account_id=account_id,
        thesis_id=thesis_id,
        conviction=max(0.0, min(1.0, float(conviction_at_eval))),
        realized_return=float(realized_return),
        direction_correct=direction_correct,
        brier=brier,
        as_of_date=as_of_date if as_of_date is not None else _utctoday(),
        payload=payload,
    )


def outcome_exists_for_date(
    session: Session, symbol: str, as_of_date: date, *, account_id: str | None = None
) -> bool:
    """True if a thesis_outcome for this symbol/account is already recorded for the day.

    The production capture path uses this to record at most ONE outcome per symbol
    per UTC day, regardless of how many times the maintenance loop re-evaluates the
    thesis — decoupling the accuracy signal from run cadence.
    """
    stmt = select(LearningEvent.id).where(
        LearningEvent.kind == LEARNING_KIND_OUTCOME,
        LearningEvent.symbol == symbol,
        LearningEvent.as_of_date == as_of_date,
    )
    if account_id:
        stmt = stmt.where(LearningEvent.account_id == account_id)
    return session.scalar(stmt.limit(1)) is not None


def get_recent_outcomes(
    session: Session, symbol: str, *, account_id: str | None = None, limit: int = 50
) -> list[LearningEvent]:
    """Most-recent realized-outcome events for a symbol (newest first)."""
    stmt = select(LearningEvent).where(
        LearningEvent.kind == LEARNING_KIND_OUTCOME,
        LearningEvent.symbol == symbol,
    )
    if account_id:
        stmt = stmt.where(LearningEvent.account_id == account_id)
    stmt = stmt.order_by(LearningEvent.id.desc()).limit(max(1, limit))
    return list(session.scalars(stmt))


def get_outcome_symbols(session: Session, *, account_id: str | None = None) -> list[str]:
    """Distinct symbols that have at least one recorded outcome (sorted)."""
    stmt = select(LearningEvent.symbol).where(
        LearningEvent.kind == LEARNING_KIND_OUTCOME,
        LearningEvent.symbol.is_not(None),
    )
    if account_id:
        stmt = stmt.where(LearningEvent.account_id == account_id)
    return sorted({s for s in session.scalars(stmt.distinct()) if s})


def accuracy_stats_for_symbol(
    session: Session,
    symbol: str,
    *,
    account_id: str | None = None,
    ewma_halflife: float = 10.0,
    recent_window: int = 20,
) -> dict:
    """Compact accuracy aggregate for a symbol — the distilled feedback signal.

    Returns ``{n, hit_rate, brier, ewma_hit_rate}``. ``ewma_hit_rate`` is recency-
    weighted over the most recent ``recent_window`` outcomes (newest weighted most,
    half-life ``ewma_halflife`` events). With no data it returns a neutral prior
    (hit_rate 0.5, n 0) so callers can treat "no evidence" as "no tilt".
    """
    events = get_recent_outcomes(session, symbol, account_id=account_id, limit=recent_window)
    n = len(events)
    if n == 0:
        return {"n": 0, "hit_rate": 0.5, "brier": 0.25, "ewma_hit_rate": 0.5}

    hits = [int(e.direction_correct or 0) for e in events]
    briers = [float(e.brier) for e in events if e.brier is not None]
    hit_rate = sum(hits) / n
    brier = (sum(briers) / len(briers)) if briers else 0.25

    # EWMA half-life in "events": newest (age 0) gets weight 1.0, decaying by half
    # every `ewma_halflife` events. `events` is newest-first, so index == age.
    hl = max(1e-6, float(ewma_halflife))
    wsum = 0.0
    acc = 0.0
    for age, e in enumerate(events):
        w = 0.5 ** (age / hl)
        acc += w * int(e.direction_correct or 0)
        wsum += w
    ewma_hit_rate = acc / wsum if wsum > 0 else hit_rate

    return {"n": n, "hit_rate": hit_rate, "brier": brier, "ewma_hit_rate": ewma_hit_rate}
