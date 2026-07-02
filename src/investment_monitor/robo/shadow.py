"""Shadow ledger maintenance — track the theses the robo did NOT trade.

Recording happens at the decision points (confluence promotion records its own
skips; ``sync_gate_reject_shadows`` scans gate-rejected buys; ``record_discovery
_shadows`` catches near-miss discovery candidates). This module marks open entries
against fresh prices, closes them at a fixed horizon, and writes the frozen
counterfactual into the learning ledger under ``shadow_outcome``.

Safety invariants (mirrors the Phase 6 learning loop):
- fail-open: shadow bookkeeping is wrapped by the caller and can never block a run;
- read-only on trading state: it never touches theses, orders, or sizing;
- distinct learning kind: hypothetical outcomes never contaminate real accuracy.
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger
from sqlalchemy import select

from investment_monitor.storage import (
    LEARNING_KIND_OUTCOME,
    LEARNING_KIND_SHADOW_OUTCOME,
    LearningEvent,
    RoboOrder,
    StockCandidate,
    get_latest_price,
    outcome_metrics,
    record_learning_event,
)
from investment_monitor.storage.shadow_models import (
    SHADOW_SOURCE_DISCOVERY,
    SHADOW_SOURCE_GATE,
)
from investment_monitor.storage.shadow_operations import (
    close_shadow_entry,
    get_open_shadow_entries,
    mark_shadow_entry,
    record_shadow_entry,
    shadow_ref_ids,
    shadow_summary,
)

# Counterfactuals freeze at this horizon — long enough for an insider-cluster thesis
# to play out, short enough that the comparison set stays current.
DEFAULT_HORIZON_DAYS = 90
# A mark older than this is stale (the name fell out of price collection); the entry
# still closes at horizon on its last known price, flagged in the payload.
_STALE_PRICE_DAYS = 10


def sync_gate_reject_shadows(session, *, days: int = 3, account_id: str | None = None) -> int:
    """Open shadow entries for recently gate-rejected BUY orders (scan, no trade-path hook).

    Reads ``robo_orders`` after the fact so the trading pipeline needs no changes and
    a shadow failure can never affect order handling. ``ref_id`` = the order row id,
    so each rejected order is shadowed at most once ever.
    """
    seen = shadow_ref_ids(session, SHADOW_SOURCE_GATE)
    stmt = select(RoboOrder).where(
        RoboOrder.gate_accepted.is_(False),
        RoboOrder.side == "buy",
    ).order_by(RoboOrder.id.desc()).limit(200)
    n = 0
    cutoff = date.today() - timedelta(days=days)
    for order in session.scalars(stmt):
        if order.id in seen or not order.symbol:
            continue
        odate = order.created_at.date() if order.created_at else date.today()
        if odate < cutoff:
            continue
        price = get_latest_price(session, order.symbol)
        entry_price = float(price.close) if price and price.close else None
        if record_shadow_entry(
            session,
            symbol=order.symbol,
            source=SHADOW_SOURCE_GATE,
            skip_reason=f"gate:{order.gate_code or 'rejected'}",
            entry_date=odate,
            entry_price=entry_price,
            account_id=account_id,
            ref_id=order.id,
            detail=order.rationale or order.reason,
            payload={"run_id": order.run_id, "gate_reason": order.gate_reason},
        ) is not None:
            n += 1
    return n


def record_discovery_shadows(
    session, *, score_floor: float, band: float = 15.0, account_id: str | None = None
) -> int:
    """Open shadow entries for discovery candidates just UNDER the promotion floor.

    Near-misses (within ``band`` composite points below the floor) are the candidates
    the floor's exact placement decides on — exactly the counterfactual needed to
    tune it. Far-below names are noise and are not tracked.
    """
    stmt = select(StockCandidate).where(
        StockCandidate.composite_score.is_not(None),
        StockCandidate.composite_score < score_floor,
        StockCandidate.composite_score >= score_floor - band,
    )
    n = 0
    for cand in session.scalars(stmt):
        price = get_latest_price(session, cand.ticker)
        if price is None or not price.close:
            continue  # unpriceable — a counterfactual we could never evaluate
        composite = float(cand.composite_score)
        if record_shadow_entry(
            session,
            symbol=cand.ticker,
            source=SHADOW_SOURCE_DISCOVERY,
            skip_reason="below_score_floor",
            entry_date=date.today(),
            entry_price=float(price.close),
            account_id=account_id,
            ref_id=cand.id,
            detail=f"discovery composite {composite:.0f} < floor {score_floor:.0f}",
            score=composite,
            conviction=max(0.0, min(1.0, composite / 100.0)),
        ) is not None:
            n += 1
    return n


def evaluate_shadow_entries(
    session,
    *,
    today: date | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict:
    """Mark open entries to the latest price; close + record those past the horizon.

    Closing writes one ``shadow_outcome`` learning event carrying the same
    direction/brier metrics as real outcomes, so traded-vs-skipped comparisons read
    from one ledger.
    """
    today = today or date.today()
    marked = closed = 0
    for entry in get_open_shadow_entries(session):
        if entry.entry_price is None or entry.entry_price <= 0:
            continue  # recorded for the record; never evaluable
        price = get_latest_price(session, entry.symbol)
        if price is None or not price.close:
            continue
        ret = float(price.close) / float(entry.entry_price) - 1.0
        if (today - entry.entry_date).days >= horizon_days:
            stale = (today - price.date).days > _STALE_PRICE_DAYS if price.date else True
            close_shadow_entry(
                session, entry,
                exit_date=price.date or today, exit_price=float(price.close),
                realized_return=ret,
            )
            conv = entry.conviction if entry.conviction is not None else 0.5
            direction_correct, brier = outcome_metrics(ret, conv)
            record_learning_event(
                session,
                kind=LEARNING_KIND_SHADOW_OUTCOME,
                symbol=entry.symbol,
                account_id=entry.account_id,
                conviction=max(0.0, min(1.0, float(conv))),
                realized_return=ret,
                direction_correct=direction_correct,
                brier=brier,
                as_of_date=today,
                payload={
                    "source": entry.source, "skip_reason": entry.skip_reason,
                    "ref_id": entry.ref_id, "horizon_days": horizon_days,
                    "stale_price": stale,
                },
                note=f"shadow close: {entry.skip_reason}",
            )
            closed += 1
        else:
            mark_shadow_entry(session, entry, realized_return=ret)
            marked += 1
    if marked or closed:
        logger.info(f"shadow ledger: {marked} marked, {closed} closed on {today}")
    return {"marked": marked, "closed": closed}


def maintain_shadow_ledger(
    session, *, score_floor: float | None = None, account_id: str | None = None
) -> dict:
    """One idempotent maintenance pass: sync recorders, then mark/close. Fail-open
    per stage so a bad price series can't abort the whole pass."""
    out = {"gate": 0, "discovery": 0, "marked": 0, "closed": 0}
    try:
        out["gate"] = sync_gate_reject_shadows(session, account_id=account_id)
    except Exception as exc:  # noqa: BLE001 - bookkeeping must never block a run
        logger.warning(f"shadow gate sync failed: {exc}")
    if score_floor is not None:
        try:
            out["discovery"] = record_discovery_shadows(
                session, score_floor=score_floor, account_id=account_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"shadow discovery sync failed: {exc}")
    try:
        out.update(evaluate_shadow_entries(session))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"shadow evaluation failed: {exc}")
    return out


def _real_outcome_stats(session) -> dict:
    """Compact aggregate over REAL thesis outcomes (the comparison baseline)."""
    stmt = select(LearningEvent).where(LearningEvent.kind == LEARNING_KIND_OUTCOME)
    rets = [float(e.realized_return) for e in session.scalars(stmt) if e.realized_return is not None]
    if not rets:
        return {"n": 0, "hit_rate": None, "avg_return": None}
    return {
        "n": len(rets),
        "hit_rate": sum(1 for r in rets if r > 0) / len(rets),
        "avg_return": sum(rets) / len(rets),
    }


def shadow_report(session) -> dict:
    """Traded-vs-skipped comparison: real outcome stats + per-source shadow stats."""
    return {"real": _real_outcome_stats(session), "shadow": shadow_summary(session)}
