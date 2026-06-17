"""Autonomous stock selection (Phase 4).

Promotes names from the existing discovery funnel (high-composite ``StockCandidate``
rows) into *active theses*, which makes them part of the tradeable universe. The
promotion gate is deterministic ('fully auto behind a score floor'); the LLM only
enriches the resulting thesis narrative/conviction when available.

Lives in ``robo`` (not ``research``) so it depends only on the storage layer and
does not pull the research package's heavy data-collector imports.

Safety: promotion only adds theses. Whether any resulting buy executes is still
decided by the guardrail gate (allowlist + ``no_active_thesis`` + caps + dry-run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    get_latest_price,
    get_latest_report,
    get_thesis,
    get_top_candidates,
    save_thesis,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.analysis.thesis_evaluator import ThesisEvaluator
    from investment_monitor.robo.config import RoboConfig

# Conservative default exits for an auto-promoted name (used by the deterministic
# fallback; the LLM may override when it generates the thesis).
_DEFAULT_INVALIDATION = {"composite_drop": 15, "price_drop_pct": 30}


def _eligible_candidates(session: "Session", config: "RoboConfig") -> list:
    ac = config.autonomy
    candidates = get_top_candidates(session, limit=50, min_score=ac.score_floor)
    if not ac.require_buy_recommendation:
        return candidates
    out = []
    for cand in candidates:
        report = get_latest_report(session, cand.ticker)
        rec = (report.recommendation if report else "") or ""
        if rec.lower() in ("buy", "strong_buy"):
            out.append(cand)
    return out


def _fallback_thesis(session: "Session", candidate, account_id: str | None) -> Thesis:
    """A deterministic thesis from the candidate's score (used when no LLM is available)."""
    composite = float(candidate.composite_score or 50.0)
    price = get_latest_price(session, candidate.ticker)
    conviction = max(0.0, min(1.0, composite / 100.0))
    return Thesis(
        symbol=candidate.ticker,
        account_id=account_id,
        narrative=f"Auto-promoted from discovery funnel (composite score {composite:.0f}).",
        conviction=conviction,
        entry_conditions={
            "entry_composite": composite,
            "entry_price": float(price.close) if price and price.close is not None else None,
        },
        invalidation_conditions=dict(_DEFAULT_INVALIDATION),
        status=ThesisStatus.DRAFT.value,
        conviction_history=[{"conviction": conviction, "trigger": "auto_promote"}],
    )


def promote_candidates(
    session: "Session",
    config: "RoboConfig",
    *,
    evaluator: "ThesisEvaluator | None" = None,
    account_id: str | None = None,
) -> list[str]:
    """Promote eligible discovery candidates to ACTIVE theses. Returns promoted symbols.

    Skips names that already have a (non-exited) thesis, caps at
    ``max_promotions_per_run``, and prefers an LLM-generated thesis, falling back to
    a deterministic score-derived one so selection works without Ollama.
    """
    ac = config.autonomy
    if not ac.enabled or ac.max_promotions_per_run <= 0:
        return []

    promoted: list[str] = []
    for candidate in _eligible_candidates(session, config):
        if len(promoted) >= ac.max_promotions_per_run:
            break
        existing = get_thesis(session, candidate.ticker, account_id)
        if existing is not None and existing.status != ThesisStatus.INVALIDATED.value:
            continue  # already maintained (active/watch/draft) — don't duplicate.
        # An INVALIDATED prior thesis does NOT block re-promotion: if the name has
        # re-cleared the score floor, a fresh thesis supersedes it.

        thesis = None
        if evaluator is not None:
            thesis = evaluator.generate(session, candidate.ticker, account_id=account_id)
        if thesis is None:
            thesis = _fallback_thesis(session, candidate, account_id)
            save_thesis(session, thesis)

        thesis.status = ThesisStatus.ACTIVE.value
        session.flush()
        promoted.append(candidate.ticker)
        logger.info("auto-promoted {t} (conviction {c:.2f})", t=candidate.ticker, c=thesis.conviction)

    return promoted
