"""Promote confluence findings into active theses — the advisor acts on insights.

The insight engine surfaces cross-source "look here" findings (insider clusters +
volume + news). This bridge turns the strongest recent ones into ACTIVE theses, which
places them in the autonomous trader's universe so it sizes and (paper-)trades them
behind the SAME guardrail gate (allowlist, caps, cash-only, market-hours). It closes
the loop the whole project was building toward:

    broad data -> cross-source insight -> thesis -> trade -> outcome ledger -> learn

Safety: promotion only ADDS theses. Sizing is conservative (no-sim names get a small
conviction-proportional weight), every resulting order is still gated, and untradeable
names (no price) are skipped. The Phase 6 learning loop then records how these
follow-the-insiders bets actually play out.
"""

from __future__ import annotations

from datetime import date

from loguru import logger

from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    get_latest_price,
    get_recent_findings,
    get_session,
    get_thesis,
    init_db,
    save_thesis,
)

# Exit if the insiders' bet sours (their cluster thesis failed).
_DEFAULT_INVALIDATION = {"price_drop_pct": 25}


def _conviction_from_score(score: float) -> float:
    """Map a confluence score (~2-12) to a conservative conviction band [0.35, 0.85]."""
    return max(0.35, min(0.85, 0.4 + 0.05 * float(score)))


def promote_confluence_findings(
    session,
    *,
    min_score: float = 4.0,
    max_promotions: int = 5,
    account_id: str | None = None,
) -> list[str]:
    """Promote the strongest recent confluence findings to ACTIVE theses.

    Skips names already tracked (a non-invalidated thesis) and names with no price
    (untradeable); caps at ``max_promotions``. Returns the promoted symbols.
    """
    findings = get_recent_findings(session, min_score=min_score, limit=max_promotions * 5)
    promoted: list[str] = []
    for f in findings:
        if len(promoted) >= max_promotions:
            break
        existing = get_thesis(session, f.ticker, account_id)
        if existing is not None and existing.status != ThesisStatus.INVALIDATED.value:
            continue  # already maintained — don't duplicate
        price = get_latest_price(session, f.ticker)
        if price is None or price.close is None:
            continue  # untradeable / no price data — skip
        conviction = _conviction_from_score(f.score)
        thesis = Thesis(
            symbol=f.ticker,
            account_id=account_id,
            narrative=f.narrative,
            conviction=conviction,
            entry_conditions={"entry_composite": f.score, "entry_price": float(price.close)},
            invalidation_conditions=dict(_DEFAULT_INVALIDATION),
            evidence_refs={"confluence_finding_id": f.id, "kind": f.kind},
            status=ThesisStatus.ACTIVE.value,
            conviction_history=[{"conviction": conviction, "trigger": f"confluence:{f.kind}"}],
        )
        save_thesis(session, thesis)
        promoted.append(f.ticker)
        logger.info(
            "promoted confluence {t} (score {s:.1f} -> conviction {c:.2f})",
            t=f.ticker, s=f.score, c=conviction,
        )
    return promoted


def run_insight_promotion(
    settings=None, *, config=None, min_score: float = 4.0, max_promotions: int = 5,
    today: date | None = None,
) -> dict:
    """Detect confluence findings and promote the strongest to theses (CLI/cron entry).

    Returns ``{findings, promoted}``. Account-less (global) theses so they apply to
    whichever account the robo trades, matching ``promote_candidates``.
    """
    from investment_monitor.analysis.confluence import detect_confluence
    from investment_monitor.config import get_settings

    settings = settings or get_settings()
    init_db(settings.db_path)
    with get_session() as session:
        findings = detect_confluence(session, config, today=today)
        promoted = promote_confluence_findings(
            session, min_score=min_score, max_promotions=max_promotions,
        )
    return {"findings": len(findings), "promoted": promoted}
