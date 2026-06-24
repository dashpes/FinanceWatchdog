"""Promote confluence findings into active theses — the advisor acts on insights.

The insight engine surfaces cross-source "look here" findings (insider clusters +
volume + news). This bridge turns the strongest RECENT ones into ACTIVE theses, which
places them in the autonomous trader's universe so it sizes and (paper-/real-)trades
them behind the SAME guardrail gate. It closes the loop the project was building toward:

    broad data -> cross-source insight -> thesis -> trade -> outcome ledger -> learn

Safety guards (this trades REAL money on a small live account):
- recency window: only act on fresh findings (never a stale multi-week-old one);
- score-ranked + one-per-ticker, so the cap promotes the genuinely strongest;
- falling-knife guard: a self-invalidated name is re-promoted only on a NEW finding;
- liquidity floor: skip sub-$min_price / thin-dollar-volume / stale-price shells;
- already-ran filter: skip names already up big (the signal is priced in);
- sizing/caps/gate still bound every resulting order downstream.
"""

from __future__ import annotations

from datetime import date

from loguru import logger

from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    get_prices,
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


def _liquidity(
    session, ticker: str, *, min_price: float, min_dollar_volume: float, lookback: int = 20
) -> tuple[bool, float | None]:
    """Return (is_tradeable, latest_close). Tradeable = a fresh price >= min_price AND a
    trailing average dollar-volume >= min_dollar_volume (excludes penny/illiquid shells)."""
    prices = get_prices(session, ticker, days=lookback + 8)  # newest-first, recent only
    if not prices or prices[0].close is None:
        return False, None
    latest_close = float(prices[0].close)
    if latest_close < min_price:
        return False, latest_close
    dollar_vols = [
        float(p.volume) * float(p.close)
        for p in prices[:lookback] if p.volume and p.close
    ]
    avg_dollar_vol = (sum(dollar_vols) / len(dollar_vols)) if dollar_vols else 0.0
    return (avg_dollar_vol >= min_dollar_volume), latest_close


def promote_confluence_findings(
    session,
    *,
    min_score: float = 4.0,
    max_promotions: int = 5,
    account_id: str | None = None,
    max_age_days: int = 3,
    min_price: float = 3.0,
    min_dollar_volume: float = 250_000.0,
    max_run_pct: float = 40.0,
) -> list[str]:
    """Promote the strongest fresh, liquid confluence findings to ACTIVE theses."""
    raw = get_recent_findings(
        session, min_score=min_score, limit=max_promotions * 6, max_age_days=max_age_days
    )
    # One candidate per ticker (strongest), strongest first.
    best: dict[str, "object"] = {}
    for f in raw:
        if f.ticker not in best or f.score > best[f.ticker].score:
            best[f.ticker] = f
    candidates = sorted(best.values(), key=lambda f: f.score, reverse=True)

    promoted: list[str] = []
    for f in candidates:
        if len(promoted) >= max_promotions:
            break
        existing = get_thesis(session, f.ticker, account_id)
        if existing is not None:
            if existing.status != ThesisStatus.INVALIDATED.value:
                continue  # actively tracked — don't duplicate
            # Self-invalidated: re-promote ONLY on a genuinely fresh finding (never the
            # stale one that drove the first buy) so we can't auto-rebuy a falling knife.
            le = existing.last_evaluated_at
            same_finding = (existing.evidence_refs or {}).get("confluence_finding_id") == f.id
            # Stale only if STRICTLY older than the last evaluation. A genuinely new
            # finding produced on the SAME calendar day (a fresh, stronger cross-source
            # signal) must still be able to re-promote; same-day must not read as stale.
            # The falling-knife guard against re-buying the SAME finding stays in
            # `same_finding`, so a same-day re-promote can only come from a new finding.
            stale = le is not None and f.as_of_date is not None and f.as_of_date < le.date()
            if same_finding or stale:
                continue
        # Already run up big since the buys? The insider signal is priced in — skip.
        if f.price_change_pct is not None and f.price_change_pct > max_run_pct:
            continue
        tradeable, close = _liquidity(
            session, f.ticker, min_price=min_price, min_dollar_volume=min_dollar_volume
        )
        if not tradeable or close is None:
            continue  # penny / illiquid / stale price — not safe to auto-trade
        conviction = _conviction_from_score(f.score)
        thesis = Thesis(
            symbol=f.ticker,
            account_id=account_id,
            narrative=f.narrative,
            conviction=conviction,
            entry_conditions={"entry_composite": f.score, "entry_price": close},
            invalidation_conditions=dict(_DEFAULT_INVALIDATION),
            evidence_refs={"confluence_finding_id": f.id, "kind": f.kind},
            status=ThesisStatus.ACTIVE.value,
            conviction_history=[{"conviction": conviction, "trigger": f"confluence:{f.kind}"}],
        )
        save_thesis(session, thesis)
        promoted.append(f.ticker)
        logger.info(
            "promoted confluence {t} (score {s:.1f} -> conviction {c:.2f}, ${p:.2f})",
            t=f.ticker, s=f.score, c=conviction, p=close,
        )
    return promoted


def run_insight_promotion(
    settings=None, *, config=None, min_score: float = 4.0, max_promotions: int = 5,
    today: date | None = None,
) -> dict:
    """Detect confluence findings and promote the strongest fresh, liquid ones (CLI/cron).

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
