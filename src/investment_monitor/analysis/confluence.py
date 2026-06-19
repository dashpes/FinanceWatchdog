"""The confluence / insight engine.

Turns the broad multi-source event stream into FIRST-CLASS insights — a stated
"look here", not a number folded into a sizing tilt. Its core is a *super-additive*
confluence score: a ticker that >=2 distinct sources (or many distinct actors within
a source) line up on scores strictly MORE than any single strong signal. That is the
deliberate opposite of the old weighted-average that diluted corroboration.

Design is source-agnostic: every source emits ``Evidence`` for a ticker, and scoring
counts distinct sources + distinct actors. v1 has one live broad source — SEC Form 4
insider *purchases* (raw_code == 'P') — so it surfaces insider-buying clusters. When
congress / volume / news come online they append Evidence and multi-source confluence
emerges with no rework.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from investment_monitor.config import Settings, get_settings
from investment_monitor.storage import (
    ConfluenceFinding,
    FINDING_INSIDER_CLUSTER,
    FINDING_MULTI_SOURCE,
    InsiderTransaction,
    finding_exists_for_date,
    get_session,
    init_db,
    save_finding,
)

# Issuer symbols that are not real tickers (some filings carry these literally).
_JUNK_TICKERS = {"", "NONE", "N/A", "NA", "--", "N\\A"}


@dataclass(frozen=True)
class Evidence:
    """One atomic signal: an actor, from a source, acting on a ticker on a date."""

    ticker: str
    source: str          # 'insider' (v1); later 'congress', 'volume', 'news'
    actor: str           # distinct actor within the source (insider name, politician)
    date: date
    value: float | None  # dollar magnitude if known
    detail: str


class ConfluenceConfig(BaseModel):
    """Tuning for the confluence engine."""

    window_days: int = Field(default=30, ge=1)
    # A single-source finding needs >= this many distinct actors (a real cluster);
    # any >=2-source agreement qualifies regardless of actor count.
    min_actors: int = Field(default=3, ge=1)
    min_score: float = Field(default=3.0, ge=0)
    max_evidence_stored: int = Field(default=25, ge=1)
    # Score modifiers.
    value_weight: float = Field(default=0.25, ge=0)
    source_bonus: float = Field(default=0.75, ge=0)   # each extra distinct source
    recency_halflife_days: float = Field(default=14.0, gt=0)
    recency_floor: float = Field(default=0.25, ge=0, le=1.0)


def score_confluence(
    evidence: list[Evidence],
    *,
    today: date,
    value_weight: float = 0.25,
    source_bonus: float = 0.75,
    recency_halflife_days: float = 14.0,
    recency_floor: float = 0.25,
) -> dict:
    """Pure: reduce a ticker's evidence to a super-additive confluence score.

    ``score = n_actors * (1 + source_bonus*(n_sources-1)) * value_factor * recency``

    - n_actors = distinct (source, actor) pairs — breadth of independent agreement.
    - the source term makes 2 sources worth strictly more than 1 (super-additive).
    - value_factor = 1 + value_weight*log10(1 + $/10k) — magnitude matters, log-damped.
    - recency = 0.5 ** (age_of_newest / halflife), floored.
    """
    if not evidence:
        return {"n_sources": 0, "n_actors": 0, "total_value": 0.0, "score": 0.0, "newest": None}

    pairs = {(e.source, e.actor) for e in evidence}
    sources = {e.source for e in evidence}
    total_value = sum(e.value or 0.0 for e in evidence)
    newest = max(e.date for e in evidence)
    age = max(0, (today - newest).days)

    recency = max(recency_floor, 0.5 ** (age / recency_halflife_days))
    value_factor = 1.0 + value_weight * math.log10(1.0 + total_value / 10_000.0)
    source_mult = 1.0 + source_bonus * (len(sources) - 1)
    score = len(pairs) * source_mult * value_factor * recency

    return {
        "n_sources": len(sources),
        "n_actors": len(pairs),
        "total_value": total_value,
        "score": score,
        "newest": newest,
    }


def gather_insider_evidence(session, window_days: int, today: date) -> list[Evidence]:
    """Evidence from GENUINE open-market insider purchases (raw_code == 'P') in window."""
    cutoff = today - timedelta(days=window_days)
    rows = session.scalars(
        select(InsiderTransaction).where(
            InsiderTransaction.raw_code == "P",
            InsiderTransaction.trade_date >= cutoff,
        )
    ).all()
    out: list[Evidence] = []
    for r in rows:
        ticker = (r.ticker or "").strip().upper()
        if ticker in _JUNK_TICKERS:
            continue
        out.append(Evidence(
            ticker=ticker,
            source="insider",
            actor=(r.owner_name or "?").strip(),
            date=r.trade_date,
            value=float(r.total_value) if r.total_value else None,
            detail=f"{(r.owner_title or 'insider')} bought {r.shares:,} sh"
                   + (f" @ ${r.price_per_share:,.2f}" if r.price_per_share else ""),
        ))
    return out


def _finding_kind(stats: dict) -> str:
    return FINDING_MULTI_SOURCE if stats["n_sources"] > 1 else FINDING_INSIDER_CLUSTER


def _build_narrative(ticker: str, evidence: list[Evidence], stats: dict, window_days: int) -> str:
    """A factual, stated insight — the 'look here'."""
    actors = sorted({e.actor for e in evidence})
    names = ", ".join(actors[:4]) + (f" (+{len(actors) - 4} more)" if len(actors) > 4 else "")
    dates = sorted(e.date for e in evidence)
    span = f"{dates[0]:%b %d}" + (f"–{dates[-1]:%b %d}" if dates[-1] != dates[0] else "")
    total = stats["total_value"] or 0.0
    if stats["n_sources"] > 1:
        srcs = ", ".join(sorted({e.source for e in evidence}))
        return (
            f"{ticker}: {stats['n_sources']} sources lined up ({srcs}) — "
            f"{stats['n_actors']} distinct actors over {window_days}d ({span}). Look here."
        )
    return (
        f"{ticker}: {len(actors)} insiders made open-market purchases"
        f"{f' totaling ${total:,.0f}' if total else ''} over {window_days}d ({span}). "
        f"Buyers: {names}."
    )


def _evidence_json(evidence: list[Evidence], cap: int) -> list[dict]:
    """Bounded, persistable evidence list (strongest by value first)."""
    ranked = sorted(evidence, key=lambda e: (e.value or 0.0), reverse=True)[:cap]
    return [
        {"source": e.source, "actor": e.actor, "date": e.date.isoformat(),
         "value": e.value, "detail": e.detail}
        for e in ranked
    ]


def detect_confluence(
    session, config: ConfluenceConfig | None = None, *, today: date | None = None
) -> list[ConfluenceFinding]:
    """Run all source detectors, score per ticker, and persist qualifying findings."""
    config = config or ConfluenceConfig()
    today = today or date.today()

    evidence = gather_insider_evidence(session, config.window_days, today)
    # Future broad sources append here — congress, volume spikes, news catalysts.

    by_ticker: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_ticker[e.ticker].append(e)

    findings: list[ConfluenceFinding] = []
    for ticker, evs in by_ticker.items():
        stats = score_confluence(
            evs, today=today, value_weight=config.value_weight,
            source_bonus=config.source_bonus,
            recency_halflife_days=config.recency_halflife_days,
            recency_floor=config.recency_floor,
        )
        qualifies = stats["n_actors"] >= config.min_actors or stats["n_sources"] >= 2
        if not qualifies or stats["score"] < config.min_score:
            continue
        kind = _finding_kind(stats)
        if finding_exists_for_date(session, ticker, kind, today):
            continue
        findings.append(save_and_return(session, ConfluenceFinding(
            ticker=ticker, kind=kind, score=round(stats["score"], 3),
            window_days=config.window_days, n_sources=stats["n_sources"],
            n_actors=stats["n_actors"], total_value=stats["total_value"],
            evidence=_evidence_json(evs, config.max_evidence_stored),
            narrative=_build_narrative(ticker, evs, stats, config.window_days),
            as_of_date=today,
        )))
    findings.sort(key=lambda f: f.score, reverse=True)
    logger.info(f"confluence: {len(findings)} findings on {today}")
    return findings


def save_and_return(session, finding: ConfluenceFinding) -> ConfluenceFinding:
    save_finding(session, finding)
    return finding


def run_confluence(
    settings: Settings | None = None, *, config: ConfluenceConfig | None = None
) -> list[dict]:
    """CLI/cron entry: detect + persist findings, return display-safe summaries."""
    settings = settings or get_settings()
    init_db(settings.db_path)
    out: list[dict] = []
    with get_session() as session:
        for f in detect_confluence(session, config):
            out.append({
                "ticker": f.ticker, "kind": f.kind, "score": f.score,
                "n_sources": f.n_sources, "n_actors": f.n_actors,
                "narrative": f.narrative,
            })
    return out
