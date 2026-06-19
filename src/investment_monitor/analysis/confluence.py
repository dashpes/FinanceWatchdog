"""The confluence / insight engine.

Turns the broad multi-source event stream into FIRST-CLASS insights — a stated
"look here", not a number folded into a sizing tilt. Its core is a confluence score
that rewards CONCENTRATED CONVICTION, not raw headcount: a ticker scores on how many
*distinct* actors (and, as more sources arrive, distinct SOURCES) line up, damped so
a mega-cap board-wide filing can't dominate, multiplied by per-actor dollar conviction
and a filing-day dispersion factor that demotes one-day administrative events. This is
the deliberate opposite of the old weighted-average that diluted corroboration.

Design is source-agnostic: every source emits ``Evidence`` for a ticker; scoring counts
distinct sources + actors. v1's one live broad source is SEC Form 4 insider *purchases*
(raw_code == 'P', individuals only) — so it surfaces genuine insider-buying clusters.
Congress / volume / news append Evidence later with no rework.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

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

# Owner-name tokens that mark a fund/entity 10%-holder rather than an individual
# insider. A board cluster of individuals is the v1 signal; entity/activist buys are a
# distinct (future) signal and otherwise dominate the actor count + dollars.
_ENTITY_RE = re.compile(
    r"\b(LLC|L\.L\.C|L\.P|LP|INC|CORP|CAPITAL|PARTNERS?|MANAGEMENT|MGMT|FUND|TRUST|"
    r"HOLDINGS?|VENTURES?|ADVISORS?|GROUP|ASSOCIATES|ASSET|PLC|LTD|GP)\b",
    re.IGNORECASE,
)


def _normalize_actor(name: str) -> str:
    return " ".join((name or "").split()).strip()


def _is_entity(name: str) -> bool:
    return bool(_ENTITY_RE.search(name or ""))


@dataclass(frozen=True)
class Evidence:
    """One atomic signal: an actor, from a source, acting on a ticker on a date."""

    ticker: str
    source: str          # 'insider' (v1); later 'congress', 'volume', 'news'
    actor: str           # distinct actor within the source (normalized name)
    date: date
    value: float | None  # dollar magnitude if known
    detail: str


class ConfluenceConfig(BaseModel):
    """Tuning for the confluence engine."""

    window_days: int = Field(default=30, ge=1)
    # A single-source finding needs >= this many distinct actors (a real cluster);
    # any >=2-source agreement qualifies regardless of actor count.
    min_actors: int = Field(default=3, ge=1)
    # Drop trivially small clusters (e.g. director qualifying shares).
    min_total_value: float = Field(default=25_000.0, ge=0)
    min_score: float = Field(default=2.0, ge=0)
    max_evidence_stored: int = Field(default=25, ge=1)
    # Saturate raw actor count so breadth can't dominate (a board-wide filing of 31
    # token buyers shouldn't outrank 7 concentrated insiders).
    breadth_cap: int = Field(default=8, ge=1)
    value_weight: float = Field(default=0.5, ge=0)    # per-actor dollar conviction
    source_bonus: float = Field(default=0.75, ge=0)   # each extra distinct source
    recency_halflife_days: float = Field(default=14.0, gt=0)
    recency_floor: float = Field(default=0.25, ge=0, le=1.0)
    exclude_entities: bool = True


def score_confluence(
    evidence: list[Evidence],
    *,
    today: date,
    breadth_cap: int = 8,
    value_weight: float = 0.5,
    source_bonus: float = 0.75,
    recency_halflife_days: float = 14.0,
    recency_floor: float = 0.25,
) -> dict:
    """Pure: reduce a ticker's evidence to a concentration-aware confluence score.

    ``score = breadth * conviction * dispersion * recency * source_mult`` where
    - breadth  = min(n_actors, breadth_cap)        — damped, so headcount can't dominate.
    - conviction = 1 + value_weight*log10(1 + median_$_per_actor/5k) — per-actor size.
    - dispersion (single-source) penalizes one-day mass events; 1.0 across sources.
    - recency  = 0.5 ** (age_of_MEDIAN_date / halflife), floored — robust to a straggler.
    - source_mult = 1 + source_bonus*(n_sources-1) — super-additive across sources.
    """
    if not evidence:
        return {"n_sources": 0, "n_actors": 0, "total_value": 0.0,
                "median_per_actor": 0.0, "distinct_days": 0, "score": 0.0, "newest": None}

    by_actor: dict[tuple[str, str], float] = defaultdict(float)
    for e in evidence:
        by_actor[(e.source, e.actor)] += max(0.0, e.value or 0.0)
    n_actors = len(by_actor)
    sources = {e.source for e in evidence}
    n_sources = len(sources)
    total_value = sum(max(0.0, e.value or 0.0) for e in evidence)
    median_per_actor = median(sorted(by_actor.values())) if by_actor else 0.0

    dates = [e.date for e in evidence]
    distinct_days = len(set(dates))
    newest = max(dates)
    med_date = sorted(dates)[len(dates) // 2]
    age = max(0, (today - med_date).days)
    recency = max(recency_floor, 0.5 ** (age / recency_halflife_days))

    breadth = min(n_actors, breadth_cap)
    conviction = 1.0 + value_weight * math.log10(1.0 + max(0.0, median_per_actor) / 5000.0)
    if n_sources > 1:
        dispersion = 1.0  # cross-source agreement is itself non-routine
    else:
        ratio = (distinct_days / n_actors) if n_actors else 0.0
        dispersion = min(1.0, max(0.4, ratio * 2.0))  # all filed one day -> 0.4
    source_mult = 1.0 + source_bonus * (n_sources - 1)
    score = breadth * conviction * dispersion * recency * source_mult

    return {
        "n_sources": n_sources, "n_actors": n_actors, "total_value": total_value,
        "median_per_actor": median_per_actor, "distinct_days": distinct_days,
        "score": score, "newest": newest,
    }


def gather_insider_evidence(
    session, window_days: int, today: date, *, exclude_entities: bool = True
) -> list[Evidence]:
    """Evidence from GENUINE open-market insider purchases (raw_code == 'P') in window.

    De-duplicates the SAME physical transaction (EDGAR indexes a Form 4 under both the
    issuer and the reporting-owner CIK, so broad ingestion stores it twice) and drops
    fund/entity 10%-holders so the actor count reflects individual insider conviction.
    """
    cutoff = today - timedelta(days=window_days)
    rows = session.scalars(
        select(InsiderTransaction).where(
            InsiderTransaction.raw_code == "P",
            InsiderTransaction.trade_date >= cutoff,
        )
    ).all()
    seen: set[tuple] = set()
    out: list[Evidence] = []
    for r in rows:
        ticker = (r.ticker or "").strip().upper()
        if ticker in _JUNK_TICKERS:
            continue
        actor = _normalize_actor(r.owner_name)
        if exclude_entities and _is_entity(actor):
            continue
        # Collapse the same physical transaction reported under two CIKs.
        key = (ticker, actor.upper(), r.trade_date, r.shares,
               round(float(r.price_per_share or 0.0), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(Evidence(
            ticker=ticker, source="insider", actor=actor, date=r.trade_date,
            value=float(r.total_value) if r.total_value else None,
            detail=f"{(r.owner_title or 'insider')} bought {r.shares:,} sh"
                   + (f" @ ${r.price_per_share:,.2f}" if r.price_per_share else ""),
        ))
    return out


def _finding_kind(stats: dict) -> str:
    return FINDING_MULTI_SOURCE if stats["n_sources"] > 1 else FINDING_INSIDER_CLUSTER


def _build_narrative(ticker: str, evidence: list[Evidence], stats: dict, window_days: int) -> str:
    """A factual, honest insight — the 'look here', with per-actor size + day spread."""
    actors = sorted({e.actor for e in evidence})
    names = ", ".join(actors[:4]) + (f" (+{len(actors) - 4} more)" if len(actors) > 4 else "")
    dates = sorted(e.date for e in evidence)
    span = f"{dates[0]:%b %d}" + (f"–{dates[-1]:%b %d}" if dates[-1] != dates[0] else "")
    if stats["n_sources"] > 1:
        srcs = ", ".join(sorted({e.source for e in evidence}))
        return (
            f"{ticker}: {stats['n_sources']} independent sources lined up ({srcs}) — "
            f"{stats['n_actors']} distinct actors over {window_days}d ({span})."
        )
    total = stats["total_value"] or 0.0
    med = stats.get("median_per_actor") or 0.0
    return (
        f"{ticker}: {len(actors)} insiders bought on the open market over {window_days}d "
        f"({span}) — ${total:,.0f} total, ~${med:,.0f}/insider median, across "
        f"{stats['distinct_days']} day(s). Buyers: {names}."
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

    evidence = gather_insider_evidence(
        session, config.window_days, today, exclude_entities=config.exclude_entities
    )
    # Future broad sources append here — congress, volume spikes, news catalysts.

    by_ticker: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_ticker[e.ticker].append(e)

    findings: list[ConfluenceFinding] = []
    for ticker, evs in by_ticker.items():
        stats = score_confluence(
            evs, today=today, breadth_cap=config.breadth_cap,
            value_weight=config.value_weight, source_bonus=config.source_bonus,
            recency_halflife_days=config.recency_halflife_days,
            recency_floor=config.recency_floor,
        )
        breadth_ok = stats["n_actors"] >= config.min_actors or stats["n_sources"] >= 2
        if not breadth_ok or stats["total_value"] < config.min_total_value:
            continue
        if stats["score"] < config.min_score:
            continue
        kind = _finding_kind(stats)
        if finding_exists_for_date(session, ticker, kind, today):
            continue
        finding = ConfluenceFinding(
            ticker=ticker, kind=kind, score=round(stats["score"], 3),
            window_days=config.window_days, n_sources=stats["n_sources"],
            n_actors=stats["n_actors"], total_value=stats["total_value"],
            evidence=_evidence_json(evs, config.max_evidence_stored),
            narrative=_build_narrative(ticker, evs, stats, config.window_days),
            as_of_date=today,
        )
        save_finding(session, finding)
        findings.append(finding)

    findings.sort(key=lambda f: f.score, reverse=True)
    logger.info(f"confluence: {len(findings)} findings on {today}")
    return findings


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
