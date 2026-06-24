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
from statistics import mean, median

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import select

from investment_monitor.collectors.insider import is_junk_ticker
from investment_monitor.config import Settings, get_settings
from investment_monitor.storage import (
    ConfluenceFinding,
    FINDING_INSIDER_CLUSTER,
    FINDING_MULTI_SOURCE,
    InsiderTransaction,
    finding_exists_for_date,
    get_prices,
    get_recent_news,
    get_session,
    init_db,
    save_finding,
)

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
    source_bonus: float = Field(default=0.75, ge=0)   # each extra STRONG source (insider/volume)
    news_bonus: float = Field(default=0.25, ge=0)     # news is weak: a smaller bonus, never a qualifier
    recency_halflife_days: float = Field(default=14.0, gt=0)
    recency_floor: float = Field(default=0.25, ge=0, le=1.0)
    exclude_entities: bool = True
    # Volume-spike second source (only evaluated for insider-active tickers, so it
    # CORROBORATES insider clusters rather than firing alone).
    volume_spike_multiple: float = Field(default=2.0, gt=1.0)
    volume_lookback: int = Field(default=20, ge=5)
    volume_min_avg: float = Field(default=50_000.0, ge=0)
    # News third source (weak, non-directional corroboration). Only for insider-active
    # tickers; requires >= this many recent headlines to count as a source.
    news_min_items: int = Field(default=2, ge=1)


def score_confluence(
    evidence: list[Evidence],
    *,
    today: date,
    breadth_cap: int = 8,
    value_weight: float = 0.5,
    source_bonus: float = 0.75,
    news_bonus: float = 0.25,
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

    # Only DOLLAR-bearing actors (e.value is not None) define breadth and conviction.
    # The volume/news pseudo-actors emit value=None: they corroborate via n_sources /
    # source_mult (and the dispersion override), but if folded into by_actor they would
    # both inflate the breadth headcount AND drag the per-actor median toward 0 — i.e.
    # cross-source corroboration would paradoxically LOWER the conviction multiplier.
    # So we exclude None-valued actors here while still counting their SOURCE below.
    by_actor: dict[tuple[str, str], float] = defaultdict(float)
    for e in evidence:
        if e.value is None:
            continue  # pseudo-actor / unknown-$: corroborates via source, not breadth
        by_actor[(e.source, e.actor)] += max(0.0, e.value)
    n_actors = len(by_actor)
    sources = {e.source for e in evidence}
    # News is weak/non-directional context: it never counts as a "strong" corroborating
    # source, so it can't flip the one-day-event penalty or satisfy the cluster gate.
    strong_sources = {s for s in sources if s != "news"}
    n_sources = len(sources)
    n_strong = len(strong_sources)
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
    if n_strong > 1:
        dispersion = 1.0  # multiple STRONG sources agreeing is itself non-routine
    else:
        ratio = (distinct_days / n_actors) if n_actors else 0.0
        dispersion = min(1.0, max(0.4, ratio * 2.0))  # all filed one day -> 0.4
    # Strong sources beyond the first add the full bonus; news adds only a small one.
    source_mult = (
        1.0 + source_bonus * max(0, n_strong - 1) + (news_bonus if "news" in sources else 0.0)
    )
    score = breadth * conviction * dispersion * recency * source_mult

    return {
        "n_sources": n_sources, "n_actors": n_actors, "n_strong": n_strong,
        "total_value": total_value, "median_per_actor": median_per_actor,
        "distinct_days": distinct_days, "score": score, "newest": newest,
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
        if is_junk_ticker(ticker):
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


def gather_volume_evidence(
    session, tickers: set[str], today: date, *,
    spike_multiple: float = 2.0, lookback: int = 20, min_avg_volume: float = 50_000.0,
) -> list[Evidence]:
    """Volume-spike Evidence — a SECOND source — for the given (insider-active) tickers.

    Evaluated only for tickers that already have insider evidence, so a volume spike
    CORROBORATES a cluster (cross-source confluence) rather than firing on its own.
    A spike = latest session volume >= ``spike_multiple`` x the trailing average.
    """
    out: list[Evidence] = []
    for ticker in tickers:
        try:
            prices = get_prices(session, ticker, days=lookback + 8)  # newest-first
            if len(prices) < max(5, lookback // 2):
                continue
            latest = prices[0]
            prior = [p for p in prices[1:lookback + 1] if p.volume]
            if not prior or latest.volume is None:
                continue
            avg_vol = mean(p.volume for p in prior)
            if avg_vol < min_avg_volume:
                continue
            if latest.volume >= spike_multiple * avg_vol:
                out.append(Evidence(
                    ticker=ticker, source="volume", actor="volume_spike",
                    date=latest.date, value=None,
                    detail=f"volume {latest.volume / avg_vol:.1f}x {lookback}d avg",
                ))
        except Exception as exc:  # noqa: BLE001 - a missing/odd price series must not abort
            logger.debug(f"volume evidence failed for {ticker}: {exc}")
    return out


def gather_news_evidence(
    session, tickers: set[str], today: date, *, window_days: int = 30, min_items: int = 2,
) -> list[Evidence]:
    """News-flow Evidence — a weak THIRD source — for insider-active tickers.

    Non-directional: a burst of recent headlines on a name that insiders are buying is
    corroborating attention, not a buy/sell signal. Requires >= ``min_items`` recent
    headlines to count, so a single stray article doesn't manufacture a source.
    """
    # Score news by PUBLICATION time, so a backfill/re-ingest of weeks-old articles
    # (created_at = now, published_at = stale) can't masquerade as fresh corroboration
    # and inflate the multi-source score. get_recent_news filters by created_at
    # (ingestion time), so we re-filter here to keep ONLY genuinely-recent articles —
    # the same date dimension we score by.
    out: list[Evidence] = []
    published_cutoff = today - timedelta(days=window_days)
    for ticker in tickers:
        try:
            # Pull a generous candidate set by ingestion time, then keep only items
            # actually PUBLISHED within the window. We over-fetch (drop the hours
            # bound's recency role) because created_at recency is not what gates here.
            items = get_recent_news(session, ticker=ticker, hours=window_days * 24)
            pub_dates = [
                i.published_at.date()
                for i in items
                if i.published_at and i.published_at.date() >= published_cutoff
            ]
            if len(pub_dates) < min_items:
                continue
            latest = max(pub_dates)
            out.append(Evidence(
                ticker=ticker, source="news", actor="news_flow", date=latest,
                value=None, detail=f"{len(pub_dates)} recent headlines",
            ))
        except Exception as exc:  # noqa: BLE001 - news is best-effort context
            logger.debug(f"news evidence failed for {ticker}: {exc}")
    return out


def _price_change_since(session, ticker: str, since: date, today: date) -> float | None:
    """Percent return from the close on/before ``since`` to the latest close (or None)."""
    try:
        prices = get_prices(session, ticker, days=(today - since).days + 8)
        closes = [(p.date, p.close) for p in prices if p.close]
        if len(closes) < 2:
            return None
        latest_close = closes[0][1]  # newest-first
        at_since = next((c for d, c in closes if d <= since), closes[-1][1])
        if not at_since or at_since <= 0:
            return None
        return (latest_close / at_since - 1.0) * 100.0
    except Exception:  # noqa: BLE001 - price context is best-effort
        return None


def _finding_kind(stats: dict) -> str:
    return FINDING_MULTI_SOURCE if stats["n_sources"] > 1 else FINDING_INSIDER_CLUSTER


def _build_narrative(
    ticker: str, evidence: list[Evidence], stats: dict, window_days: int,
    price_change: float | None = None,
) -> str:
    """A factual, honest insight — the 'look here': per-actor size, day spread,
    a volume-corroboration tag, and price context."""
    ins = [e for e in evidence if e.source == "insider"]
    insiders = sorted({e.actor for e in ins})
    names = ", ".join(insiders[:4]) + (f" (+{len(insiders) - 4} more)" if len(insiders) > 4 else "")
    idates = sorted(e.date for e in ins) or sorted(e.date for e in evidence)
    span = f"{idates[0]:%b %d}" + (f"–{idates[-1]:%b %d}" if idates[-1] != idates[0] else "")
    total = stats["total_value"] or 0.0
    med = stats.get("median_per_actor") or 0.0
    vol = " + unusual volume" if any(e.source == "volume" for e in evidence) else ""
    pc = f" {price_change:+.0f}% since buys." if price_change is not None else ""
    return (
        f"{ticker}: {len(insiders)} insiders bought on the open market{vol} over "
        f"{window_days}d ({span}) — ${total:,.0f} total, ~${med:,.0f}/insider, "
        f"{len(set(idates))} day(s).{pc} Buyers: {names}."
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

    insider_ev = gather_insider_evidence(
        session, config.window_days, today, exclude_entities=config.exclude_entities
    )
    # Second source: volume spikes, evaluated ONLY for insider-active tickers so they
    # corroborate clusters (true cross-source confluence). Congress/news append here next.
    active_tickers = {e.ticker for e in insider_ev}
    volume_ev = gather_volume_evidence(
        session, active_tickers, today, spike_multiple=config.volume_spike_multiple,
        lookback=config.volume_lookback, min_avg_volume=config.volume_min_avg,
    )
    news_ev = gather_news_evidence(
        session, active_tickers, today,
        window_days=config.window_days, min_items=config.news_min_items,
    )
    evidence = insider_ev + volume_ev + news_ev

    by_ticker: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        by_ticker[e.ticker].append(e)

    findings: list[ConfluenceFinding] = []
    for ticker, evs in by_ticker.items():
        stats = score_confluence(
            evs, today=today, breadth_cap=config.breadth_cap,
            value_weight=config.value_weight, source_bonus=config.source_bonus,
            news_bonus=config.news_bonus,
            recency_halflife_days=config.recency_halflife_days,
            recency_floor=config.recency_floor,
        )
        # A real finding needs >= min_actors INSIDERS, or >= 2 STRONG sources
        # (insider + volume). News alone can corroborate but never qualify a name.
        insider_actors = len({e.actor for e in evs if e.source == "insider"})
        breadth_ok = insider_actors >= config.min_actors or stats["n_strong"] >= 2
        if not breadth_ok or stats["total_value"] < config.min_total_value:
            continue
        if stats["score"] < config.min_score:
            continue
        kind = _finding_kind(stats)
        if finding_exists_for_date(session, ticker, kind, today):
            continue
        # Price context: return since the median insider-buy date (best-effort).
        insider_dates = sorted(e.date for e in evs if e.source == "insider")
        since = insider_dates[len(insider_dates) // 2] if insider_dates else stats["newest"]
        price_change = _price_change_since(session, ticker, since, today)
        finding = ConfluenceFinding(
            ticker=ticker, kind=kind, score=round(stats["score"], 3),
            window_days=config.window_days, n_sources=stats["n_sources"],
            n_actors=stats["n_actors"], total_value=stats["total_value"],
            price_change_pct=round(price_change, 2) if price_change is not None else None,
            evidence=_evidence_json(evs, config.max_evidence_stored),
            narrative=_build_narrative(ticker, evs, stats, config.window_days, price_change),
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
