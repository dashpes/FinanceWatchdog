"""Walk-forward backtest of the confluence -> promotion -> exit pipeline.

Replays whatever insider + price history is in the DB through the REAL production
scoring (``score_confluence`` via ``gather_insider_evidence``) and the same
promotion guards (score floor, run-up, liquidity) and exit policy (drawdown trip,
horizon close) as-of each historical date — so the tunables that are currently
vibes (promotion ``min_score``, the conviction band, the 25% drawdown trip) get an
empirical report card. Depth of history = whatever was ingested (EDGAR daily
indexes go back decades; ``insider.collect_all(days_back=N)`` backfills).

Fidelity notes:
- Sources replayed: insider clusters + volume spikes (both reconstructable as-of
  any date from stored rows). News/8-K corroboration is NOT replayed (no deep
  history), so backtested scores are slightly conservative vs live.
- No look-ahead: every query here is bounded by ``as_of`` on both sides, and
  entries fill at the first close ON/AFTER the signal date.
- This is per-trade analytics, not portfolio accounting: no sizing, cash, or
  overlap constraints — the question is "is the signal real and where should the
  floor sit", not "what would the account have made".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean, median

from loguru import logger
from sqlalchemy import select

from investment_monitor.analysis.confluence import (
    ConfluenceConfig,
    Evidence,
    gather_insider_evidence,
    score_confluence,
)
from investment_monitor.storage import Price

# Mirror the production promotion/exit defaults (confluence_promotion.py).
DEFAULT_PROMOTE_MIN_SCORE = 4.0
DEFAULT_MAX_RUN_PCT = 40.0
DEFAULT_MIN_PRICE = 3.0
DEFAULT_MIN_DOLLAR_VOLUME = 250_000.0
DEFAULT_DRAWDOWN_EXIT_PCT = 25.0
DEFAULT_HORIZON_DAYS = 90


@dataclass
class BacktestTrade:
    """One hypothetical round trip produced by the replay."""

    ticker: str
    score: float
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # horizon | drawdown | end_of_data

    @property
    def ret(self) -> float | None:
        if self.exit_price is None or not self.entry_price:
            return None
        return self.exit_price / self.entry_price - 1.0


@dataclass
class BacktestResult:
    start: date = None
    end: date = None
    steps: int = 0
    trades: list[BacktestTrade] = field(default_factory=list)

    def summary(self) -> dict:
        """Per-trade stats overall and per score band (the tuning view)."""
        closed = [t for t in self.trades if t.ret is not None]
        out = {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "steps": self.steps,
            "n_trades": len(self.trades),
            "n_closed": len(closed),
            "overall": _stats(closed),
            "by_score_band": {},
            "by_exit_reason": {},
        }
        bands = [(0.0, 6.0, "4-6"), (6.0, 8.0, "6-8"), (8.0, float("inf"), "8+")]
        for lo, hi, label in bands:
            out["by_score_band"][label] = _stats(
                [t for t in closed if lo <= t.score < hi]
            )
        for reason in sorted({t.exit_reason for t in closed if t.exit_reason}):
            out["by_exit_reason"][reason] = _stats(
                [t for t in closed if t.exit_reason == reason]
            )
        return out


def _stats(trades: list[BacktestTrade]) -> dict:
    rets = [t.ret for t in trades if t.ret is not None]
    if not rets:
        return {"n": 0, "hit_rate": None, "avg": None, "median": None,
                "best": None, "worst": None}
    return {
        "n": len(rets),
        "hit_rate": sum(1 for r in rets if r > 0) / len(rets),
        "avg": mean(rets),
        "median": median(rets),
        "best": max(rets),
        "worst": min(rets),
    }


def _prices_asof(session, ticker: str, as_of: date, *, days: int) -> list[Price]:
    """Daily rows in (as_of - days, as_of], newest first — bounded on BOTH sides."""
    cutoff = as_of - timedelta(days=days)
    stmt = (
        select(Price)
        .where(Price.ticker == ticker, Price.date > cutoff, Price.date <= as_of)
        .order_by(Price.date.desc())
    )
    return list(session.scalars(stmt))


def _close_on_or_after(session, ticker: str, d: date, *, max_days: int = 7):
    """First (date, close) ON/AFTER ``d`` — the realistic fill for a signal on ``d``."""
    stmt = (
        select(Price)
        .where(Price.ticker == ticker, Price.date >= d,
               Price.date <= d + timedelta(days=max_days), Price.close.is_not(None))
        .order_by(Price.date.asc())
        .limit(1)
    )
    p = session.scalar(stmt)
    return (p.date, float(p.close)) if p else (None, None)


def _volume_evidence_asof(
    session, tickers: set[str], as_of: date, config: ConfluenceConfig
) -> list[Evidence]:
    """As-of replica of ``gather_volume_evidence`` (same spike rule, bounded window)."""
    out: list[Evidence] = []
    for ticker in tickers:
        prices = _prices_asof(session, ticker, as_of, days=config.volume_lookback + 8)
        if len(prices) < max(5, config.volume_lookback // 2):
            continue
        latest = prices[0]
        prior = [p for p in prices[1:config.volume_lookback + 1] if p.volume]
        if not prior or latest.volume is None:
            continue
        avg_vol = mean(p.volume for p in prior)
        if avg_vol < config.volume_min_avg:
            continue
        if latest.volume >= config.volume_spike_multiple * avg_vol:
            out.append(Evidence(
                ticker=ticker, source="volume", actor="volume_spike",
                date=latest.date, value=None,
                detail=f"volume {latest.volume / avg_vol:.1f}x avg",
            ))
    return out


def _liquid_asof(
    session, ticker: str, as_of: date, *, min_price: float, min_dollar_volume: float,
    lookback: int = 20,
) -> tuple[bool, float | None]:
    """As-of replica of the promotion liquidity floor."""
    prices = _prices_asof(session, ticker, as_of, days=lookback + 8)
    if not prices or prices[0].close is None:
        return False, None
    latest_close = float(prices[0].close)
    if latest_close < min_price:
        return False, latest_close
    dollar_vols = [
        float(p.volume) * float(p.close)
        for p in prices[:lookback] if p.volume and p.close
    ]
    avg = (sum(dollar_vols) / len(dollar_vols)) if dollar_vols else 0.0
    return (avg >= min_dollar_volume), latest_close


def _price_change_since_asof(session, ticker: str, since: date, as_of: date) -> float | None:
    """As-of replica of the run-up guard's price-change context."""
    prices = _prices_asof(session, ticker, as_of, days=(as_of - since).days + 8)
    closes = [(p.date, p.close) for p in prices if p.close]
    if len(closes) < 2:
        return None
    latest_close = float(closes[0][1])
    at_since = next((c for d, c in closes if d <= since), closes[-1][1])
    if not at_since or at_since <= 0:
        return None
    return (latest_close / float(at_since) - 1.0) * 100.0


def run_confluence_backtest(
    session,
    *,
    start: date,
    end: date,
    step_days: int = 5,
    config: ConfluenceConfig | None = None,
    promote_min_score: float = DEFAULT_PROMOTE_MIN_SCORE,
    max_run_pct: float = DEFAULT_MAX_RUN_PCT,
    min_price: float = DEFAULT_MIN_PRICE,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    drawdown_exit_pct: float = DEFAULT_DRAWDOWN_EXIT_PCT,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> BacktestResult:
    """Replay confluence scoring + promotion + exits over [start, end]."""
    config = config or ConfluenceConfig()
    result = BacktestResult(start=start, end=end)
    open_trades: dict[str, BacktestTrade] = {}

    as_of = start
    while as_of <= end:
        if as_of.weekday() < 5:  # signals only re-evaluated on weekdays
            result.steps += 1

            # --- exits first (no same-day flip-flop with entries) ---------------
            for ticker, trade in list(open_trades.items()):
                prices = _prices_asof(session, ticker, as_of, days=10)
                latest = next((p for p in prices if p.close), None)
                if latest is None:
                    continue
                px = float(latest.close)
                drawdown = (trade.entry_price - px) / trade.entry_price * 100.0
                held = (as_of - trade.entry_date).days
                reason = None
                if drawdown >= drawdown_exit_pct:
                    reason = "drawdown"
                elif held >= horizon_days:
                    reason = "horizon"
                if reason:
                    trade.exit_date, trade.exit_price, trade.exit_reason = (
                        latest.date, px, reason
                    )
                    del open_trades[ticker]

            # --- entries: real scoring, as-of evidence ---------------------------
            insider_ev = gather_insider_evidence(
                session, config.window_days, as_of,
                exclude_entities=config.exclude_entities,
            )
            active = {e.ticker for e in insider_ev}
            volume_ev = _volume_evidence_asof(session, active, as_of, config)
            by_ticker: dict[str, list[Evidence]] = {}
            for e in insider_ev + volume_ev:
                by_ticker.setdefault(e.ticker, []).append(e)

            for ticker, evs in by_ticker.items():
                if ticker in open_trades:
                    continue
                stats = score_confluence(
                    evs, today=as_of, breadth_cap=config.breadth_cap,
                    value_weight=config.value_weight, source_bonus=config.source_bonus,
                    news_bonus=config.news_bonus,
                    recency_halflife_days=config.recency_halflife_days,
                    recency_floor=config.recency_floor,
                )
                insider_actors = len({e.actor for e in evs if e.source == "insider"})
                if insider_actors < config.min_actors and stats["n_strong"] < 2:
                    continue
                if stats["total_value"] < config.min_total_value:
                    continue
                if stats["score"] < max(config.min_score, promote_min_score):
                    continue
                insider_dates = sorted(e.date for e in evs if e.source == "insider")
                since = insider_dates[len(insider_dates) // 2] if insider_dates else as_of
                run = _price_change_since_asof(session, ticker, since, as_of)
                if run is not None and run > max_run_pct:
                    continue
                tradeable, _close = _liquid_asof(
                    session, ticker, as_of,
                    min_price=min_price, min_dollar_volume=min_dollar_volume,
                )
                if not tradeable:
                    continue
                fill_date, fill_px = _close_on_or_after(session, ticker, as_of)
                if fill_px is None:
                    continue
                trade = BacktestTrade(
                    ticker=ticker, score=round(stats["score"], 3),
                    entry_date=fill_date, entry_price=fill_px,
                )
                open_trades[ticker] = trade
                result.trades.append(trade)

        as_of += timedelta(days=max(1, step_days))

    # Anything still open closes at its last known price, labeled honestly.
    for ticker, trade in open_trades.items():
        prices = _prices_asof(session, ticker, end, days=10)
        latest = next((p for p in prices if p.close), None)
        if latest is not None:
            trade.exit_date, trade.exit_price = latest.date, float(latest.close)
            trade.exit_reason = "end_of_data"

    logger.info(
        f"backtest: {result.steps} steps, {len(result.trades)} trades "
        f"({start} -> {end})"
    )
    return result
