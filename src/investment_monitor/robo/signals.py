"""Event-driven signals for the robo advisor (Phase 2).

This module turns FinanceWatchdog's already-collected event data — insider Form 4
buys/sells, congressional cluster trades, volume spikes, news relevance, and
earnings proximity — into a compact per-symbol summary the proposer can reason
about. The advisor reacts to events instead of only rebalancing to fixed weights.

Design (mirrors ``gate.py``'s I/O-vs-pure split):

* ``collect_raw_signals`` / ``fetch_signals`` are the only impure functions: they
  read the database via the existing storage query helpers. No scoring.
* ``score_symbol`` / ``build_snapshot`` / ``tilt_targets`` are PURE: given
  hand-built ``SignalEvent`` lists they are fully unit-testable with no DB.

CRITICAL SAFETY INVARIANTS (verified by tests):

* Signals are *advisory only*. They never reach the guardrail gate and cannot
  relax any cap. ``tilt_targets`` moves a target weight by at most
  ``max_event_tilt`` and the gate still re-checks every resulting order.
* When ``config.signals.enabled`` is False (the default), ``fetch_signals``
  returns an empty snapshot and the proposer behaves exactly as the baseline
  drift-to-target rebalancer.
* Directional tilts come only from insider and congressional activity. News and
  volume are non-directional "attention" context (the LLM may read them for
  direction). Earnings proximity is pure CAUTION: it suppresses buying into a
  name but never drives a sale on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import TYPE_CHECKING, Sequence

from loguru import logger

from investment_monitor.robo.config import RoboConfig, SignalConfig
from investment_monitor.robo.models import CASH_SYMBOL
from investment_monitor.storage import (
    get_insider_transactions,
    get_prices,
    get_recent_news,
    get_trades_for_ticker,
    get_upcoming_earnings,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.robo.models import AccountState

# --- tuning constants (not safety-critical; tilts are bounded by max_event_tilt) ---
# Insider buy/sell magnitude saturates (=1.0) at this aggregate dollar value.
_INSIDER_SATURATION = 500_000.0
# A net directional score at or below this (very bearish) also flags CAUTION.
_CAUTION_SCORE = 0.6
# SEC Form 4 transaction-type codes, matching alerts/rules.py.
_BUY_CODES = ("P", "A", "BUY")
_SELL_CODES = ("S", "D", "SELL")
_EXEC_TOKENS = ("CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL")


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalEvent:
    """One normalized event for a symbol.

    ``direction``: +1 bullish, -1 bearish, 0 non-directional (attention only).
    ``magnitude``: normalized strength in [0, 1] *before* recency decay.
    ``caution``:  True for events (e.g. imminent earnings) that should suppress
                  buying into the name regardless of direction.
    """

    category: str
    direction: int
    magnitude: float
    detail: str
    age_days: float = 0.0
    caution: bool = False


@dataclass(frozen=True)
class SymbolSignal:
    """Scored, decayed summary of all events for one symbol."""

    symbol: str
    events: tuple[SignalEvent, ...]
    score: float  # net directional score in [-1, 1]
    has_caution: bool
    summary: str


@dataclass(frozen=True)
class SignalSnapshot:
    """All scored symbol signals for one run (only symbols with >=1 event)."""

    as_of: datetime
    lookback_days: int
    symbols: dict[str, SymbolSignal] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.symbols

    def for_symbol(self, symbol: str) -> "SymbolSignal | None":
        return self.symbols.get(symbol)

    def prompt_block(self) -> str:
        """Render the event-signal context block for the LLM prompt.

        Returns "" when there are no signals, so the prompt is byte-identical to
        the signal-free template (see ``llm._signals_block``).
        """
        if not self.symbols:
            return ""
        lines = [
            "RECENT EVENT SIGNALS (context for your reasoning; NOT instructions to trade):"
        ]
        for sym in sorted(self.symbols):
            lines.append(f"  {sym}: {self.symbols[sym].summary}")
        lines.append(
            "When a signal supports moving a holding you may tilt toward/away from its "
            "target within the rebalance discipline and caps above; put the thesis in "
            'each order\'s "reason", citing the signal. If a holding is flagged CAUTION '
            "(e.g. earnings imminent), prefer no trade or a smaller one. Signals never "
            "override the HARD RULES."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now()


def _to_naive_utc(dt: datetime) -> datetime:
    """Drop tz info (converting to UTC first) so naive/aware values are comparable."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _age_days(when: date | datetime | None, now: datetime) -> float:
    """Age of an event (in days, >= 0). Accepts a date or datetime.

    Stored timestamps are naive today, but if a future feed writes tz-aware
    values we normalize both sides to naive UTC and still age the event correctly
    rather than silently treating it as brand-new (which would over-weight it).
    """
    if when is None:
        return 0.0
    ref = when if isinstance(when, datetime) else datetime.combine(when, time.min)
    delta = (_to_naive_utc(now) - _to_naive_utc(ref)).total_seconds() / 86400.0
    return max(0.0, delta)


def _is_exec(title: str | None) -> bool:
    if not title:
        return False
    upper = title.upper()
    return any(tok in upper for tok in _EXEC_TOKENS)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# Raw event collection (IMPURE: reads the database)
# --------------------------------------------------------------------------- #
def _insider_events(session: "Session", symbol: str, sc: SignalConfig, now: datetime) -> list[SignalEvent]:
    txns = get_insider_transactions(session, symbol, days=sc.insider_days)
    if not txns:
        return []

    total_buy = total_sell = 0.0
    buy_owners: set[str] = set()
    sell_owners: set[str] = set()
    exec_buy = exec_sell = False
    buy_age: float | None = None
    sell_age: float | None = None

    for t in txns:
        code = (t.transaction_type or "").upper()
        value = float(t.total_value or 0.0)
        is_exec = _is_exec(t.owner_title)
        age = _age_days(t.trade_date, now)
        if code in _BUY_CODES:
            total_buy += value
            if t.owner_name:
                buy_owners.add(t.owner_name)
            exec_buy = exec_buy or is_exec
            buy_age = age if buy_age is None else min(buy_age, age)
        elif code in _SELL_CODES:
            total_sell += value
            if t.owner_name:
                sell_owners.add(t.owner_name)
            exec_sell = exec_sell or is_exec
            sell_age = age if sell_age is None else min(sell_age, age)

    events: list[SignalEvent] = []
    cluster = sc.cluster_min_unique

    buy_notable = total_buy >= sc.insider_buy_min_value or exec_buy or len(buy_owners) >= cluster
    if buy_notable and buy_owners:
        mag = _clamp(total_buy / _INSIDER_SATURATION, 0.0, 1.0)
        if exec_buy:
            mag = min(mag + 0.2, 1.0)
        if len(buy_owners) >= cluster:
            mag = max(mag, 0.8)
        events.append(SignalEvent(
            category="insider", direction=1, magnitude=mag,
            detail=(
                f"insider buys ${total_buy:,.0f} by {len(buy_owners)} owner(s)"
                f"{', incl. exec' if exec_buy else ''}"
                f"{', cluster' if len(buy_owners) >= cluster else ''}"
                f" ({buy_age:.0f}d ago)"
            ),
            age_days=buy_age or 0.0,
        ))

    sell_notable = total_sell >= sc.insider_sell_min_value or exec_sell or len(sell_owners) >= cluster
    if sell_notable and sell_owners:
        mag = _clamp(total_sell / _INSIDER_SATURATION, 0.0, 1.0)
        if exec_sell:
            mag = min(mag + 0.2, 1.0)
        if len(sell_owners) >= cluster:
            mag = max(mag, 0.8)
        events.append(SignalEvent(
            category="insider", direction=-1, magnitude=mag,
            detail=(
                f"insider sells ${total_sell:,.0f} by {len(sell_owners)} owner(s)"
                f"{', incl. exec' if exec_sell else ''}"
                f"{', cluster' if len(sell_owners) >= cluster else ''}"
                f" ({sell_age:.0f}d ago)"
            ),
            age_days=sell_age or 0.0,
        ))
    return events


def _congress_events(session: "Session", symbol: str, sc: SignalConfig, now: datetime) -> list[SignalEvent]:
    trades = get_trades_for_ticker(session, symbol, days=sc.congress_days)
    if not trades:
        return []

    buy_pols: set[str] = set()
    sell_pols: set[str] = set()
    buy_age: float | None = None
    sell_age: float | None = None
    for tr in trades:
        kind = (tr.trade_type or "").lower()
        age = _age_days(tr.trade_date, now)
        if kind == "buy" and tr.politician:
            buy_pols.add(tr.politician)
            buy_age = age if buy_age is None else min(buy_age, age)
        elif kind == "sell" and tr.politician:
            sell_pols.add(tr.politician)
            sell_age = age if sell_age is None else min(sell_age, age)

    events: list[SignalEvent] = []
    cluster = sc.cluster_min_unique
    # Only cluster activity is trusted as directional (mirrors alert rules).
    if len(buy_pols) >= cluster:
        mag = _clamp(0.6 + 0.1 * (len(buy_pols) - cluster), 0.0, 1.0)
        events.append(SignalEvent(
            category="congress", direction=1, magnitude=mag,
            detail=f"{len(buy_pols)} members bought ({buy_age:.0f}d ago)",
            age_days=buy_age or 0.0,
        ))
    if len(sell_pols) >= cluster:
        mag = _clamp(0.6 + 0.1 * (len(sell_pols) - cluster), 0.0, 1.0)
        events.append(SignalEvent(
            category="congress", direction=-1, magnitude=mag,
            detail=f"{len(sell_pols)} members sold ({sell_age:.0f}d ago)",
            age_days=sell_age or 0.0,
        ))
    return events


def _volume_events(session: "Session", symbol: str, sc: SignalConfig, now: datetime) -> list[SignalEvent]:
    """Recompute the volume-spike check from alerts/rules.py (no stored flag).

    Non-directional: a spike means "something is happening", not a buy/sell.
    """
    prices = get_prices(session, symbol, days=sc.volume_lookback + 5)
    if len(prices) < 2:
        return []
    latest = prices[0]
    if not latest.volume:
        return []
    historical = [p for p in prices[1:] if p.volume and p.volume > 0]
    if len(historical) < sc.volume_lookback // 2:
        return []
    window = historical[: sc.volume_lookback]
    avg = sum(p.volume for p in window) / len(window)
    if avg <= 0:
        return []
    multiple = latest.volume / avg
    if multiple < sc.volume_spike_multiplier:
        return []
    # Saturate magnitude at ~2x the trigger multiplier so big spikes read stronger.
    mag = _clamp((multiple - 1.0) / (2.0 * sc.volume_spike_multiplier - 1.0), 0.0, 1.0)
    return [SignalEvent(
        category="volume", direction=0, magnitude=mag,
        detail=f"volume {multiple:.1f}x {sc.volume_lookback}d avg",
        age_days=_age_days(latest.date, now),
    )]


def _news_events(session: "Session", symbol: str, sc: SignalConfig, now: datetime) -> list[SignalEvent]:
    """News as a non-directional attention signal.

    Direction is intentionally 0: ``NewsItem.sentiment`` is not populated, so we
    surface high-relevance headlines for the LLM to read and judge direction. The
    deterministic tilt path gets no directional pull from news.
    """
    items = get_recent_news(session, ticker=symbol, hours=sc.news_hours)
    relevant = [
        n for n in items
        if n.relevance_score is not None and n.relevance_score >= sc.news_relevance_min
    ]
    if not relevant:
        return []
    top = max(n.relevance_score for n in relevant)
    # relevance_score is 1-10; map 5->0 .. 10->1.
    mag = _clamp((top - 5.0) / 5.0, 0.0, 1.0)
    newest = relevant[0]  # query returns newest-first
    headline = (newest.headline or "").strip()[:80]
    # Age on the same clock the window query uses (created_at / ingestion time),
    # so the displayed age and the news_hours window stay consistent.
    when = newest.created_at or newest.published_at
    return [SignalEvent(
        category="news", direction=0, magnitude=mag,
        detail=f'{len(relevant)} headline(s), top rel {top:.0f}/10: "{headline}"',
        age_days=_age_days(when, now),
    )]


def _earnings_events(session: "Session", symbol: str, sc: SignalConfig, now: datetime) -> list[SignalEvent]:
    """Earnings proximity → CAUTION (suppress buying), never a sell trigger."""
    upcoming = get_upcoming_earnings(session, [symbol], days_ahead=sc.earnings_days_ahead)
    events: list[SignalEvent] = []
    for e in upcoming:
        days_until = (e.earnings_date - now.date()).days
        if days_until < 0:
            continue
        confirmed = " (confirmed)" if e.confirmed else " (estimated)"
        if days_until <= sc.earnings_caution_days:
            # Closer earnings → stronger caution. Direction 0: never drives a sale.
            mag = _clamp(1.0 - days_until / (sc.earnings_caution_days + 1), 0.3, 1.0)
            events.append(SignalEvent(
                category="earnings", direction=0, magnitude=mag,
                detail=f"earnings in {days_until}d{confirmed} — caution",
                age_days=0.0, caution=True,
            ))
        else:
            events.append(SignalEvent(
                category="earnings", direction=0, magnitude=0.3,
                detail=f"earnings in {days_until}d{confirmed}",
                age_days=0.0,
            ))
        break  # only the soonest matters
    return events


def collect_raw_signals(
    session: "Session",
    symbols: Sequence[str],
    config: RoboConfig,
    *,
    now: datetime | None = None,
) -> dict[str, list[SignalEvent]]:
    """Read every signal source and emit raw ``SignalEvent`` lists per symbol.

    Impure (DB reads) but does no scoring. A failure in any one source is logged
    and skipped so a single bad table never breaks the whole run.
    """
    now = now or _now()
    sc = config.signals
    out: dict[str, list[SignalEvent]] = {}
    collectors = (_insider_events, _congress_events, _volume_events, _news_events, _earnings_events)
    for symbol in symbols:
        if symbol == CASH_SYMBOL:
            continue
        events: list[SignalEvent] = []
        for collector in collectors:
            try:
                events.extend(collector(session, symbol, sc, now))
            except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
                logger.warning(
                    "signal collector {c} failed for {s}: {e}",
                    c=collector.__name__, s=symbol, e=exc,
                )
        if events:
            out[symbol] = events
    return out


# --------------------------------------------------------------------------- #
# Scoring (PURE)
# --------------------------------------------------------------------------- #
def _decayed(magnitude: float, age_days: float, half_life: float) -> float:
    if age_days <= 0:
        return magnitude
    return magnitude * (0.5 ** (age_days / half_life))


def _summarize(events: Sequence[SignalEvent], score: float, caution: bool) -> str:
    parts = []
    for e in events:
        sign = "+" if e.direction > 0 else ("-" if e.direction < 0 else "~")
        parts.append(f"{e.category.upper()}({sign}{e.magnitude:.2f}: {e.detail})")
    head = f"net={score:+.2f}"
    if caution:
        head += " CAUTION"
    return f"{head} | " + ", ".join(parts)


def score_symbol(symbol: str, events: Sequence[SignalEvent], sc: SignalConfig) -> SymbolSignal:
    """Pure: collapse a symbol's events into a net directional score + caution flag.

    Only directional events (``direction != 0``) contribute to the score; their
    decayed magnitudes are weighted by category and normalized to [-1, 1].
    Non-directional events (news/volume attention) appear in the summary but do
    not move the score. CAUTION is set by any caution event or a strongly bearish
    score.
    """
    num = 0.0
    den = 0.0
    caution = False
    for e in events:
        caution = caution or e.caution
        if e.direction == 0:
            continue
        weight = sc.weights.get(e.category, 0.0)
        if weight <= 0:
            continue
        decayed = _decayed(e.magnitude, e.age_days, sc.recency_half_life_days)
        num += e.direction * decayed * weight
        den += weight
    score = _clamp(num / den, -1.0, 1.0) if den > 0 else 0.0
    caution = caution or score <= -_CAUTION_SCORE
    return SymbolSignal(
        symbol=symbol,
        events=tuple(events),
        score=score,
        has_caution=caution,
        summary=_summarize(events, score, caution),
    )


def build_snapshot(
    raw: dict[str, list[SignalEvent]],
    config: RoboConfig,
    *,
    now: datetime | None = None,
) -> SignalSnapshot:
    """Pure: score every symbol's raw events into a snapshot."""
    now = now or _now()
    symbols = {
        sym: score_symbol(sym, events, config.signals)
        for sym, events in raw.items()
        if events
    }
    return SignalSnapshot(
        as_of=now,
        lookback_days=config.signals.congress_days,
        symbols=symbols,
    )


# --------------------------------------------------------------------------- #
# Target tilting (PURE) — the deterministic path's reaction to events
# --------------------------------------------------------------------------- #
def tilt_targets(
    target_allocation: dict[str, float],
    snapshot: SignalSnapshot,
    max_event_tilt: float,
) -> dict[str, float]:
    """Nudge target weights toward/away from names with directional signals.

    CASH is the explicit shock absorber: it funds buys and receives the proceeds
    of trims, so the freed/used weight never leaks into *other* holdings. A name
    with no signal keeps its exact target weight.

    * A signalled name's weight moves by ``score * max_event_tilt`` (score in
      [-1, 1]), never below 0.
    * A CAUTION flag clamps the move to <= 0 (never add into an imminent-earnings
      name), so caution can only reduce, never increase, exposure to that name.
    * When CASH cannot fund the net buys (would go negative, or there is no CASH
      bucket), the positive tilts are scaled down so deployment uses no more than
      the available cash. Weights stay >= 0 and sum to the original total (1.0).

    Pure: no DB, no clock. Returns a new dict (the input is not mutated).
    """
    if max_event_tilt <= 0 or snapshot.is_empty:
        return dict(target_allocation)

    # Per-name deltas, bounded by max_event_tilt and floored so a target never
    # goes negative. No-signal names get no delta.
    deltas: dict[str, float] = {}
    for sym, weight in target_allocation.items():
        if sym == CASH_SYMBOL:
            continue
        sig = snapshot.for_symbol(sym)
        if sig is None:
            continue
        delta = sig.score * max_event_tilt
        if sig.has_caution:
            delta = min(delta, 0.0)
        deltas[sym] = max(delta, -weight)

    cash = target_allocation.get(CASH_SYMBOL, 0.0)
    net_delta = sum(deltas.values())

    if net_delta > cash:
        # Not enough cash to fund the net buys: scale positive tilts so the net
        # deployment uses all available cash and no more (trims still apply fully).
        pos = sum(d for d in deltas.values() if d > 0)
        neg = sum(d for d in deltas.values() if d < 0)
        scale = (cash - neg) / pos if pos > 0 else 0.0
        for sym, d in deltas.items():
            if d > 0:
                deltas[sym] = d * scale
        net_delta = sum(deltas.values())

    tilted = dict(target_allocation)
    for sym, d in deltas.items():
        tilted[sym] = target_allocation[sym] + d
    # CASH absorbs the net change (max() guards float dust only).
    tilted[CASH_SYMBOL] = max(0.0, cash - net_delta)
    return tilted


# --------------------------------------------------------------------------- #
# Orchestration (IMPURE)
# --------------------------------------------------------------------------- #
def fetch_signals(
    session: "Session",
    config: RoboConfig,
    account_state: "AccountState",
    *,
    now: datetime | None = None,
) -> SignalSnapshot:
    """Collect + score signals for the allowlist and held positions.

    Returns an empty snapshot when signals are disabled, so callers can pass the
    result straight to the proposer unconditionally.
    """
    now = now or _now()
    if not config.signals.enabled:
        return SignalSnapshot(as_of=now, lookback_days=0, symbols={})
    symbols = sorted(set(config.allowlist) | {p.symbol for p in account_state.positions})
    raw = collect_raw_signals(session, symbols, config, now=now)
    return build_snapshot(raw, config, now=now)


__all__ = [
    "SignalEvent",
    "SymbolSignal",
    "SignalSnapshot",
    "collect_raw_signals",
    "score_symbol",
    "build_snapshot",
    "tilt_targets",
    "fetch_signals",
]
