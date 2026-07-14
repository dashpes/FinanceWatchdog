"""Risk-adjusted conviction -> target-weight sizing (Phase 3, autonomous mode).

The LLM produces *conviction* (a thesis judgment); this module turns conviction +
Monte-Carlo risk metrics into a bounded portfolio target weight using DETERMINISTIC
math (never an LLM). The guardrail gate still re-checks every resulting order, so
this layer can only *propose* — it can never relax a cap.

Split like ``gate.py``: ``size_position`` / ``decay_conviction`` are PURE and fully
unit-testable; ``compute_conviction_weights`` is the thin impure orchestrator that
reads theses + simulations from the DB and threads the pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from loguru import logger

from investment_monitor.robo.config import RoboConfig, SizingConfig
from investment_monitor.robo.models import CASH_SYMBOL
from investment_monitor.storage import (
    accuracy_stats_for_symbol,
    get_active_theses,
    get_simulation_results,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Defensive key lookup: the simulation horizon JSON has used both `base_var_95`
# and `var_95` shapes across the codebase; accept either.
_VAR_KEYS = ("base_var_95", "var_95")
_CVAR_KEYS = ("base_cvar_95", "cvar_95")


@dataclass(frozen=True)
class RiskMetrics:
    """The risk inputs sizing needs, extracted from a SimulationResult."""

    drift: float        # annualized expected return
    volatility: float   # annualized
    var_95: float       # 90d VaR (return; <= 0 for a loss)
    cvar_95: float      # 90d CVaR / expected shortfall (return; <= 0 for a loss)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _age_days(when: datetime | None, now: datetime) -> float:
    if when is None:
        return 0.0
    delta = (_to_naive_utc(now) - _to_naive_utc(when)).total_seconds() / 86400.0
    return max(0.0, delta)


def _first_metric(horizon: dict, keys: Sequence[str]) -> float:
    for k in keys:
        v = horizon.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def risk_from_sim(sim) -> RiskMetrics | None:
    """Extract RiskMetrics from a SimulationResult ORM row (None -> None)."""
    if sim is None:
        return None
    horizon = sim.results_90d or {}
    return RiskMetrics(
        drift=float(sim.drift or 0.0),
        volatility=float(sim.volatility or 0.0),
        var_95=_first_metric(horizon, _VAR_KEYS),
        cvar_95=_first_metric(horizon, _CVAR_KEYS),
    )


def smoothed_conviction(convictions: Sequence[float], current: float, halflife_points: float) -> float:
    """Pure: recency-weighted (EWMA) mean of recent conviction points, to damp wobble.

    The newest point weighs most (weight halves every ``halflife_points`` going back), but
    no single point dominates — so a one-off spike that reverts (an overnight 0.7->0.4->0.7)
    barely moves the size, while a SUSTAINED shift (repeated 0.4s, or a decay toward 0) is
    followed. ``halflife_points <= 0`` or an empty series returns the raw ``current`` value
    (smoothing off). Callers must NOT smooth a broken/invalidated thesis — pass its zero
    straight through so exits stay prompt.
    """
    if halflife_points <= 0 or not convictions:
        return _clamp01(current)
    decay = 0.5 ** (1.0 / halflife_points)
    acc = 0.0
    weight_sum = 0.0
    w = 1.0
    for conv in reversed(list(convictions)):  # newest first, decaying into the past
        acc += w * _clamp01(conv)
        weight_sum += w
        w *= decay
    return _clamp01(acc / weight_sum) if weight_sum > 0 else _clamp01(current)


def decay_conviction(conviction: float, age_days: float, cfg: SizingConfig) -> float:
    """Decay raw conviction toward ``cfg.conviction_floor`` with a half-life.

    A fresh re-eval resets age to ~0 (no decay). Absent new evidence, conviction
    relaxes toward the neutral floor so stale max-conviction positions don't linger.
    """
    conviction = _clamp01(conviction)
    if age_days <= 0:
        return conviction
    floor = cfg.conviction_floor
    factor = 0.5 ** (age_days / cfg.conviction_half_life_days)
    return _clamp01(floor + (conviction - floor) * factor)


def size_position(conviction: float, risk: RiskMetrics | None, cfg: SizingConfig) -> float:
    """Pure: conviction + risk -> target weight in ``[0, max_position_weight]``.

    * No conviction -> no weight.
    * No simulation -> a conservative conviction-proportional floor size.
    * Otherwise the risk layer MODULATES conviction but never vetoes it:
      ``max(fractional-Kelly-on-Sharpe, conviction floor) x CVaR tail-haircut``.
      A strong trailing Sharpe sizes a name UP past the floor; a weak/negative one
      cannot zero a live thesis — confluence entries are contrarian by construction
      (insiders cluster-buy beaten-down names), so trailing drift argues against
      exactly the setups the system exists to trade. Tail risk still shrinks every
      size: deeper 90d CVaR (expected shortfall) -> smaller position.
    """
    conviction = _clamp01(conviction)
    if conviction <= 0:
        return 0.0
    floor = conviction * cfg.no_sim_weight_per_conviction
    if risk is None:
        return _clamp(floor, 0.0, cfg.max_position_weight)

    sharpe = (risk.drift - cfg.risk_free) / max(risk.volatility, cfg.min_vol)
    kelly = conviction * cfg.kelly_fraction * max(sharpe, 0.0)
    # More downside (larger |CVaR|) -> smaller size.
    tail_haircut = 1.0 / (1.0 + cfg.cvar_aversion * abs(risk.cvar_95))
    return _clamp(max(kelly, floor) * tail_haircut, 0.0, cfg.max_position_weight)


def accuracy_multiplier(
    stats: dict,
    *,
    accuracy_weight: float,
    floor: float,
    ceiling: float,
    min_samples: int,
) -> float:
    """Pure: a bounded sizing tilt from a symbol's realized-accuracy stats.

    ``multiplier = 1 + accuracy_weight * (hit_rate - 0.5) * 2`` clamped to
    ``[floor, ceiling]``. Neutral (1.0) until ``min_samples`` outcomes exist, so a
    thin track record never moves sizing. With ``ceiling == 1.0`` the tilt is
    shrink-only: it can dampen a poor-performing name but never inflate exposure.
    """
    n = int(stats.get("n", 0) or 0)
    if accuracy_weight <= 0 or n < min_samples:
        return 1.0
    hit_rate = float(stats.get("ewma_hit_rate", stats.get("hit_rate", 0.5)))
    return _clamp(1.0 + accuracy_weight * (hit_rate - 0.5) * 2.0, floor, ceiling)


def select_top_positions(
    raw: dict[str, float],
    max_positions: int,
    *,
    held: set[str] | frozenset[str] = frozenset(),
    hysteresis: float = 0.0,
) -> dict[str, float]:
    """Pure: choose which names keep capital when the book exceeds ``max_positions``.

    Plain top-N by weight — except a HELD incumbent keeps its slot unless a
    challenger's weight beats it by ``hysteresis`` (0.25 = 25% larger). In a
    saturated conviction band the raw ranking moves on LLM noise; without a margin,
    rank-8-vs-9 flips a real position every day (sell EML after 1 day, rebuy later).
    Exits are never delayed: a broken/exited/sub-floor name has no weight, so it is
    not in ``raw`` and holds no slot. Ties break held-first then by symbol — once
    the position cap flattens many strong names to the same weight, an EXACT tie
    must never rotate a real position, and dict order never decides capital.
    """
    if not max_positions or max_positions <= 0 or len(raw) <= max_positions:
        return dict(raw)
    ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0] not in held, kv[0]))
    if hysteresis <= 0 or not held:
        return dict(ranked[:max_positions])

    incumbents = [(s, w) for s, w in ranked if s in held][:max_positions]
    challengers = [(s, w) for s, w in ranked if s not in held]
    kept = dict(incumbents)
    # Fill the slots incumbents don't use with the strongest challengers outright.
    free = max_positions - len(incumbents)
    kept.update(challengers[:free])
    # Remaining challengers may only EVICT an incumbent by a clear margin,
    # strongest challenger against weakest incumbent first.
    remaining = challengers[free:]
    ci = 0
    for symbol, weight in sorted(incumbents, key=lambda kv: (kv[1], kv[0])):
        if ci >= len(remaining):
            break
        ch_symbol, ch_weight = remaining[ci]
        if ch_weight >= weight * (1.0 + hysteresis):
            del kept[symbol]
            kept[ch_symbol] = ch_weight
            ci += 1
    return kept


def is_averaging_up_without_support(
    *,
    avg_cost: float | None,
    ref_price: float | None,
    current_conviction: float,
    entry_conviction: float | None,
    cfg: SizingConfig,
) -> bool:
    """Pure: should a BUY that ADDS to a held position be BLOCKED as unjustified chasing?

    True only when the buy would raise cost basis — ``ref_price`` above ``avg_cost`` beyond
    ``average_up_tolerance`` — AND the thesis has NOT strengthened: conviction is neither
    already strong (``>= strong_add_conviction``) nor materially above where it was at entry
    (``>= entry_conviction + add_strengthen_margin``). Averaging DOWN (buying at/below cost)
    is never blocked. Missing cost/price data fails OPEN (returns False), and disabling
    ``block_average_up`` returns False. Opening a NEW position is the caller's concern.
    """
    if not cfg.block_average_up or avg_cost is None or ref_price is None or float(avg_cost) <= 0:
        return False
    if float(ref_price) <= float(avg_cost) * (1.0 + cfg.average_up_tolerance):
        return False  # at/below cost (averaging down) — always allowed
    cur = _clamp01(current_conviction)
    strengthened = cur >= cfg.strong_add_conviction or (
        entry_conviction is not None
        and cur >= _clamp01(entry_conviction) + cfg.add_strengthen_margin
    )
    return not strengthened


def _accuracy_mult(session: "Session", symbol: str, account_id: str | None, lcfg) -> float:
    """Impure: read a symbol's accuracy stats and derive its multiplier (fail-open)."""
    try:
        stats = accuracy_stats_for_symbol(
            session, symbol, account_id=account_id,
            ewma_halflife=lcfg.ewma_halflife, recent_window=lcfg.recent_window,
        )
        return accuracy_multiplier(
            stats,
            accuracy_weight=lcfg.accuracy_weight,
            floor=lcfg.modifier_floor,
            ceiling=lcfg.modifier_ceiling,
            min_samples=lcfg.min_samples,
        )
    except Exception as exc:  # noqa: BLE001 - a learning bug must never break sizing
        logger.warning("accuracy multiplier failed for {s}: {e}", s=symbol, e=exc)
        return 1.0


def _latest_sim(session: "Session", symbol: str):
    try:
        sims = get_simulation_results(session, symbol, limit=1)
    except Exception as exc:  # noqa: BLE001 - missing sim must not break sizing
        logger.warning("sim fetch failed for {s}: {e}", s=symbol, e=exc)
        return None
    return sims[0] if sims else None


def _sizing_conviction(thesis, cfg: SizingConfig) -> float:
    """Conviction to SIZE from: EWMA-smoothed over recent history to damp churn.

    A broken/invalidated thesis (conviction 0, or status INVALIDATED) is passed straight
    through UNsmoothed, so its exit is never delayed by an averaged-in past high. Bounds
    the history tail so ancient points can't anchor today's size.
    """
    current = _clamp01(thesis.conviction)
    if current <= 0 or str(getattr(thesis, "status", "") or "").upper() == "INVALIDATED":
        return current
    history = thesis.conviction_history or []
    convs = [
        float(h["conviction"])
        for h in history
        if isinstance(h, dict) and h.get("conviction") is not None
    ]
    return smoothed_conviction(convs[-24:], current, cfg.conviction_smoothing_halflife)


def compute_conviction_weights(
    session: "Session",
    config: RoboConfig,
    *,
    account_id: str | None = None,
    now: datetime | None = None,
    held_symbols: set[str] | None = None,
) -> dict[str, float]:
    """Build a target allocation from live theses (autonomous mode).

    Reads ACTIVE theses (WATCH = benched: kept + monitored but never sized), applies
    conviction time-decay, sizes each with the latest simulation's risk metrics, caps
    total equity at ``1 - min_cash_weight`` (scaling down proportionally if over), and
    lets CASH absorb the remainder. Returns ``{symbol: weight}`` summing to 1.0,
    always including CASH. ``held_symbols`` (when the caller knows the account state)
    enables selection hysteresis so noise can't rotate real positions.
    """
    now = now or datetime.now(timezone.utc)
    cfg = config.sizing
    lcfg = getattr(config, "learning", None)
    use_accuracy = lcfg is not None and lcfg.enabled and lcfg.accuracy_sizing

    raw: dict[str, float] = {}
    for thesis in get_active_theses(session, account_id):
        # Benched (WATCH) theses stay tracked but get NO capital — otherwise the
        # over-cap bench below would be undone right here on the next sizing pass.
        if str(getattr(thesis, "status", "") or "").lower() != "active":
            continue
        # Smooth the raw (noisy, ~45-min) re-eval conviction so sizing moves on SUSTAINED
        # thesis changes, not intraday wobble.
        smoothed = _sizing_conviction(thesis, cfg)
        # Concentration: a below-threshold-conviction thesis gets no capital (hold fewer,
        # stronger names). Gate on the SMOOTHED, PRE-decay conviction — decay relaxes a
        # stale value toward conviction_floor (0.5) and could otherwise re-inflate a
        # weak/zero name back above the drop threshold.
        if smoothed < cfg.min_conviction_to_hold:
            continue
        # Then the usual time-decay for the SIZING magnitude.
        eff_conviction = decay_conviction(smoothed, _age_days(thesis.last_evaluated_at, now), cfg)
        weight = size_position(eff_conviction, risk_from_sim(_latest_sim(session, thesis.symbol)), cfg)
        # Feedback loop: tilt size by the symbol's realized accuracy (no-op at 1.0
        # until enough outcomes accrue; shrink-only unless modifier_ceiling > 1.0).
        if weight > 0 and use_accuracy:
            weight *= _accuracy_mult(session, thesis.symbol, account_id, lcfg)
        if weight > 0:
            # Clamp at the per-name cap even if multiple live theses exist for one
            # symbol, so accumulation can't exceed max_position_weight.
            raw[thesis.symbol] = min(
                raw.get(thesis.symbol, 0.0) + weight, cfg.max_position_weight
            )

    # Concentration cap: keep only the top-N names by size, so a long tail of small theses
    # doesn't spread the book thin (the rest of the intended equity falls to cash/the ETF).
    # Held incumbents get hysteresis so a ±0.02 conviction wobble can't flip a position.
    raw = select_top_positions(
        raw, config.caps.max_positions,
        held=held_symbols or frozenset(),
        hysteresis=cfg.selection_hysteresis,
    )

    # Keep at least min_cash_weight in cash: scale equity down proportionally if over.
    total = sum(raw.values())
    max_equity = 1.0 - cfg.min_cash_weight
    if total > max_equity and total > 0:
        scale = max_equity / total
        raw = {s: w * scale for s, w in raw.items()}

    alloc = dict(raw)
    cash_remainder = max(0.0, 1.0 - sum(raw.values()))
    # Park the cash above the raw min_cash_weight buffer in the T-bill ETF, if configured,
    # so uninvested capital earns yield instead of sitting idle. A raw-cash buffer remains
    # for fees/settlement and to fund small buys without first selling the ETF.
    cash_etf = (getattr(config, "cash_etf", "") or "").strip().upper()
    if cash_etf and cash_etf != CASH_SYMBOL and cash_remainder > cfg.min_cash_weight:
        alloc[cash_etf] = alloc.get(cash_etf, 0.0) + (cash_remainder - cfg.min_cash_weight)
        alloc[CASH_SYMBOL] = cfg.min_cash_weight
    else:
        alloc[CASH_SYMBOL] = cash_remainder
    return alloc


__all__ = [
    "RiskMetrics",
    "risk_from_sim",
    "decay_conviction",
    "smoothed_conviction",
    "size_position",
    "select_top_positions",
    "accuracy_multiplier",
    "is_averaging_up_without_support",
    "compute_conviction_weights",
]
