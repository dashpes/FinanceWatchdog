"""Deterministic thesis invalidation and profit-taking exits (Phase 3).

A thesis is *invalidated* by hard data, never by LLM judgment — so an exit cannot
be hallucinated away and the trip is fully auditable. ``check_invalidation`` is a
pure function; the evaluator gathers the inputs and acts on the verdict (set the
thesis INVALIDATED → conviction 0 → next run sells toward a 0 weight).

Supported ``invalidation_conditions`` keys (all optional):
  * ``composite_drop``  — invalidate if composite score fell by >= this many points
  * ``price_drop_pct``  — invalidate on a drawdown >= this percent from entry
  * ``keywords``        — invalidate if a recent HIGH-priority news alert matched any

``check_invalidation`` covers only the DOWNSIDE ("the thesis broke"). ``check_exit``
is its take-profit twin ("the thesis played out"): same pure-predicate contract, but
it trips the EXITED status instead of INVALIDATED. Keys (all optional; a value <= 0
disables that trigger, so config defaults can be selectively overridden per thesis):
  * ``profit_target_pct``  — exit once the gain from entry reaches this percent
  * ``trailing_giveback_pct`` — exit once the position has given back this percent of
    its PEAK GAIN, armed at ``trailing_arm_pct``. This is the profit-protecting trail:
    it exits at ``peak_gain x (1 - giveback)``, which is always a REAL gain.
  * ``trailing_stop_pct``  — legacy price-distance trail: exit this percent below the
    post-entry high. OFF by default, because it measures the fall against the peak
    PRICE, so the exit lands at ``peak x (1 - trail)``: at the shipped 15%, a position
    had to peak at +17.6% merely to break even and a +12% peak exited at -4.8%. Live,
    that band was where nearly every position actually peaked, so the mechanism could
    only ever realize losses (audited 2026-07-21: 0 take-profits in ~30 round-trips).
    Kept for explicit per-thesis use; prefer ``trailing_giveback_pct``.
  * ``max_hold_days``      — exit after this many days regardless (the signal's edge
    has a shelf life; the walk-forward backtest models this same horizon exit)

Both trails require ``high_water_mark``; the caller maintains it. Whichever condition
fires first wins, and the profit target is checked before either trail.
"""

from __future__ import annotations

# Trailing stops arm only after the high-water gain reaches this percent (overridable
# per thesis via the `trailing_arm_pct` key).
DEFAULT_TRAILING_ARM_PCT = 10.0

# Percentage-point tolerance on threshold comparisons. Entry/high-water/last prices are
# floats, so a position sitting EXACTLY on a threshold (peak 11.2 from entry 10.0 giving
# back exactly 40%) otherwise fires or not on 14th-decimal noise. Exits must be
# deterministic at the boundary.
_EPS = 1e-9


def entry_basis(entry_conditions: dict | None) -> float | None:
    """The price a position's return is measured from (pure).

    Prefers the broker's real ``fill_cost`` (written once a live fill is reconciled)
    over the ``entry_price`` quote captured at idea time — matching the learning
    loop's realized-outcome basis. None if neither is usable.
    """
    entry = entry_conditions or {}
    for key in ("fill_cost", "entry_price"):
        try:
            value = float(entry[key])
            if value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            continue
    return None


def with_vol_target(conditions: dict, vol_target_pct: float | None) -> dict:
    """Pure: overlay a volatility-scaled profit target onto a conditions dict.

    Keeps the EARLIER of the vol-scaled target and any explicit per-thesis target: the
    explicit number reflects the thesis, the vol number reflects what the name can
    plausibly reach over the hold horizon, and taking the minimum banks a gain as soon
    as EITHER says the move has played out. ``None`` leaves the conditions untouched.
    """
    if vol_target_pct is None or vol_target_pct <= 0:
        return conditions
    out = dict(conditions)
    try:
        explicit = float(out.get("profit_target_pct") or 0.0)
    except (TypeError, ValueError):
        explicit = 0.0
    out["profit_target_pct"] = min(explicit, vol_target_pct) if explicit > 0 else vol_target_pct
    return out


def check_exit(
    exit_conditions: dict | None,
    *,
    entry_price: float | None = None,
    latest_price: float | None = None,
    high_water_mark: float | None = None,
    days_held: float | None = None,
) -> str | None:
    """Return a human-readable reason if any take-profit/exit condition fired, else None."""
    cond = exit_conditions or {}

    def _pos(key: str) -> float | None:
        try:
            value = float(cond[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if value > 0 else None

    target = _pos("profit_target_pct")
    if (
        target is not None
        and latest_price is not None
        and entry_price
        and entry_price > 0
    ):
        gain = (latest_price - entry_price) / entry_price * 100.0
        if gain >= target:
            return (
                f"profit target hit: up {gain:.1f}% from entry "
                f"${entry_price:.2f} (>= {target:g}%)"
            )

    trail_ready = (
        latest_price is not None
        and high_water_mark
        and high_water_mark > 0
        and entry_price
        and entry_price > 0
    )
    if trail_ready:
        arm = _pos("trailing_arm_pct") or DEFAULT_TRAILING_ARM_PCT
        hwm_gain = (high_water_mark - entry_price) / entry_price * 100.0
        gain = (latest_price - entry_price) / entry_price * 100.0

        # Give back at most this share of the PEAK GAIN. Exits at hwm_gain x (1-giveback),
        # so a protected position always books a real profit: a +12% peak with a 40%
        # giveback exits at +7.2% (the price-distance trail below would exit it at -4.8%).
        giveback = _pos("trailing_giveback_pct")
        if giveback is not None and hwm_gain >= arm - _EPS:
            keep = hwm_gain * (1.0 - min(giveback, 100.0) / 100.0)
            if gain <= keep + _EPS:
                return (
                    f"trailing giveback: gave back {hwm_gain - gain:.1f} of "
                    f"{hwm_gain:.1f} pts peak gain (>= {giveback:g}%), holding +{gain:.1f}%"
                )

        # Legacy price-distance trail (off by default — see the module docstring).
        trail = _pos("trailing_stop_pct")
        if trail is not None and hwm_gain >= arm - _EPS:
            fall = (high_water_mark - latest_price) / high_water_mark * 100.0
            if fall >= trail - _EPS:
                return (
                    f"trailing stop: {fall:.1f}% below the ${high_water_mark:.2f} high "
                    f"(>= {trail:g}%, peak gain {hwm_gain:.1f}%)"
                )

    horizon = _pos("max_hold_days")
    if horizon is not None and days_held is not None and days_held >= horizon:
        return f"held {days_held:.0f}d, past the {horizon:.0f}d horizon"

    return None


def check_invalidation(
    invalidation_conditions: dict | None,
    *,
    entry_composite: float | None = None,
    latest_composite: float | None = None,
    entry_price: float | None = None,
    latest_price: float | None = None,
    recent_alert_keywords: list[str] | None = None,
) -> str | None:
    """Return a human-readable reason if any invalidation condition fired, else None."""
    cond = invalidation_conditions or {}

    drop = cond.get("composite_drop")
    if (
        drop is not None
        and latest_composite is not None
        and entry_composite is not None
        and latest_composite <= entry_composite - float(drop)
    ):
        return (
            f"composite score fell {entry_composite - latest_composite:.0f} pts "
            f"(>= {drop}) to {latest_composite:.0f}"
        )

    pdp = cond.get("price_drop_pct")
    if (
        pdp is not None
        and latest_price is not None
        and entry_price
        and entry_price > 0
    ):
        drawdown = (entry_price - latest_price) / entry_price * 100.0
        if drawdown >= float(pdp):
            return f"price down {drawdown:.1f}% from entry ${entry_price:.2f} (>= {pdp}%)"

    keywords = cond.get("keywords") or []
    if keywords and recent_alert_keywords:
        wanted = {str(k).lower() for k in keywords}
        seen = {str(k).lower() for k in recent_alert_keywords}
        hit = sorted(wanted & seen)
        if hit:
            return f"invalidating news keyword(s) detected: {', '.join(hit)}"

    return None
