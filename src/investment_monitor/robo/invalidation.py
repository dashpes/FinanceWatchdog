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
  * ``trailing_stop_pct``  — exit this percent below the post-entry high-water mark,
    armed only once the high-water gain reaches ``trailing_arm_pct`` (default 10) so
    a fresh, flat position can't be stopped out on noise
  * ``max_hold_days``      — exit after this many days regardless (the signal's edge
    has a shelf life; the walk-forward backtest models this same horizon exit)
"""

from __future__ import annotations

# Trailing stop arms only after the high-water gain reaches this percent (overridable
# per thesis via the `trailing_arm_pct` key).
DEFAULT_TRAILING_ARM_PCT = 10.0


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

    trail = _pos("trailing_stop_pct")
    if (
        trail is not None
        and latest_price is not None
        and high_water_mark
        and high_water_mark > 0
        and entry_price
        and entry_price > 0
    ):
        arm = _pos("trailing_arm_pct") or DEFAULT_TRAILING_ARM_PCT
        hwm_gain = (high_water_mark - entry_price) / entry_price * 100.0
        fall = (high_water_mark - latest_price) / high_water_mark * 100.0
        if hwm_gain >= arm and fall >= trail:
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
