"""Deterministic thesis invalidation (Phase 3).

A thesis is *invalidated* by hard data, never by LLM judgment — so an exit cannot
be hallucinated away and the trip is fully auditable. ``check_invalidation`` is a
pure function; the evaluator gathers the inputs and acts on the verdict (set the
thesis INVALIDATED → conviction 0 → next run sells toward a 0 weight).

Supported ``invalidation_conditions`` keys (all optional):
  * ``composite_drop``  — invalidate if composite score fell by >= this many points
  * ``price_drop_pct``  — invalidate on a drawdown >= this percent from entry
  * ``keywords``        — invalidate if a recent HIGH-priority news alert matched any
"""

from __future__ import annotations


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
