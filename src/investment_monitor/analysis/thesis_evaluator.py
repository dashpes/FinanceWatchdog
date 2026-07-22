"""LLM thesis evaluator (Phase 3) — develops and maintains theses.

The evaluator is the *judgment* layer: it reads fresh evidence and asks a local
LLM to (re)write the narrative and conviction for a name. Two hard safety rails
keep it honest:

* **Invalidation is deterministic** (``robo.invalidation.check_invalidation``),
  evaluated BEFORE the LLM. If a hard trigger fired, the thesis is invalidated and
  the LLM is never given a chance to argue it away.
* **Fail-safe parsing**: if the LLM is unavailable or its output can't be parsed,
  the prior thesis is left unchanged (conviction is never silently wiped, which
  would dump a live position).

Sizing/execution stay deterministic and behind the guardrail gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from investment_monitor.analysis.thesis_prompts import (
    THESIS_GENERATE_PROMPT,
    THESIS_UPDATE_PROMPT,
    THESIS_UPDATE_PROMPT_WITH_OUTCOME,
)
from investment_monitor.robo.invalidation import (
    check_exit,
    check_invalidation,
    entry_basis,
    with_vol_target,
)
from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    accuracy_stats_for_symbol,
    bench_thesis,
    exit_thesis,
    get_latest_price,
    get_latest_report,
    get_latest_score,
    get_prices,
    get_recent_news,
    invalidate_thesis,
    outcome_exists_for_date,
    record_conviction_update,
    record_thesis_outcome,
    save_thesis,
    set_target_weight,
    update_high_water,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.analysis.local_llm import LocalLLM
    from investment_monitor.robo.config import RoboConfig

# Below this absolute return, a position is treated as "no directional signal" and
# not recorded — avoids scoring a flat/just-opened thesis as a directional loss.
_MIN_SIGNAL_RETURN = 0.001

# Severe tokens that, if present in a recent headline, can trip a keyword invalidation.
_SEVERE_KEYWORDS = (
    "fraud", "bankruptcy", "sec", "investigation", "lawsuit", "delist",
    "default", "restate", "probe", "subpoena", "halt",
)


@dataclass
class ThesisUpdate:
    """Parsed LLM thesis output (defensively typed)."""

    narrative: str
    conviction: float
    invalidation_conditions: dict
    exit_conditions: dict
    raw: dict


def _extract_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a possibly-noisy LLM response (pure)."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _coerce_conviction(value: Any) -> float | None:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, c))


def _sanitize_invalidation(conditions: dict) -> dict:
    """Coerce LLM-supplied invalidation thresholds to sane values.

    Drop thresholds are *magnitudes*: the model sometimes emits a negative number
    (e.g. composite_drop: -15 meaning "a 15-point drop"), which would otherwise trip
    the condition immediately. Force positive, drop non-positive/garbage values, and
    keep only clean keyword strings.
    """
    out: dict = {}
    for key in ("composite_drop", "price_drop_pct"):
        if key in conditions:
            try:
                value = abs(float(conditions[key]))
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[key] = value
    keywords = conditions.get("keywords")
    if isinstance(keywords, list):
        clean = [str(k).strip() for k in keywords if str(k).strip()]
        if clean:
            out["keywords"] = clean
    return out


# LLM-proposed exit thresholds are clamped into these bands, so a hallucinated
# "profit_target_pct: 2" can't scalp-exit a live position and "5000" can't
# effectively disable the target. Keys absent/garbage -> fall back to config defaults.
_EXIT_CLAMPS = {
    "profit_target_pct": (10.0, 200.0),
    "trailing_giveback_pct": (10.0, 90.0),
    "trailing_stop_pct": (5.0, 50.0),
    "trailing_arm_pct": (0.0, 100.0),
    "max_hold_days": (10.0, 365.0),
}


def _sanitize_exit(conditions: dict) -> dict:
    """Coerce LLM-supplied take-profit thresholds to sane, clamped magnitudes."""
    out: dict = {}
    for key, (lo, hi) in _EXIT_CLAMPS.items():
        if key in conditions:
            try:
                value = abs(float(conditions[key]))
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[key] = max(lo, min(hi, value))
    return out


def parse_thesis_response(text: str) -> ThesisUpdate | None:
    """Parse an LLM thesis response into a ThesisUpdate, or None if unusable (pure)."""
    obj = _extract_json_object(text)
    if obj is None:
        return None
    conviction = _coerce_conviction(obj.get("conviction"))
    if conviction is None:
        return None
    narrative = str(obj.get("narrative") or "").strip()
    if not narrative:
        return None
    inv = obj.get("invalidation_conditions")
    inv = _sanitize_invalidation(inv if isinstance(inv, dict) else {})
    ext = obj.get("exit_conditions")
    ext = _sanitize_exit(ext if isinstance(ext, dict) else {})
    return ThesisUpdate(narrative=narrative, conviction=conviction,
                        invalidation_conditions=inv, exit_conditions=ext, raw=obj)


# --------------------------------------------------------------------------- #
# Context gathering (impure)
# --------------------------------------------------------------------------- #
def _score_block(session: "Session", symbol: str) -> tuple[str, float | None]:
    score = get_latest_score(session, symbol)
    if score is None:
        return "unavailable", None
    parts = []
    for name in ("value", "growth", "quality", "momentum", "sentiment", "composite"):
        val = getattr(score, f"{name}_score", None)
        if val is not None:
            parts.append(f"{name}={val:.0f}")
    return ", ".join(parts) or "unavailable", score.composite_score


def _news_block(session: "Session", symbol: str, hours: int = 72) -> tuple[str, list[str]]:
    items = get_recent_news(session, ticker=symbol, hours=hours)[:6]
    if not items:
        return "(none)", []
    lines = [f"- {(n.headline or '').strip()[:100]}" for n in items]
    severe = [
        kw for kw in _SEVERE_KEYWORDS
        if any(kw in (n.headline or "").lower() for n in items)
    ]
    return "\n".join(lines), severe


def _latest_close(session: "Session", symbol: str) -> float | None:
    price = get_latest_price(session, symbol)
    return float(price.close) if price and price.close is not None else None


def vol_scaled_conditions(session: "Session", symbol: str, ecfg, conditions: dict) -> dict:
    """Exit conditions with the volatility-scaled profit target overlaid (impure lookup).

    A flat percent target ignores what a name can actually deliver (+20% is ~4 sigma of
    a mega cap's 30-day move but ~1.4 sigma of a small cap's), which is why the live
    book's targets never fired once it pivoted to mega caps. Shared by the twice-daily
    evaluator and the hourly sentinel so both judge an exit identically. Fully fail-open:
    a missing sim or any lookup error leaves the conditions unchanged.
    """
    try:
        if not getattr(ecfg, "vol_scaled_target", False):
            return conditions
        from investment_monitor.robo.sizing import _latest_sim

        sim = _latest_sim(session, symbol)
        if sim is None:
            return conditions
        return with_vol_target(
            conditions,
            ecfg.vol_target_pct(getattr(sim, "volatility", None), conditions.get("max_hold_days")),
        )
    except Exception as exc:  # noqa: BLE001 - a sim lookup must never block an exit check
        logger.warning("vol-scaled target failed for {s}: {e}", s=symbol, e=exc)
        return conditions


def _evidence_hash(score_block: str, news_block: str) -> str:
    """Fingerprint of the qualitative evidence the re-eval prompt shows the LLM.

    Deliberately EXCLUDES price/outcome data: price action is handled by the
    deterministic invalidation/take-profit layers, and hashing it would re-trigger
    the LLM on every tick — the whole point is that conviction only moves when
    there is something new to reason about.
    """
    return hashlib.sha256(f"{score_block}\n{news_block}".encode("utf-8")).hexdigest()[:16]


def _last_evidence_hash(thesis: Thesis) -> str | None:
    """The evidence fingerprint of the most recent hash-bearing history point."""
    for point in reversed(thesis.conviction_history or []):
        if isinstance(point, dict) and point.get("evidence_hash"):
            return str(point["evidence_hash"])
    return None


def _parse_history_ts(value: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None
    if when is not None and when.tzinfo:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


def _conviction_baseline(thesis: Thesis, now: datetime) -> float | None:
    """Conviction as it stood ~24h ago, for the per-day rate limit.

    The last recorded point at or before ``now - 24h``; a younger thesis falls back
    to its oldest recorded point, then to the current conviction (bounding even the
    first re-eval of a fresh promotion). None only when nothing is recorded at all.
    """
    cutoff = now - timedelta(days=1)
    baseline: float | None = None
    oldest: float | None = None
    for point in thesis.conviction_history or []:
        if not isinstance(point, dict):
            continue
        try:
            conv = float(point.get("conviction"))
        except (TypeError, ValueError):
            continue
        if oldest is None:
            oldest = conv
        when = _parse_history_ts(point.get("ts"))
        if when is not None and when <= cutoff:
            baseline = conv  # history is append-only chronological: keep the latest pre-cutoff point
    if baseline is not None:
        return baseline
    if oldest is not None:
        return oldest
    return float(thesis.conviction) if thesis.conviction is not None else None


# --------------------------------------------------------------------------- #
# Feedback loop (Phase 6): realized-outcome capture + compact prompt context
# --------------------------------------------------------------------------- #
def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _days_held(thesis: Thesis) -> int:
    """Whole days since the thesis was created (entry-date proxy; >= 0)."""
    created = thesis.created_at
    if created is None:
        return 0
    created = created.replace(tzinfo=None) if created.tzinfo else created
    return max(0, (_utcnow_naive() - created).days)


def _realized_return(entry_price: Any, latest_price: float | None) -> float | None:
    """Paper return since entry (fraction). None if either price is missing/invalid."""
    try:
        ep = float(entry_price)
        if ep <= 0 or latest_price is None:
            return None
        return float(latest_price) / ep - 1.0
    except (TypeError, ValueError):
        return None


def _benchmark_return(session: "Session", symbol: str, days: int) -> float | None:
    """Benchmark (e.g. SPY) return over roughly the holding window. None if <2 rows."""
    try:
        prices = get_prices(session, symbol, days=max(days, 1) + 7)  # newest-first
        if len(prices) < 2:
            return None
        latest = prices[0].close
        oldest = prices[-1].close
        if not latest or not oldest or float(oldest) <= 0:
            return None
        return float(latest) / float(oldest) - 1.0
    except Exception:  # noqa: BLE001 - benchmark is best-effort context only
        return None


def _outcome_block(session: "Session", thesis: Thesis, latest_price: float | None, lcfg) -> str:
    """Build the COMPACT realized-performance + track-record line for the prompt.

    One factual line (numbers only). Returns "" when there is no entry price / no
    current price, so the caller falls back to the byte-for-byte-unchanged base
    prompt. Fully fail-open: any error yields "" rather than crashing evaluate().
    """
    try:
        # Prefer the real fill cost over the idea-time quote (see evaluate()), so the
        # line shown to the LLM matches the outcome the feedback loop records.
        basis = entry_basis(thesis.entry_conditions)
        ret = _realized_return(basis, latest_price)
        if ret is None:
            return ""
        days = _days_held(thesis)
        # Suppress just-opened noise — no realized line until the thesis has aged.
        if days < lcfg.min_days_held:
            return ""
        cap = float(lcfg.max_abs_return_pct) / 100.0
        ret_disp = max(-cap, min(cap, ret))
        entry_price = float(basis)
        line = (
            f"opened ${entry_price:.2f} ~{days}d ago; now ${float(latest_price):.2f} "
            f"({ret_disp * 100:+.1f}%)"
        )
        bench = _benchmark_return(session, lcfg.benchmark_symbol, days)
        if bench is not None:
            line += (
                f"; {lcfg.benchmark_symbol} {bench * 100:+.1f}% "
                f"(excess {(ret_disp - bench) * 100:+.1f}%)"
            )
        stats = accuracy_stats_for_symbol(
            session, thesis.symbol,
            ewma_halflife=lcfg.ewma_halflife, recent_window=lcfg.recent_window,
        )
        if stats["n"] >= lcfg.min_samples:
            line += (
                f". Track record (last {stats['n']} evals): "
                f"{stats['ewma_hit_rate'] * 100:.0f}% directionally right, "
                f"calibration {1.0 - stats['brier']:.2f}"
            )
        return line
    except Exception as exc:  # noqa: BLE001 - advisory context must never crash a run
        logger.warning("outcome block failed for {s}: {e}", s=thesis.symbol, e=exc)
        return ""


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class ThesisEvaluator:
    """Generates new theses and re-evaluates existing ones (LLM judgment layer)."""

    def __init__(self, local_llm: "LocalLLM | None", config: "RoboConfig") -> None:
        self._llm = local_llm
        self._config = config

    def _generate_text(self, prompt: str) -> str | None:
        if self._llm is None or not self._llm.is_available():
            return None
        try:
            response = self._llm.client.generate(
                model=self._llm.model,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 512},
            )
            return (response.get("response") or "").strip()
        except Exception as exc:  # noqa: BLE001 - any LLM failure -> caller keeps prior thesis
            logger.warning("thesis LLM call failed: {e}", e=exc)
            return None

    def evaluate(self, session: "Session", thesis: Thesis, *, account_id: str | None = None) -> str:
        """Re-evaluate one existing thesis. Returns the action taken.

        Order: deterministic invalidation first, then the deterministic take-profit
        exit check, then (if still live) an LLM conviction/narrative update. On any
        LLM failure the thesis is unchanged.
        """
        composite_str, latest_composite = _score_block(session, thesis.symbol)
        news_str, severe = _news_block(session, thesis.symbol)
        latest_price = _latest_close(session, thesis.symbol)
        entry = thesis.entry_conditions or {}
        lcfg = getattr(self._config, "learning", None)

        # Feedback loop: record the realized outcome (conviction HELD vs realized
        # price return) BEFORE any state change, so even an about-to-be-invalidated
        # thesis contributes its final, most-informative data point. Fully fail-open
        # — a learning-ledger bug must never stall the 24/7 maintenance loop.
        if lcfg is not None and lcfg.enabled and lcfg.record_outcomes:
            try:
                # Prefer the broker's real fill cost (written by the rebalance run once
                # a live position exists) over the quote captured at idea time, so the
                # learned outcome reflects the actual trade. Falls back to entry_price
                # in paper / before any fill — keeping that path byte-identical.
                realized = _realized_return(entry_basis(entry), latest_price)
                today = _utcnow_naive().date()
                # Record one outcome per symbol per day, only once the thesis has aged
                # past min_days_held and actually moved — so intraday re-evals don't
                # flood the window with autocorrelated copies of the same return.
                if (
                    realized is not None
                    and abs(realized) >= _MIN_SIGNAL_RETURN
                    and _days_held(thesis) >= lcfg.min_days_held
                    and not outcome_exists_for_date(
                        session, thesis.symbol, today, account_id=account_id
                    )
                ):
                    record_thesis_outcome(
                        session,
                        symbol=thesis.symbol,
                        conviction_at_eval=thesis.conviction,
                        realized_return=realized,
                        account_id=account_id,
                        thesis_id=thesis.id,
                        as_of_date=today,
                    )
            except Exception as exc:  # noqa: BLE001 - never crash on ledger write
                logger.warning("outcome capture failed for {s}: {e}", s=thesis.symbol, e=exc)

        # Stops tighter than ordinary noise are raised to the configured floor at CHECK
        # time, so the 5%/3% stops the LLM already persisted onto live theses stop
        # realizing losses on normal two-week wobble without needing a migration.
        icfg = getattr(self._config, "invalidation", None)
        inval_conditions = (
            icfg.floored(thesis.invalidation_conditions)
            if icfg is not None else thesis.invalidation_conditions
        )
        reason = check_invalidation(
            inval_conditions,
            entry_composite=entry.get("entry_composite"),
            latest_composite=latest_composite,
            entry_price=entry.get("entry_price"),
            latest_price=latest_price,
            recent_alert_keywords=severe,
        )
        if reason is not None:
            invalidate_thesis(session, thesis, reason)
            logger.info("thesis {s} INVALIDATED: {r}", s=thesis.symbol, r=reason)
            return "invalidated"

        # Take-profit twin: maintain the high-water mark, then check the deterministic
        # exit conditions (config defaults overlaid by any per-thesis overrides). Runs
        # AFTER invalidation so a broken thesis is always recorded as broken, and
        # BEFORE the LLM so a played-out thesis can't be argued into overstaying.
        update_high_water(session, thesis, latest_price)
        ecfg = getattr(self._config, "exits", None)
        if ecfg is not None and ecfg.enabled:
            conditions = vol_scaled_conditions(
                session, thesis.symbol, ecfg,
                {**ecfg.as_conditions(), **(thesis.exit_conditions or {})},
            )
            exit_reason = check_exit(
                conditions,
                entry_price=entry_basis(entry),
                latest_price=latest_price,
                high_water_mark=thesis.high_water_mark,
                days_held=_days_held(thesis),
            )
            if exit_reason is not None:
                exit_thesis(session, thesis, exit_reason)
                logger.info("thesis {s} EXITED: {r}", s=thesis.symbol, r=exit_reason)
                return "exited"

        # Evidence gate: if the qualitative evidence the prompt would show is identical
        # to what the LLM already saw at its last re-eval, there is nothing to update ON
        # — a re-run can only inject anchoring noise (live root cause of daily churn:
        # phi3:mini walked held names' conviction ±0.05/eval for days, straight through
        # market-closed weekends, forcing floor-crossing exits and re-buys). The
        # deterministic invalidation and take-profit checks above have already run.
        evidence = _evidence_hash(composite_str, news_str)
        acfg = getattr(self._config, "autonomy", None)
        if (
            acfg is not None
            and getattr(acfg, "skip_reeval_unchanged_evidence", False)
            and evidence == _last_evidence_hash(thesis)
        ):
            logger.info("thesis {s} re-eval skipped: evidence unchanged", s=thesis.symbol)
            return "unchanged_evidence"

        # Outcome-aware re-eval: inject the compact track-record block only when the
        # loop is enabled AND there is a real block to show; otherwise the base prompt
        # is used byte-for-byte unchanged (no KeyError, no empty section).
        outcome_block = ""
        if lcfg is not None and lcfg.enabled and lcfg.outcome_aware_reeval:
            outcome_block = _outcome_block(session, thesis, latest_price, lcfg)

        if outcome_block:
            prompt = THESIS_UPDATE_PROMPT_WITH_OUTCOME.format(
                symbol=thesis.symbol,
                narrative=thesis.narrative,
                conviction=f"{thesis.conviction:.2f}",
                score_block=composite_str,
                news_block=news_str,
                signals_block="(see scores/news)",
                outcome_block=outcome_block,
            )
        else:
            prompt = THESIS_UPDATE_PROMPT.format(
                symbol=thesis.symbol,
                narrative=thesis.narrative,
                conviction=f"{thesis.conviction:.2f}",
                score_block=composite_str,
                news_block=news_str,
                signals_block="(see scores/news)",
            )
        text = self._generate_text(prompt)
        update = parse_thesis_response(text) if text else None
        if update is None:
            logger.info("thesis {s} unchanged (no usable LLM update)", s=thesis.symbol)
            return "unchanged"

        thesis.narrative = update.narrative
        if update.invalidation_conditions:
            thesis.invalidation_conditions = update.invalidation_conditions
        if update.exit_conditions:
            thesis.exit_conditions = update.exit_conditions
        # Rate limit: bound the move against where conviction stood 24h ago, so an
        # anchored walk covers at most max_conviction_delta_per_day per day. When the
        # clamp BINDS, withhold the evidence hash — the model's view is not yet
        # absorbed, so the next cycle must re-run rather than skip, stepping toward it
        # at the capped rate. Hard breaks still exit via invalidation, never this path.
        new_conviction = update.conviction
        cap = getattr(getattr(self._config, "sizing", None), "max_conviction_delta_per_day", 0.0)
        clamp_bound = False
        if cap > 0:
            baseline = _conviction_baseline(thesis, _utcnow_naive())
            if baseline is not None:
                bounded = min(max(new_conviction, baseline - cap), baseline + cap)
                if bounded != new_conviction:
                    clamp_bound = True
                    logger.info(
                        "thesis {s} conviction move clamped {a:.2f}->{b:.2f} (±{c:.2f}/day)",
                        s=thesis.symbol, a=new_conviction, b=bounded, c=cap,
                    )
                    new_conviction = bounded
        record_conviction_update(
            session, thesis, new_conviction, trigger="llm_reeval",
            evidence_hash=None if clamp_bound else evidence,
        )
        return "updated"

    def generate(self, session: "Session", symbol: str, *, account_id: str | None = None) -> Thesis | None:
        """Create a fresh DRAFT thesis for a new name (None if no usable LLM output)."""
        composite_str, latest_composite = _score_block(session, symbol)
        news_str, _ = _news_block(session, symbol)
        report = get_latest_report(session, symbol)
        recommendation = (report.recommendation if report else None) or "n/a"
        latest_price = _latest_close(session, symbol)

        prompt = THESIS_GENERATE_PROMPT.format(
            symbol=symbol,
            score_block=composite_str,
            recommendation=recommendation,
            news_block=news_str,
        )
        text = self._generate_text(prompt)
        update = parse_thesis_response(text) if text else None
        if update is None:
            return None

        thesis = Thesis(
            symbol=symbol,
            account_id=account_id,
            narrative=update.narrative,
            conviction=update.conviction,
            target_weight=0.0,
            entry_conditions={
                "entry_composite": latest_composite,
                "entry_price": latest_price,
            },
            invalidation_conditions=update.invalidation_conditions,
            exit_conditions=update.exit_conditions or None,
            evidence_refs={"report_id": getattr(report, "id", None)},
            status=ThesisStatus.DRAFT.value,
            conviction_history=[{"conviction": update.conviction, "trigger": "generated"}],
        )
        save_thesis(session, thesis)
        return thesis


def refresh_target_weights(session: "Session", config: "RoboConfig", *, account_id: str | None = None) -> None:
    """Recompute and cache each active thesis's sized target weight."""
    from investment_monitor.robo.sizing import compute_conviction_weights
    from investment_monitor.storage import get_active_theses

    weights = compute_conviction_weights(session, config, account_id=account_id)
    for thesis in get_active_theses(session, account_id):
        set_target_weight(session, thesis, weights.get(thesis.symbol, 0.0))


# --------------------------------------------------------------------------- #
# Book hygiene: the active set is a WORKING set, not an archive
# --------------------------------------------------------------------------- #
def _sustained_sub_floor(thesis: Thesis, config: "RoboConfig", now: datetime) -> bool:
    """Pure-ish: has this thesis been unsizeable (below the conviction floor) so long
    that daily LLM maintenance is wasted on it?

    True only when the SMOOTHED conviction (the same value sizing gates on) is below
    ``min_conviction_to_hold`` AND no recorded conviction point inside the last
    ``bench_after_days`` reached the floor AND the thesis is at least that old — so a
    fresh promotion or a brief dip is never benched.
    """
    from investment_monitor.robo.sizing import _sizing_conviction

    acfg = config.autonomy
    if acfg.bench_after_days <= 0:
        return False
    floor = config.sizing.min_conviction_to_hold
    if _sizing_conviction(thesis, config.sizing) >= floor:
        return False
    created = thesis.created_at
    if created is None:
        return False
    created = created.replace(tzinfo=None) if created.tzinfo else created
    if (now - created).total_seconds() < acfg.bench_after_days * 86400:
        return False
    cutoff = now - timedelta(days=acfg.bench_after_days)
    for point in thesis.conviction_history or []:
        if not isinstance(point, dict):
            continue
        ts = point.get("ts")
        try:
            when = datetime.fromisoformat(str(ts)) if ts else None
        except ValueError:
            when = None
        if when is not None and when.tzinfo:
            when = when.replace(tzinfo=None)
        # Points without a timestamp predate the window bookkeeping — ignore them.
        if when is None or when < cutoff:
            continue
        try:
            if float(point.get("conviction", 0.0)) >= floor:
                return False  # showed real strength inside the window
        except (TypeError, ValueError):
            continue
    return True


def run_maintenance(
    session: "Session",
    evaluator: ThesisEvaluator,
    config: "RoboConfig",
    *,
    account_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """One maintenance pass over the live book. Returns action counts.

    ACTIVE theses get the full daily ``evaluate`` (invalidation -> take-profit ->
    LLM). A name that has stayed below the conviction floor for
    ``autonomy.bench_after_days`` is BENCHED to WATCH — kept, but out of the daily
    LLM rotation, so an ever-growing book can't starve the schedule (83 theses x a
    14B model = ~4h/pass on a Pi). Benched theses are re-evaluated only every
    ``autonomy.bench_reeval_days`` and return to ACTIVE when conviction recovers.
    Finally the ``autonomy.max_active_theses`` cap benches the weakest overflow.

    Benching never touches positions: a sub-floor name gets no capital either way,
    so its held shares (if any) are already being sold toward 0 by the sizing floor.
    """
    from investment_monitor.robo.sizing import _sizing_conviction

    now = now or _utcnow_naive()
    acfg = config.autonomy
    floor = config.sizing.min_conviction_to_hold
    counts = {"invalidated": 0, "exited": 0, "updated": 0, "unchanged": 0,
              "unchanged_evidence": 0, "benched": 0, "revived": 0, "skipped_benched": 0}

    from investment_monitor.storage import get_active_theses

    live = get_active_theses(session, account_id)
    active = [t for t in live if t.status == ThesisStatus.ACTIVE.value]
    benched = [t for t in live if t.status == ThesisStatus.WATCH.value]

    for thesis in active:
        counts[evaluator.evaluate(session, thesis, account_id=account_id)] += 1
        if thesis.status == ThesisStatus.ACTIVE.value and _sustained_sub_floor(thesis, config, now):
            bench_thesis(
                session, thesis,
                f"conviction below {floor:.2f} for {acfg.bench_after_days:g}d",
            )
            counts["benched"] += 1

    for thesis in benched:
        le = thesis.last_evaluated_at
        if le is not None:
            le = le.replace(tzinfo=None) if le.tzinfo else le
            if (now - le).total_seconds() < acfg.bench_reeval_days * 86400:
                counts["skipped_benched"] += 1
                continue
        action = evaluator.evaluate(session, thesis, account_id=account_id)
        counts[action] += 1
        if (
            thesis.status == ThesisStatus.WATCH.value
            and _sizing_conviction(thesis, config.sizing) >= floor
        ):
            thesis.status = ThesisStatus.ACTIVE.value
            session.flush()
            counts["revived"] += 1
            logger.info("thesis {s} revived from the bench", s=thesis.symbol)

    # Hard cap: the book is a working set. Weakest overflow goes to the bench.
    if acfg.max_active_theses and acfg.max_active_theses > 0:
        actives = [
            t for t in get_active_theses(session, account_id)
            if t.status == ThesisStatus.ACTIVE.value
        ]
        overflow = len(actives) - acfg.max_active_theses
        if overflow > 0:
            actives.sort(key=lambda t: (_sizing_conviction(t, config.sizing), t.symbol))
            for thesis in actives[:overflow]:
                bench_thesis(session, thesis, f"book over max_active_theses ({acfg.max_active_theses})")
                counts["benched"] += 1

    return counts
