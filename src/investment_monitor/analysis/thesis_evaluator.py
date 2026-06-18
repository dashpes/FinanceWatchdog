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

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from investment_monitor.analysis.thesis_prompts import (
    THESIS_GENERATE_PROMPT,
    THESIS_UPDATE_PROMPT,
    THESIS_UPDATE_PROMPT_WITH_OUTCOME,
)
from investment_monitor.robo.invalidation import check_invalidation
from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    accuracy_stats_for_symbol,
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
    return ThesisUpdate(narrative=narrative, conviction=conviction,
                        invalidation_conditions=inv, raw=obj)


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
        entry = thesis.entry_conditions or {}
        ret = _realized_return(entry.get("entry_price"), latest_price)
        if ret is None:
            return ""
        days = _days_held(thesis)
        # Suppress just-opened noise — no realized line until the thesis has aged.
        if days < lcfg.min_days_held:
            return ""
        cap = float(lcfg.max_abs_return_pct) / 100.0
        ret_disp = max(-cap, min(cap, ret))
        entry_price = float(entry["entry_price"])
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

        Order: deterministic invalidation first, then (if still valid) an LLM
        conviction/narrative update. On any LLM failure the thesis is unchanged.
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
                realized = _realized_return(entry.get("entry_price"), latest_price)
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

        reason = check_invalidation(
            thesis.invalidation_conditions,
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
        record_conviction_update(session, thesis, update.conviction, trigger="llm_reeval")
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
