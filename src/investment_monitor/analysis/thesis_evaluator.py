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
from typing import TYPE_CHECKING, Any

from loguru import logger

from investment_monitor.analysis.thesis_prompts import (
    THESIS_GENERATE_PROMPT,
    THESIS_UPDATE_PROMPT,
)
from investment_monitor.robo.invalidation import check_invalidation
from investment_monitor.storage import (
    Thesis,
    ThesisStatus,
    get_latest_price,
    get_latest_report,
    get_latest_score,
    get_recent_news,
    invalidate_thesis,
    record_conviction_update,
    save_thesis,
    set_target_weight,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.analysis.local_llm import LocalLLM
    from investment_monitor.robo.config import RoboConfig

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
    inv = inv if isinstance(inv, dict) else {}
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
