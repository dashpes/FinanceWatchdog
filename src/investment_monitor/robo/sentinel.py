"""Intraday sentinel — an hourly market-hours watchdog over OPEN positions.

The scheduled trade runs happen twice a day, so a position that gaps down on bad
news mid-morning would otherwise sit unexamined for hours. The sentinel closes
that blind spot with a deliberately tiny mandate:

- it re-checks ONLY names with a live thesis (never the discovery universe);
- it may only FLAG (alert email) or TRIP the existing deterministic invalidation
  (which zeroes conviction so the next gated trade run sells toward zero);
- it can NEVER buy, place orders, or touch sizing — there is no order path here.

Runs hourly via systemd during market hours; outside regular trading hours it is
a no-op, so the timer needs no market-calendar knowledge.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from investment_monitor.collectors.prices import PriceCollector
from investment_monitor.config import Settings
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.invalidation import check_invalidation
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.storage import (
    AlertSent,
    SIGNAL_ITEM_CODES,
    alert_exists_by_dedup_key,
    get_active_theses,
    get_latest_price,
    get_material_events,
    get_session,
    invalidate_thesis,
    save_alert,
)

# A material 8-K stays alert-worthy this long; the dedup key stops repeats.
_EVENT_WINDOW_DAYS = 3
_EVENT_DEDUP_HOURS = 96


def run_sentinel(
    settings: Settings, config: RoboConfig, *, now: datetime | None = None
) -> dict:
    """One sentinel pass. Returns {status, checked, tripped, flagged}."""
    if not is_market_open(now):
        return {"status": "market_closed", "checked": 0, "tripped": [], "flagged": []}

    tripped: list[str] = []
    flagged: list[str] = []
    with get_session() as session:
        theses = get_active_theses(session, config.account_id or None)
        if not theses:
            return {"status": "ok", "checked": 0, "tripped": [], "flagged": []}
        symbols = sorted({t.symbol for t in theses})

        # Refresh quotes for held names only (cheap; yfinance includes today's
        # evolving session bar). Best-effort — stale prices just mean no new trips.
        try:
            collector = PriceCollector(session, settings, days_to_fetch=5)
            asyncio.run(collector.collect(symbols))
        except Exception as exc:  # noqa: BLE001 - checks proceed on stored prices
            logger.warning(f"sentinel: price refresh failed ({exc}); using stored prices")

        for thesis in theses:
            latest = get_latest_price(session, thesis.symbol)
            latest_price = float(latest.close) if latest and latest.close else None
            entry_price = (thesis.entry_conditions or {}).get("entry_price")
            reason = check_invalidation(
                thesis.invalidation_conditions,
                entry_price=float(entry_price) if entry_price else None,
                latest_price=latest_price,
            )
            if reason:
                invalidate_thesis(session, thesis, f"sentinel: {reason}")
                tripped.append(f"{thesis.symbol}: {reason}")
                continue

            # Fresh HIGH-SIGNAL 8-K on a held name: flag it (once per filing).
            for ev in get_material_events(session, thesis.symbol, days=_EVENT_WINDOW_DAYS):
                signal = sorted(set(ev.items or []) & SIGNAL_ITEM_CODES)
                if not signal:
                    continue
                key = f"sentinel-8k:{ev.sec_url}"
                if alert_exists_by_dedup_key(session, key, hours=_EVENT_DEDUP_HOURS):
                    continue
                msg = (
                    f"{thesis.symbol}: material 8-K (items {', '.join(signal)}) "
                    f"filed {ev.filed_date}"
                )
                save_alert(session, AlertSent(
                    alert_type="sentinel_8k", ticker=thesis.symbol, message=msg,
                    priority="high", channel="email", dedup_key=key,
                ))
                flagged.append(msg)

        checked = len(theses)

    if tripped or flagged:
        _notify(settings, tripped=tripped, flagged=flagged)
    logger.info(
        f"sentinel: checked {checked}, tripped {len(tripped)}, flagged {len(flagged)}"
    )
    return {"status": "ok", "checked": checked, "tripped": tripped, "flagged": flagged}


def _notify(settings: Settings, *, tripped: list[str], flagged: list[str]) -> None:
    """One consolidated alert email per pass (fail-open, plain text)."""
    try:
        from investment_monitor.robo.notify import _compose, _send, _subject

        lines: list[str] = []
        if tripped:
            lines.append("Invalidated (will sell toward zero at the next trade run):")
            lines.extend(f"  - {t}" for t in tripped)
        if flagged:
            lines.append("Material 8-K filings on held names:")
            lines.extend(f"  - {f}" for f in flagged)
        title = "Sentinel Alert"
        _send(settings, _compose(title, "\n".join(lines)), subject=_subject(title))
    except Exception as exc:  # noqa: BLE001 - alerting must never fail the pass
        logger.warning(f"sentinel notify failed (ignored): {exc}")
