"""Intraday sentinel — an hourly market-hours watchdog over OPEN positions.

The scheduled trade runs happen twice a day, so a position that gaps down on bad
news mid-morning would otherwise sit unexamined for hours. The sentinel closes
that blind spot with a deliberately tiny mandate:

- it re-checks ONLY names with a live thesis (never the discovery universe);
- it may only FLAG (alert email) or TRIP the existing deterministic invalidation
  or take-profit exit (either zeroes conviction so the next gated trade run sells
  toward zero) — plus maintain each thesis's high-water mark for the trailing stop;
- it can NEVER buy, place orders, or touch sizing — there is no order path here.

Runs hourly via systemd during market hours; outside regular trading hours it is
a no-op, so the timer needs no market-calendar knowledge.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from loguru import logger

from investment_monitor.analysis.thesis_evaluator import vol_scaled_conditions
from investment_monitor.collectors.prices import PriceCollector
from investment_monitor.config import Settings
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.invalidation import check_exit, check_invalidation, entry_basis
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.storage import (
    AlertSent,
    SIGNAL_ITEM_CODES,
    ThesisStatus,
    alert_exists_by_dedup_key,
    exit_thesis,
    get_active_theses,
    get_latest_price,
    get_material_events,
    get_session,
    invalidate_thesis,
    save_alert,
    update_high_water,
)

# A material 8-K stays alert-worthy this long; the dedup key stops repeats.
_EVENT_WINDOW_DAYS = 3
_EVENT_DEDUP_HOURS = 96


def _days_held(thesis, now: datetime | None = None) -> float:
    """Days since the thesis was created (entry-date proxy; >= 0)."""
    created = thesis.created_at
    if created is None:
        return 0.0
    created = created.replace(tzinfo=None) if created.tzinfo else created
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    now = now.replace(tzinfo=None) if now.tzinfo else now
    return max(0.0, (now - created).total_seconds() / 86400.0)


def run_sentinel(
    settings: Settings, config: RoboConfig, *, now: datetime | None = None
) -> dict:
    """One sentinel pass. Returns {status, checked, tripped, exited, flagged}."""
    if not is_market_open(now):
        return {"status": "market_closed", "checked": 0, "tripped": [], "exited": [],
                "flagged": []}

    tripped: list[str] = []
    exited: list[str] = []
    flagged: list[str] = []
    ecfg = getattr(config, "exits", None)
    icfg = getattr(config, "invalidation", None)
    exit_defaults = ecfg.as_conditions() if ecfg is not None and ecfg.enabled else None
    with get_session() as session:
        # ACTIVE only: benched (WATCH) names are unsized by definition — any held
        # shares are already being sold toward 0 — so hourly quotes for them are
        # wasted; they keep their weekly re-eval coverage in maintenance instead.
        theses = [
            t for t in get_active_theses(session, config.account_id or None)
            if t.status == ThesisStatus.ACTIVE.value
        ]
        if not theses:
            return {"status": "ok", "checked": 0, "tripped": [], "exited": [], "flagged": []}
        symbols = sorted({t.symbol for t in theses})

        # Refresh quotes for held names only (cheap; yfinance includes today's
        # evolving session bar). Best-effort — stale prices just mean no new trips.
        try:
            collector = PriceCollector(session, settings, days_to_fetch=5)
            asyncio.run(collector.collect(symbols))
        except Exception as exc:  # noqa: BLE001 - checks proceed on stored prices
            logger.warning(f"sentinel: price refresh failed ({exc}); using stored prices")

        for thesis in theses:
            # Each name is its own unit of work: commit a trip/flag the instant it is made
            # so a failure on a LATER name can never roll it back, and a single bad name is
            # logged and skipped rather than aborting the whole pass (fail-open, like the
            # price refresh above). Both used to share one transaction that committed only
            # at loop end — any mid-pass error silently discarded every trip in the pass.
            try:
                latest = get_latest_price(session, thesis.symbol)
                latest_price = float(latest.close) if latest and latest.close else None
                entry_price = (thesis.entry_conditions or {}).get("entry_price")
                reason = check_invalidation(
                    icfg.floored(thesis.invalidation_conditions)
                    if icfg is not None else thesis.invalidation_conditions,
                    entry_price=float(entry_price) if entry_price else None,
                    latest_price=latest_price,
                )
                if reason:
                    invalidate_thesis(session, thesis, f"sentinel: {reason}")
                    session.commit()
                    tripped.append(f"{thesis.symbol}: {reason}")
                    continue

                # Take-profit twin: keep the high-water mark fresh intraday (the
                # trailing stop is only as good as its peak), then check the exit
                # conditions the same way the twice-daily evaluator does.
                update_high_water(session, thesis, latest_price)
                if exit_defaults is not None:
                    exit_reason = check_exit(
                        vol_scaled_conditions(
                            session, thesis.symbol, ecfg,
                            {**exit_defaults, **(thesis.exit_conditions or {})},
                        ),
                        entry_price=entry_basis(thesis.entry_conditions),
                        latest_price=latest_price,
                        high_water_mark=thesis.high_water_mark,
                        days_held=_days_held(thesis, now),
                    )
                    if exit_reason:
                        exit_thesis(session, thesis, f"sentinel: {exit_reason}")
                        session.commit()
                        exited.append(f"{thesis.symbol}: {exit_reason}")
                        continue
                session.commit()  # persist a raised high-water mark even with no trip

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
                    session.commit()
                    flagged.append(msg)
            except Exception as exc:  # noqa: BLE001 - one bad name must not abort the pass or discard earlier trips
                session.rollback()
                logger.warning(f"sentinel: check failed for {thesis.symbol} ({exc}); skipped")

        checked = len(theses)

    if tripped or exited or flagged:
        _notify(settings, tripped=tripped, exited=exited, flagged=flagged)
    logger.info(
        f"sentinel: checked {checked}, tripped {len(tripped)}, "
        f"exited {len(exited)}, flagged {len(flagged)}"
    )
    return {"status": "ok", "checked": checked, "tripped": tripped, "exited": exited,
            "flagged": flagged}


def _notify(
    settings: Settings, *, tripped: list[str], exited: list[str], flagged: list[str]
) -> None:
    """One consolidated alert email per pass (fail-open, plain text)."""
    try:
        from investment_monitor.robo.notify import _compose, _send, _subject

        lines: list[str] = []
        if tripped:
            lines.append("Invalidated (will sell toward zero at the next trade run):")
            lines.extend(f"  - {t}" for t in tripped)
        if exited:
            lines.append("Take-profit exits (will sell at the next trade run):")
            lines.extend(f"  - {e}" for e in exited)
        if flagged:
            lines.append("Material 8-K filings on held names:")
            lines.extend(f"  - {f}" for f in flagged)
        title = "Sentinel Alert"
        _send(settings, _compose(title, "\n".join(lines)), subject=_subject(title))
    except Exception as exc:  # noqa: BLE001 - alerting must never fail the pass
        logger.warning(f"sentinel notify failed (ignored): {exc}")
