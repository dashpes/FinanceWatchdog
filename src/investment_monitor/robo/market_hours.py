"""US equity market-hours check (pure).

The autonomous loop's *research* runs 24/7, but live order placement is gated to
regular trading hours so the broker doesn't queue trades outside the session.

NYSE full-day holidays and 13:00-ET early-close half-days ARE modeled (hardcoded
below) so we never *assume* the market is open on a weekday holiday — the broker
rejecting the order is a backstop, not the control. UPDATE THE TABLES ANNUALLY.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)

# NYSE full-day closures. 2026 is verified; 2027 is best-effort — VERIFY/UPDATE
# annually. A wrongly-included date only skips a trading day (safe); a wrongly-
# omitted one is backstopped by broker rejection + graceful place_failed handling.
_HOLIDAYS = frozenset({
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
})
# Half-days closing at 13:00 ET (day after Thanksgiving, Christmas Eve when a weekday).
_EARLY_CLOSE_DAYS = frozenset({
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
})


def is_market_open(now: datetime | None = None) -> bool:
    """True during US equity regular trading hours (Mon-Fri, 9:30-16:00 ET).

    Returns False on weekends, NYSE holidays, and after 13:00 ET on early-close
    half-days. ``now`` may be tz-aware (any zone) or naive (assumed ET); defaults
    to the current time in ET.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Saturday / Sunday
        return False
    today = et.date()
    if today in _HOLIDAYS:
        return False
    close = _EARLY_CLOSE if today in _EARLY_CLOSE_DAYS else _CLOSE
    return _OPEN <= et.time() < close
