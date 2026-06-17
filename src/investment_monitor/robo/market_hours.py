"""US equity market-hours check (pure).

The autonomous loop's *research* runs 24/7, but live order placement is gated to
regular trading hours so the broker doesn't queue trades outside the session.
Holidays are not modeled (v1) — the broker rejects orders on closed days anyway.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)


def is_market_open(now: datetime | None = None) -> bool:
    """True during US equity regular trading hours (Mon-Fri, 9:30-16:00 ET).

    ``now`` may be tz-aware (any zone) or naive (assumed to already be ET). Defaults
    to the current time in ET.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Saturday / Sunday
        return False
    return _OPEN <= et.time() < _CLOSE
