"""US equity market-hours check (pure).

The autonomous loop's *research* runs 24/7, but live order placement is gated to
regular trading hours so the broker doesn't queue trades outside the session.

NYSE full-day holidays and 13:00-ET early-close half-days are MODELED so we never
*assume* the market is open on a weekday holiday — the broker rejecting the order
is a backstop, not the control. The calendars are COMPUTED per year (nth-weekday
rules, Western computus Good Friday, and NYSE weekend-observance), not hardcoded,
so there is no annual-staleness foot-gun: any future year is derived on demand.
``is_trading_day`` exposes the full-day calendar publicly.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_EARLY_CLOSE = time(13, 0)

# First year the NYSE observed Juneteenth as a full-day closure.
_JUNETEENTH_FIRST_YEAR = 2022


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon=0 .. Sun=6) of ``month`` in ``year`` (1-based n)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` (Mon=0 .. Sun=6) of ``month`` in ``year``."""
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian (Western) Easter Sunday via the Anonymous/Meeus computus."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _good_friday(year: int) -> date:
    """The Friday before Western Easter Sunday."""
    return _easter_sunday(year) - timedelta(days=2)


def _observed(d: date) -> date:
    """NYSE weekend observance for a FIXED-date holiday.

    A Saturday holiday is observed the preceding Friday; a Sunday holiday is
    observed the following Monday. Callers handle the NYSE New-Year exception
    (a Saturday New Year's Day is NOT observed) separately.
    """
    if d.weekday() == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def _holidays(year: int) -> frozenset[date]:
    """COMPUTED set of NYSE full-day closures for ``year``.

    Fixed-date holidays (New Year, Juneteenth, Independence Day, Christmas) get
    weekend observance via ``_observed``, with the NYSE exception that a Saturday
    New Year's Day is NOT observed (no preceding-Dec-31 closure). The remaining
    holidays are nth-/last-weekday rules plus Western-computus Good Friday.
    """
    days: set[date] = set()

    # New Year's Day — observed, EXCEPT a Saturday Jan 1 is not observed.
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        days.add(_observed(new_year))

    days.add(_nth_weekday(year, 1, 0, 3))   # MLK Day: 3rd Monday of January
    days.add(_nth_weekday(year, 2, 0, 3))   # Washington's Birthday: 3rd Monday of February
    days.add(_good_friday(year))            # Good Friday
    days.add(_last_weekday(year, 5, 0))     # Memorial Day: last Monday of May

    # Juneteenth — NYSE holiday from 2022 onward only; observed for weekends.
    if year >= _JUNETEENTH_FIRST_YEAR:
        days.add(_observed(date(year, 6, 19)))

    days.add(_observed(date(year, 7, 4)))   # Independence Day — observed
    days.add(_nth_weekday(year, 9, 0, 1))   # Labor Day: 1st Monday of September
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving: 4th Thursday of November
    days.add(_observed(date(year, 12, 25)))  # Christmas — observed

    return frozenset(days)


@lru_cache(maxsize=None)
def _early_close_days(year: int) -> frozenset[date]:
    """COMPUTED set of 13:00-ET NYSE half-days for ``year``.

    Models the same half-days the NYSE keeps:
      * the Friday after Thanksgiving (always — it is a regular open day);
      * July 3, when July 4 is observed on the 4th itself (i.e. Jul 4 is a weekday)
        so Jul 3 is an open weekday immediately before the full holiday;
      * December 24 (Christmas Eve), when it is a weekday and is NOT itself the
        observed Christmas closure (a Saturday Dec 25 is observed on Fri Dec 24,
        a full closure, so no half-day then).
    """
    days: set[date] = set()

    # Friday after Thanksgiving (Thanksgiving is the 4th Thursday of November).
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))

    # July 3 half-day: only when Jul 4 is a weekday (observed on the 4th) AND Jul 3
    # is itself a weekday. If Jul 4 is Sat (observed Fri Jul 3 = full close) or Sun
    # (observed Mon Jul 5; Jul 3 is Sat), there is no Jul-3 half-day.
    jul4 = date(year, 7, 4)
    jul3 = date(year, 7, 3)
    if jul4.weekday() < 5 and jul3.weekday() < 5:
        days.add(jul3)

    # December 24 half-day: only when it is a weekday AND not the observed Christmas.
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5 and dec24 not in _holidays(year):
        days.add(dec24)

    return frozenset(days)


def is_trading_day(d: date) -> bool:
    """True if ``d`` is a NYSE trading day: a weekday that is not a full-day holiday.

    Note an early-close half-day IS a trading day (the market is open, just for a
    shorter session) — only weekends and full-day closures return False.
    """
    if d.weekday() >= 5:  # Saturday / Sunday
        return False
    return d not in _holidays(d.year)


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
    if not is_trading_day(today):
        return False
    close = _EARLY_CLOSE if today in _early_close_days(today.year) else _CLOSE
    return _OPEN <= et.time() < close
