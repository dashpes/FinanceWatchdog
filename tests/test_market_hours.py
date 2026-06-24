"""Tests for the COMPUTED NYSE calendar in robo.market_hours.

The full-day-holiday and early-close sets used to be hardcoded and needed manual
annual updates (a staleness foot-gun). They are now computed per year. The HARD
correctness anchor here is that the computed 2026/2027 sets EQUAL the previously
hand-verified hardcoded values; future years are pinned to authoritative known
NYSE math (nth-weekday rules, Western-computus Good Friday, weekend observance).
"""

from __future__ import annotations

from datetime import date, datetime

from zoneinfo import ZoneInfo

from investment_monitor.robo.market_hours import (
    _easter_sunday,
    _early_close_days,
    _holidays,
    is_market_open,
    is_trading_day,
)

# --------------------------------------------------------------------------- #
# Parity anchor: the OLD hardcoded values (hand-verified) captured verbatim.
# The computed sets MUST equal these for 2026 and 2027.
# --------------------------------------------------------------------------- #
_OLD_HOLIDAYS_2026 = frozenset({
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
})
_OLD_HOLIDAYS_2027 = frozenset({
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
})
_OLD_EARLY_CLOSE_2026 = frozenset({date(2026, 11, 27), date(2026, 12, 24)})
_OLD_EARLY_CLOSE_2027 = frozenset({date(2027, 11, 26)})


def test_holiday_parity_2026():
    assert _holidays(2026) == _OLD_HOLIDAYS_2026


def test_holiday_parity_2027():
    assert _holidays(2027) == _OLD_HOLIDAYS_2027


def test_early_close_parity_2026():
    assert _early_close_days(2026) == _OLD_EARLY_CLOSE_2026


def test_early_close_parity_2027():
    assert _early_close_days(2027) == _OLD_EARLY_CLOSE_2027


# --------------------------------------------------------------------------- #
# Future-year math pinned to authoritative known NYSE dates.
# --------------------------------------------------------------------------- #
def test_nth_and_last_weekday_holidays_2030():
    h = _holidays(2030)
    assert date(2030, 1, 21) in h   # MLK: 3rd Monday of January
    assert date(2030, 2, 18) in h   # Washington's Birthday: 3rd Monday of February
    assert date(2030, 5, 27) in h   # Memorial Day: last Monday of May
    assert date(2030, 9, 2) in h    # Labor Day: 1st Monday of September
    assert date(2030, 11, 28) in h  # Thanksgiving: 4th Thursday of November


def test_good_friday_computus():
    # Authoritative Western Easter Sundays; Good Friday is two days earlier.
    assert _easter_sunday(2026) == date(2026, 4, 5)
    assert _easter_sunday(2027) == date(2027, 3, 28)
    assert _easter_sunday(2030) == date(2030, 4, 21)
    assert date(2026, 4, 3) in _holidays(2026)   # Good Friday 2026
    assert date(2030, 4, 19) in _holidays(2030)  # Good Friday 2030


def test_july4_saturday_observed_friday_2026():
    # Jul 4 2026 is a Saturday -> observed Fri Jul 3; the Saturday itself is not a
    # listed closure, and there is NO Jul-3 half-day (it is a FULL closure).
    h = _holidays(2026)
    assert date(2026, 7, 3) in h
    assert date(2026, 7, 4) not in h
    assert date(2026, 7, 3) not in _early_close_days(2026)


def test_independence_day_sunday_observed_monday_2027():
    # Jul 4 2027 is a Sunday -> observed Mon Jul 5.
    assert date(2027, 7, 5) in _holidays(2027)


def test_new_year_saturday_not_observed_2028():
    # NYSE exception: a Saturday New Year's Day is NOT observed (no Dec-31 closure).
    h2028 = _holidays(2028)
    assert date(2028, 1, 1) not in h2028        # Sat, not observed
    assert date(2027, 12, 31) not in _holidays(2027)  # no preceding-Friday closure
    # Sanity: the normal Sunday/Monday observance still applies elsewhere.
    # Jan 1 2023 is a Sunday -> observed Mon Jan 2 2023.
    assert date(2023, 1, 2) in _holidays(2023)


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 19) not in _holidays(2021)
    assert _observed_juneteenth_in(2022)
    assert _observed_juneteenth_in(2026)


def _observed_juneteenth_in(year: int) -> bool:
    # Juneteenth Jun 19 with weekend observance.
    target = date(year, 6, 19)
    if target.weekday() == 5:
        target = date(year, 6, 18)
    elif target.weekday() == 6:
        target = date(year, 6, 20)
    return target in _holidays(year)


def test_christmas_eve_half_day_only_when_open_weekday():
    # 2026: Dec 24 is Thu and not the observed Christmas -> half-day.
    assert date(2026, 12, 24) in _early_close_days(2026)
    # 2027: Dec 25 is Sat -> observed Fri Dec 24 (FULL closure) -> no half-day.
    assert date(2027, 12, 24) not in _early_close_days(2027)
    assert date(2027, 12, 24) in _holidays(2027)


# --------------------------------------------------------------------------- #
# Public is_trading_day + unchanged is_market_open behavior.
# --------------------------------------------------------------------------- #
def test_is_trading_day():
    assert is_trading_day(date(2026, 6, 17)) is True    # Wednesday
    assert is_trading_day(date(2026, 6, 20)) is False   # Saturday
    assert is_trading_day(date(2026, 6, 21)) is False   # Sunday
    assert is_trading_day(date(2026, 6, 19)) is False   # Juneteenth (full holiday)
    # An early-close half-day is STILL a trading day (market is open, shorter session).
    assert is_trading_day(date(2026, 11, 27)) is True   # day after Thanksgiving


def test_is_market_open_unchanged_behavior():
    et = ZoneInfo("America/New_York")
    # Regular session boundaries.
    assert is_market_open(datetime(2026, 6, 17, 11, 0, tzinfo=et)) is True
    assert is_market_open(datetime(2026, 6, 17, 9, 0, tzinfo=et)) is False    # pre-open
    assert is_market_open(datetime(2026, 6, 17, 16, 0, tzinfo=et)) is False   # close exclusive
    assert is_market_open(datetime(2026, 6, 17, 15, 59, tzinfo=et)) is True
    assert is_market_open(datetime(2026, 6, 20, 11, 0, tzinfo=et)) is False   # Saturday
    # Holiday.
    assert is_market_open(datetime(2026, 6, 19, 12, 0)) is False
    # Early close: open before 13:00, closed at/after 13:00.
    assert is_market_open(datetime(2026, 11, 27, 12, 0)) is True
    assert is_market_open(datetime(2026, 11, 27, 14, 0)) is False
