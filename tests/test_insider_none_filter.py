"""Tests for the Form 4 junk-ticker filter in _parse_form4.

Some Form 4 filings carry placeholder issuer symbols ('NONE'/'N/A'/'NA'/'--'/'')
instead of a real ticker. Those must be skipped so junk symbols never reach the DB
or the confluence engine.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from investment_monitor.collectors.insider import (
    InsiderCollector,
    JUNK_TICKERS,
    is_junk_ticker,
)

# A minimal, well-formed Form 4 ownership document with a real symbol.
_FORM4_TEMPLATE = """<ownershipDocument>
  <issuer><issuerTradingSymbol>{symbol}</issuerTradingSymbol></issuer>
  <reportingOwner><reportingOwnerId><rptOwnerName>Jane Insider</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2026-06-16</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>500</value></transactionShares>
      <transactionPricePerShare><value>42.00</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


def _parser():
    """A bare collector instance (no session/network) to call _parse_form4 directly."""
    return InsiderCollector.__new__(InsiderCollector)


@pytest.mark.parametrize("junk", ["NONE", "none", "N/A", "n/a", "NA", "na", "", "  ", "--"])
def test_form4_junk_symbol_skips_filing(junk):
    # Junk placeholder symbols (case-insensitive, after strip) yield NO transactions.
    xml = _FORM4_TEMPLATE.format(symbol=junk)
    assert _parser()._parse_form4(xml, None, "http://x#junk") == []


def test_form4_normal_symbol_still_parses():
    # A genuine symbol still produces a transaction attributed to that ticker.
    xml = _FORM4_TEMPLATE.format(symbol="NVDA")
    (txn,) = _parser()._parse_form4(xml, None, "http://x#ok")
    assert txn.ticker == "NVDA"
    assert txn.shares == 500 and txn.transaction_type == "P"


# --------------------------------------------------------------------------- #
# #15 shared junk-ticker helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("junk", ["", "  ", "NONE", "none", "N/A", "n/a", "NA", "na",
                                  "--", "N\\A", None])
def test_is_junk_ticker_true_for_placeholders(junk):
    assert is_junk_ticker(junk) is True


@pytest.mark.parametrize("real", ["AAPL", "nvda", " tsla ", "BRK", "A"])
def test_is_junk_ticker_false_for_real_symbols(real):
    assert is_junk_ticker(real) is False


def test_junk_tickers_keeps_union_of_historical_entries():
    # Nothing previously filtered by insider/congress/confluence may slip through.
    for sym in ("", "NONE", "N/A", "NA", "--", "N\\A"):
        assert sym in JUNK_TICKERS


# --------------------------------------------------------------------------- #
# #6 weekend / holiday / early-morning gap in _recent_business_dates
# --------------------------------------------------------------------------- #
def _dates_on(today: date, days_back: int = 1) -> list[date]:
    c = InsiderCollector.__new__(InsiderCollector)
    with patch("investment_monitor.collectors.insider.date") as md:
        md.today.return_value = today
        md.side_effect = lambda *a, **k: date(*a, **k)
        return c._recent_business_dates(days_back)


def test_weekend_run_bridges_back_to_last_trading_day():
    # Saturday 2026-06-20 / Sunday 2026-06-21: a days_back=1 run must NOT be empty;
    # it must cover the most recent TRADING day so no trading-day's Form 4s are
    # silently skipped. Fri 2026-06-19 is Juneteenth (a NYSE holiday), so the last
    # open day is Thu 06-18 — the bridge must reach it and must NOT include the
    # holiday.
    for weekend_day in (date(2026, 6, 20), date(2026, 6, 21)):
        out = _dates_on(weekend_day)
        assert out, "weekend run produced no business dates"
        assert date(2026, 6, 18) in out          # last trading day bridged in
        assert date(2026, 6, 19) not in out       # Juneteenth holiday excluded
        assert all(d.weekday() < 5 for d in out)  # only weekdays


def test_monday_run_picks_up_prior_trading_days_late_filings():
    # Monday 2026-06-22 with days_back=1 must reach the prior TRADING day so Form 4s
    # filed late then (and any not-yet-posted early-morning index) are ingested.
    # Fri 06-19 is Juneteenth (holiday), so the prior trading day is Thu 06-18: the
    # bridge must span the holiday gap and reach it, NOT stop at the holiday.
    out = _dates_on(date(2026, 6, 22))
    assert date(2026, 6, 22) in out   # Monday itself
    assert date(2026, 6, 19) not in out  # Juneteenth holiday is NOT a trading day
    assert date(2026, 6, 18) in out   # prior trading day bridged in across holiday


def test_tuesday_after_monday_holiday_includes_prior_friday():
    # REGRESSION (#6 residual holiday gap): a Tuesday run after a Monday market
    # holiday must still cover the immediately-preceding Friday, otherwise that
    # Friday's late-filed Form 4s are silently skipped. Mon 2026-01-19 is MLK Day
    # (a NYSE holiday); a days_back=1 run on Tue 01-20 must reach Fri 01-16 and must
    # NOT include the Monday holiday. A fixed +1-weekday bridge fails this (it stops
    # at the Monday holiday); spanning real trading days fixes it.
    out = _dates_on(date(2026, 1, 20))
    assert date(2026, 1, 20) in out      # Tuesday itself
    assert date(2026, 1, 19) not in out  # MLK Monday holiday excluded
    assert date(2026, 1, 16) in out      # prior Friday's filings still covered
    assert all(d.weekday() < 5 for d in out)


def test_weekday_run_covers_today_and_prior_business_day():
    out = _dates_on(date(2026, 6, 24))  # Wednesday
    assert out[0] == date(2026, 6, 24)        # newest first
    assert date(2026, 6, 23) in out           # Tuesday bridge
    assert all(d.weekday() < 5 for d in out)
