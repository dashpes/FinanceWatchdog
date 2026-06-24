"""Tests for broad, universe-independent multi-source collection."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from investment_monitor.collectors.congress import CongressTradesCollector
from investment_monitor.collectors.insider import InsiderCollector
from investment_monitor.config import Settings
from investment_monitor.storage import (
    CongressionalTrade,
    InsiderTransaction,
    get_session,
    init_db,
)

# Three trades in names that are NOT in any portfolio — the whole point of "broad".
HOUSE = [
    {"representative": "Rep A", "ticker": "NVDA", "type": "purchase",
     "amount": "$1,001 - $15,000", "transaction_date": "2026-06-01", "party": "D"},
    {"representative": "Rep B", "ticker": "SMCI", "type": "sale",
     "amount": "$15,001 - $50,000", "transaction_date": "2026-05-15"},
]
SENATE = [
    {"senator": "Sen C", "ticker": "TSLA", "type": "purchase",
     "amount": "$1,001 - $15,000", "transaction_date": "2026-06-10"},
]


def _settings(tmp_path, db):
    return Settings(public_api_token="t", config_dir=tmp_path, data_dir=tmp_path,
                    log_dir=tmp_path, db_path=db)


def _collector(tmp_path, db, *, house=HOUSE, senate=SENATE):
    """A collector wired to a fresh session with the feeds mocked (no network)."""
    init_db(db)
    session = get_session().__enter__()  # plain session; collect_all commits itself
    c = CongressTradesCollector(session, _settings(tmp_path, db))
    c.fetch_house_trades = AsyncMock(return_value=house)
    c.fetch_senate_trades = AsyncMock(return_value=senate)
    return c


def _count(db):
    init_db(db)
    with get_session() as s:
        return s.scalar(select(func.count()).select_from(CongressionalTrade))


def _tickers(db):
    init_db(db)
    with get_session() as s:
        return {t.ticker for t in s.scalars(select(CongressionalTrade))}


@pytest.mark.asyncio
async def test_collect_all_retains_market_wide(tmp_path):
    db = tmp_path / "t.db"
    result = await _collector(tmp_path, db).collect_all()
    assert result.success and result.records_collected == 3
    # All three retained, even though none belong to any configured universe.
    assert _tickers(db) == {"NVDA", "SMCI", "TSLA"}


@pytest.mark.asyncio
async def test_collect_all_dedups_on_second_run(tmp_path):
    db = tmp_path / "t.db"
    await _collector(tmp_path, db).collect_all()
    result = await _collector(tmp_path, db).collect_all()  # same feed again
    assert result.records_collected == 0  # nothing new
    assert _count(db) == 3


@pytest.mark.asyncio
async def test_collect_all_since_filter_bounds_volume(tmp_path):
    db = tmp_path / "t.db"
    result = await _collector(tmp_path, db).collect_all(since=date(2026, 6, 1))
    assert result.records_collected == 2  # SMCI (2026-05-15) excluded
    assert _tickers(db) == {"NVDA", "TSLA"}


@pytest.mark.asyncio
async def test_collect_all_survives_one_chamber_failing(tmp_path):
    db = tmp_path / "t.db"
    c = _collector(tmp_path, db)
    c.fetch_senate_trades = AsyncMock(side_effect=RuntimeError("S3 down"))
    result = await c.collect_all()
    assert result.records_collected == 2          # House still retained
    assert not result.success and result.errors   # Senate failure surfaced
    assert _tickers(db) == {"NVDA", "SMCI"}


# --------------------------------------------------------------------------- #
# Broad SEC insider (Form 4) via EDGAR daily index
# --------------------------------------------------------------------------- #
_FORM4_NVDA = """<SEC-DOCUMENT>
<DOCUMENT><TYPE>4<TEXT><XML>
<ownershipDocument>
  <issuer><issuerTradingSymbol>NVDA</issuerTradingSymbol></issuer>
  <reportingOwner><reportingOwnerId><rptOwnerName>Jensen Huang</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2026-06-16</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>1000</value></transactionShares>
      <transactionPricePerShare><value>120.50</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>
</XML></TEXT></DOCUMENT>"""

_DAILY_INDEX = """Description: Daily Index of Form Types
Form Type    Company Name      CIK       Date Filed    File Name
-------------------------------------------------------------------------
3            SOME CO           111       2026-06-17    edgar/data/111/x.txt
4            NVIDIA CORP       1045810   2026-06-17    edgar/data/1045810/0001045810-26-000050.txt
4/A          OLD AMEND         222       2026-06-17    edgar/data/222/y.txt
"""


def _insider(tmp_path, db):
    """An insider collector with a fixed index date and mocked HTTP (no network)."""
    init_db(db)
    session = get_session().__enter__()
    c = InsiderCollector(session, _settings(tmp_path, db))
    c._recent_business_dates = lambda days_back: [date(2026, 6, 17)]

    async def fake_get(client, url):
        return _DAILY_INDEX if "daily-index" in url else _FORM4_NVDA

    c._get = fake_get
    return c


@pytest.mark.asyncio
async def test_broad_insider_parses_form4_market_wide(tmp_path):
    db = tmp_path / "i.db"
    result = await _insider(tmp_path, db).collect_all(days_back=1)
    assert result.success and result.records_collected == 1
    init_db(db)
    with get_session() as s:
        txns = list(s.scalars(select(InsiderTransaction)))
        assert len(txns) == 1
        t = txns[0]
        assert t.ticker == "NVDA"      # derived from the FILING, not a configured universe
        assert t.transaction_type == "P" and t.shares == 1000
        assert t.raw_code == "P"       # genuine open-market purchase preserved
        assert abs((t.price_per_share or 0) - 120.50) < 1e-9


def test_form4_parser_preserves_raw_code():
    # A grant (code A) collapses to transaction_type 'P' but must stay distinguishable
    # via raw_code — else the confluence engine reads mass RSU grants as cluster-buying.
    c = InsiderCollector.__new__(InsiderCollector)
    award_xml = _FORM4_NVDA.replace(
        "<transactionCode>P</transactionCode>", "<transactionCode>A</transactionCode>"
    )
    (grant,) = c._parse_form4(award_xml, None, "http://x#a")
    assert grant.raw_code == "A" and grant.transaction_type == "P"
    (buy,) = c._parse_form4(_FORM4_NVDA, None, "http://x#b")
    assert buy.raw_code == "P" and buy.transaction_type == "P"


def test_broad_insider_index_parse_filters_to_form4(tmp_path):
    # Form 3 and 4/A lines are dropped; only the exact Form 4 yields a filing URL.
    c = InsiderCollector.__new__(InsiderCollector)
    urls = c._parse_index_for_form4(_DAILY_INDEX)
    assert urls == ["https://www.sec.gov/Archives/edgar/data/1045810/0001045810-26-000050.txt"]


@pytest.mark.asyncio
async def test_broad_insider_dedups_across_runs(tmp_path):
    db = tmp_path / "i.db"
    await _insider(tmp_path, db).collect_all(days_back=1)
    result = await _insider(tmp_path, db).collect_all(days_back=1)  # same filing again
    assert result.records_collected == 0
    init_db(db)
    with get_session() as s:
        assert s.scalar(select(func.count()).select_from(InsiderTransaction)) == 1
