"""Tests for the Form 4 junk-ticker filter in _parse_form4.

Some Form 4 filings carry placeholder issuer symbols ('NONE'/'N/A'/'NA'/'--'/'')
instead of a real ticker. Those must be skipped so junk symbols never reach the DB
or the confluence engine.
"""

from __future__ import annotations

import pytest

from investment_monitor.collectors.insider import InsiderCollector

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
