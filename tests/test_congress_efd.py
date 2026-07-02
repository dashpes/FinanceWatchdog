"""Tests for the Senate eFD congress source and its confluence wiring."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

from investment_monitor.analysis.confluence import (
    ConfluenceConfig,
    _amount_midpoint,
    detect_confluence,
    gather_congress_evidence,
)
from investment_monitor.collectors.congress import CongressTradesCollector
from investment_monitor.config import Settings
from investment_monitor.storage import (
    CongressionalTrade,
    Price,
    get_session,
    init_db,
)

TODAY = date.today()

_PTR_HTML = """
<table>
<tr><th>#</th></tr>
<tr>
<td>1</td><td>06/02/2026</td><td>Joint</td>
<td><a href="https://x">NVDA</a></td>
<td>NVIDIA Corp</td><td>Stock</td><td>Purchase</td>
<td>$100,001 - $250,000</td><td>--</td>
</tr>
<tr>
<td>2</td><td>06/03/2026</td><td>Self</td>
<td>--</td>
<td>SOME MUNI BOND</td><td>Corporate Bond</td><td>Sale (Full)</td>
<td>$1,001 - $15,000</td><td>--</td>
</tr>
</table>
"""


def _collector(tmp_path):
    init_db(tmp_path / "t.db")
    session = get_session().__enter__()
    return CongressTradesCollector(session, Settings())


def test_amount_midpoint():
    assert _amount_midpoint("$100,001 - $250,000") == 175000.5
    assert _amount_midpoint("$1,001 - $15,000") == 8000.5
    assert _amount_midpoint("Over $50,000,000") == 50_000_000.0
    assert _amount_midpoint("") is None


def test_parse_efd_ptr_page(tmp_path):
    c = _collector(tmp_path)
    client = MagicMock()
    resp = MagicMock()
    resp.text = _PTR_HTML
    resp.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=resp)
    rows = asyncio.run(c._parse_efd_ptr_page(
        client, "https://efdsearch.senate.gov/search/view/ptr/x/", "John Doe", "06/30/2026"
    ))
    assert len(rows) == 2
    stock = rows[0]
    assert stock["ticker"] == "NVDA" and stock["type"] == "Purchase"
    assert stock["amount"] == "$100,001 - $250,000"
    assert stock["senator"] == "John Doe"
    # And parse_trade maps the eFD shape (keeping the PTR source URL).
    trade = c.parse_trade(stock, "Senate")
    assert trade.ticker == "NVDA" and trade.trade_type == "buy"
    assert trade.trade_date == date(2026, 6, 2)
    assert trade.source_url.endswith("/ptr/x/")
    # The bond row has ticker '--' -> dropped by the junk filter.
    assert c.parse_trade(rows[1], "Senate") is None


def _seed_congress(s, ticker, politician, *, days_ago=3, amount="$100,001 - $250,000",
                   trade_type="buy"):
    s.add(CongressionalTrade(
        ticker=ticker, politician=politician, chamber="Senate", trade_type=trade_type,
        amount_range=amount, trade_date=TODAY - timedelta(days=days_ago),
    ))


def test_gather_congress_evidence_buys_only(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_congress(s, "NVDA", "Sen A")
        _seed_congress(s, "NVDA", "Sen B", trade_type="sell")
        _seed_congress(s, "--", "Sen C")  # junk ticker
    with get_session() as s:
        ev = gather_congress_evidence(s, 30, TODAY)
    assert len(ev) == 1
    assert ev[0].source == "congress" and ev[0].actor == "Sen A"
    assert ev[0].value == 175000.5


def test_congress_only_cluster_is_a_finding(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        for i, days_ago in enumerate((2, 4, 6)):  # 3 members, spread days
            _seed_congress(s, "CLUS", f"Senator {i}", days_ago=days_ago)
        # Liquidity/price context for the narrative path.
        for i in range(25):
            s.add(Price(ticker="CLUS", date=TODAY - timedelta(days=i), open=10, high=10,
                        low=10, close=10.0, volume=200_000))
    with get_session() as s:
        findings = detect_confluence(s, ConfluenceConfig(), today=TODAY)
        clus = [f for f in findings if f.ticker == "CLUS"]
        assert clus and clus[0].kind == "congress_cluster"
        assert "3 member(s) of Congress bought" in clus[0].narrative


def test_two_members_below_cluster_floor_not_a_finding(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_congress(s, "TWO", "Sen A", days_ago=2)
        _seed_congress(s, "TWO", "Sen B", days_ago=4)
    with get_session() as s:
        findings = detect_confluence(s, ConfluenceConfig(), today=TODAY)
        assert not [f for f in findings if f.ticker == "TWO"]
