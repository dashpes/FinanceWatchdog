"""Tests for the SEC 8-K material-events collector and its confluence wiring."""

from __future__ import annotations

from datetime import date, timedelta

from investment_monitor.analysis.confluence import (
    Evidence,
    gather_filing_evidence,
    score_confluence,
)
from investment_monitor.collectors.material_events import (
    MaterialEventsCollector,
    map_item_descriptions,
    parse_sgml_header,
)
from investment_monitor.config import Settings
from investment_monitor.storage import (
    MaterialEvent,
    get_material_events,
    get_session,
    init_db,
)

TODAY = date.today()

_HEADER = """<SEC-DOCUMENT>0001104659-26-012345.txt : 20260630
<SEC-HEADER>0001104659-26-012345.hdr.sgml : 20260630
ACCESSION NUMBER:\t\t0001104659-26-012345
CONFORMED SUBMISSION TYPE:\t8-K
PUBLIC DOCUMENT COUNT:\t\t3
CONFORMED PERIOD OF REPORT:\t20260629
ITEM INFORMATION:\t\tDeparture of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers: Compensatory Arrangements of Certain Officers
ITEM INFORMATION:\t\tFinancial Statements and Exhibits
FILED AS OF DATE:\t\t20260630
FILER:
\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tACME WIDGETS INC
\t\tCENTRAL INDEX KEY:\t\t\t0000320193
</SEC-HEADER>
"""

_IDX = """Form Type   Company Name   CIK   Date Filed   File Name
---------------------------------------------------------------
4           SOMEONE        111   2026-06-30   edgar/data/111/0001-26-000001.txt
8-K         ACME WIDGETS   320193 2026-06-30  edgar/data/320193/0001104659-26-012345.txt
8-K/A       OTHER CO       222   2026-06-30   edgar/data/222/0001-26-000002.txt
10-K        BIG CO         333   2026-06-30   edgar/data/333/0001-26-000003.txt
"""


def test_parse_sgml_header():
    h = parse_sgml_header(_HEADER)
    assert h["cik"] == "320193"
    assert h["company_name"] == "ACME WIDGETS INC"
    assert h["filed_date"] == date(2026, 6, 30)
    assert len(h["item_descriptions"]) == 2


def test_map_item_descriptions():
    h = parse_sgml_header(_HEADER)
    assert map_item_descriptions(h["item_descriptions"]) == ["5.02", "9.01"]
    assert map_item_descriptions(["Regulation FD Disclosure"]) == ["7.01"]
    assert map_item_descriptions(["Something unrecognized"]) == []


def test_parse_index_filters_to_exact_8k(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        collector = MaterialEventsCollector(s, Settings())
        urls = collector._parse_index_for_8k(_IDX)
    assert urls == [
        "https://www.sec.gov/Archives/edgar/data/320193/0001104659-26-012345.txt"
    ]


def test_material_event_roundtrip_and_query(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        s.add(MaterialEvent(
            ticker="ACME", cik="320193", company_name="ACME WIDGETS INC",
            items=["5.02", "9.01"], filed_date=TODAY - timedelta(days=2),
            sec_url="https://sec.gov/a.txt",
        ))
        s.add(MaterialEvent(
            ticker="ACME", cik="320193", items=["2.02"],
            filed_date=TODAY - timedelta(days=200), sec_url="https://sec.gov/old.txt",
        ))
    with get_session() as s:
        recent = get_material_events(s, "ACME", days=30)
        assert len(recent) == 1 and recent[0].items == ["5.02", "9.01"]


def test_filing_evidence_requires_signal_item(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        s.add(MaterialEvent(  # high-signal: exec departure
            ticker="SIG", cik="1", items=["5.02", "9.01"],
            filed_date=TODAY - timedelta(days=3), sec_url="https://sec.gov/1.txt",
        ))
        s.add(MaterialEvent(  # routine: earnings release + exhibits only
            ticker="ROUT", cik="2", items=["2.02", "9.01"],
            filed_date=TODAY - timedelta(days=3), sec_url="https://sec.gov/2.txt",
        ))
    with get_session() as s:
        ev = gather_filing_evidence(s, {"SIG", "ROUT"}, TODAY)
    assert [e.ticker for e in ev] == ["SIG"]
    assert ev[0].source == "filing" and "5.02" in ev[0].actor


def test_filing_counts_as_strong_source_in_scoring():
    insiders = [
        Evidence("X", "insider", f"Insider {i}", TODAY - timedelta(days=i + 1),
                 50_000.0, "buy")
        for i in range(3)
    ]
    filing = [Evidence("X", "filing", "8-K 5.02", TODAY - timedelta(days=2), None, "8-K")]
    base = score_confluence(insiders, today=TODAY)
    combined = score_confluence(insiders + filing, today=TODAY)
    assert combined["n_strong"] == 2          # insider + filing, both strong
    assert combined["score"] > base["score"]  # cross-source corroboration is super-additive
