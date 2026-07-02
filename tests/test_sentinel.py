"""Tests for the intraday sentinel (flag/invalidate only, never buys)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch

from investment_monitor.config import Settings
from investment_monitor.robo.config import RoboConfig
from investment_monitor.robo.sentinel import run_sentinel
from investment_monitor.storage import (
    MaterialEvent,
    Price,
    Thesis,
    ThesisStatus,
    get_active_theses,
    get_session,
    get_thesis,
    init_db,
)

TODAY = date.today()
# A guaranteed-open moment: Wednesday 2026-07-01 was a full trading day, 11:00 ET.
OPEN_NOW = datetime(2026, 7, 1, 11, 0)
CLOSED_NOW = datetime(2026, 7, 4, 11, 0)  # Saturday


def _cfg() -> RoboConfig:
    return RoboConfig(mode="autonomous", dry_run=True)


def _seed_thesis(s, symbol, *, entry_price=10.0, drop_pct=25):
    s.add(Thesis(
        symbol=symbol, narrative=f"{symbol} thesis", conviction=0.6,
        entry_conditions={"entry_price": entry_price},
        invalidation_conditions={"price_drop_pct": drop_pct},
        status=ThesisStatus.ACTIVE.value,
    ))


def _seed_price(s, symbol, close):
    s.add(Price(ticker=symbol, date=TODAY, open=close, high=close, low=close,
                close=close, volume=100000))


def _run(settings, cfg, now):
    # Stub the quote refresh so tests never hit yfinance; stored prices decide.
    with patch(
        "investment_monitor.robo.sentinel.PriceCollector.collect",
        new_callable=AsyncMock,
    ), patch("investment_monitor.robo.sentinel._notify") as notify:
        return run_sentinel(settings, cfg, now=now), notify


def test_noop_when_market_closed(tmp_path):
    init_db(tmp_path / "t.db")
    out, _ = _run(Settings(), _cfg(), CLOSED_NOW)
    assert out["status"] == "market_closed" and out["checked"] == 0


def test_trips_price_drop_invalidation(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_thesis(s, "DROP", entry_price=10.0, drop_pct=25)
        _seed_price(s, "DROP", close=7.0)   # -30% from entry
        _seed_thesis(s, "FINE", entry_price=10.0, drop_pct=25)
        _seed_price(s, "FINE", close=9.5)   # -5%: fine
    out, notify = _run(Settings(), _cfg(), OPEN_NOW)
    assert out["checked"] == 2
    assert len(out["tripped"]) == 1 and out["tripped"][0].startswith("DROP:")
    assert notify.called
    with get_session() as s:
        assert get_thesis(s, "DROP").status == ThesisStatus.INVALIDATED.value
        assert get_thesis(s, "DROP").conviction == 0.0
        assert {t.symbol for t in get_active_theses(s)} == {"FINE"}


def test_flags_material_8k_once(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_thesis(s, "EVNT", entry_price=10.0)
        _seed_price(s, "EVNT", close=10.0)
        s.add(MaterialEvent(
            ticker="EVNT", cik="1", items=["5.02"],
            filed_date=TODAY - timedelta(days=1), sec_url="https://sec.gov/e.txt",
        ))
    out1, _ = _run(Settings(), _cfg(), OPEN_NOW)
    assert len(out1["flagged"]) == 1 and "5.02" in out1["flagged"][0]
    # Second pass: the same filing must not re-alert (dedup key).
    out2, notify2 = _run(Settings(), _cfg(), OPEN_NOW)
    assert out2["flagged"] == [] and not notify2.called
    # Thesis untouched — a flag is information, not an action.
    with get_session() as s:
        assert get_thesis(s, "EVNT").status == ThesisStatus.ACTIVE.value


def test_routine_8k_not_flagged(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        _seed_thesis(s, "ERN", entry_price=10.0)
        _seed_price(s, "ERN", close=10.0)
        s.add(MaterialEvent(  # earnings release + exhibits: routine
            ticker="ERN", cik="2", items=["2.02", "9.01"],
            filed_date=TODAY, sec_url="https://sec.gov/r.txt",
        ))
    out, notify = _run(Settings(), _cfg(), OPEN_NOW)
    assert out["flagged"] == [] and out["tripped"] == [] and not notify.called
