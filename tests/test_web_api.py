"""API tests for the dashboard (TestClient against a tmp-path seeded store)."""

from datetime import date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from investment_monitor.config import Settings
from investment_monitor.storage import get_session, init_db
from investment_monitor.storage.learning_models import LearningEvent
from investment_monitor.storage.models import InsiderTransaction, Price
from investment_monitor.storage.robo_models import RoboOrder, RoboRun
from investment_monitor.storage.thesis_models import Thesis


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "portfolio.db"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "robo.yaml").write_text(
        "account_id: TEST\ndry_run: true\n# hand-written comment survives\n"
    )

    init_db(db_path)
    with get_session() as s:
        s.add(RoboRun(
            run_id="run-1", dry_run=True, status="completed", account_id="TEST",
            source="llm", total_value=4000.0, settled_cash=2500.0,
            num_proposed=2, num_accepted=2, num_rejected=0, num_placed=1,
        ))
        s.add(RoboOrder(
            run_id="run-1", symbol="EML", side="buy", order_type="market",
            notional=1.94, source="llm", gate_accepted=True, gate_code="accepted",
            placed=True, simulated=False, status="placed", fill_price=12.30,
            fill_quantity=0.157, fill_status="filled", thesis_id=1,
            rationale="78% conviction — Insider cluster",
        ))
        s.add(RoboOrder(
            run_id="run-1", symbol="GME", side="buy", order_type="market",
            notional=50.0, source="llm", gate_accepted=False, gate_code="blocklisted",
            gate_reason="learned blocklist", placed=False, simulated=False,
        ))
        s.add(Thesis(
            id=1, symbol="EML", narrative="Insider cluster: three insiders bought.",
            conviction=0.78, target_weight=0.1, status="active",
            entry_conditions={}, invalidation_conditions={"price_below": 9.0},
            evidence_refs={}, conviction_history=[
                {"ts": "2026-06-01", "conviction": 0.7, "trigger": "promotion"},
                {"ts": "2026-06-20", "conviction": 0.78, "trigger": "re-eval"},
            ],
        ))
        s.add(LearningEvent(
            kind="thesis_outcome", symbol="EML", as_of_date=date(2026, 6, 25),
            conviction=0.78, realized_return=0.04, direction_correct=1, brier=0.05,
        ))
        s.add(Price(ticker="EML", date=date(2026, 6, 30), open=12.0, high=12.6,
                    low=11.9, close=12.5, volume=100000))
        s.add(InsiderTransaction(
            ticker="EML", filing_date=datetime(2026, 6, 20), trade_date=date(2026, 6, 18),
            owner_name="A. Director", owner_title="CEO", transaction_type="P",
            shares=1000, price_per_share=12.0, total_value=12000.0,
        ))

    settings = Settings(
        db_path=db_path, config_dir=config_dir, data_dir=tmp_path,
        dashboard_token="test-pin", robo_force_dry_run=True,
    )

    # Rebuild the web read-only engine for this tmp DB (module-global otherwise).
    from investment_monitor.web import db as webdb

    webdb.init_read_only(db_path)

    from investment_monitor.web.app import create_app

    app = create_app(settings)

    # Never let tests reach a real broker: account cache serves nothing.
    async def _no_account():
        return {"account": None, "stale": False, "as_of": None}

    app.state.account_cache.get = _no_account
    return TestClient(app)


def _auth(token="test-pin"):
    return {"Authorization": f"Bearer {token}"}


# --- pages & health -------------------------------------------------------------

def test_health(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_overview_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "ARCHIE" in res.text.upper()
    assert "Personal Private Equity" in res.text


@pytest.mark.parametrize("path", [
    "/", "/theses", "/theses/EML", "/trades", "/charts",
    "/signals", "/learning", "/system", "/settings",
])
def test_every_page_renders(client, path):
    res = client.get(path)
    assert res.status_code == 200, path
    assert "Personal Private Equity" in res.text


# --- overview API ------------------------------------------------------------------

def test_overview_falls_back_to_last_run(client):
    data = client.get("/api/overview").json()
    assert data["account"]["source"] == "last_run"
    assert data["account"]["total_value"] == 4000.0
    assert data["account"]["stale"] is True
    assert data["bot"]["last_run"]["run_id"] == "run-1"
    assert data["bot"]["env_force_dry_run"] is True
    # placed order shows; gate-rejected one does not
    symbols = [o["symbol"] for o in data["todays_orders"]]
    assert "GME" not in symbols


def test_equity_curve(client):
    pts = client.get("/api/overview/equity").json()["points"]
    assert len(pts) == 1
    assert pts[0]["total_value"] == 4000.0


# --- theses ---------------------------------------------------------------------------

def test_theses_list_and_detail(client):
    listing = client.get("/api/theses").json()["theses"]
    assert listing[0]["symbol"] == "EML"
    detail = client.get("/api/theses/eml").json()
    assert "three insiders" in detail["narrative"]
    assert detail["invalidation_conditions"] == {"price_below": 9.0}
    assert len(detail["conviction_history"]) == 2
    assert detail["orders"][0]["rationale"].startswith("78%")
    assert detail["learning_events"][0]["brier"] == 0.05
    assert client.get("/api/theses/NOPE").status_code == 404


def test_thesis_monitor(client):
    mon = client.get("/api/theses/EML/monitor").json()
    assert mon["entry"]["fill_price"] == 12.30
    assert mon["latest_close"] == 12.5
    assert abs(mon["return_since_entry"] - (12.5 / 12.30 - 1)) < 1e-9


# --- ledger ------------------------------------------------------------------------------

def test_runs_and_orders(client):
    runs = client.get("/api/runs").json()["runs"]
    assert runs[0]["run_id"] == "run-1"
    detail = client.get("/api/runs/run-1").json()
    assert len(detail["orders"]) == 2
    rejected = [o for o in detail["orders"] if not o["gate_accepted"]]
    assert rejected[0]["gate_code"] == "blocklisted"
    filtered = client.get("/api/orders", params={"symbol": "eml"}).json()["orders"]
    assert len(filtered) == 1


def test_pnl(client):
    data = client.get("/api/pnl").json()
    assert data["per_symbol"][0]["symbol"] == "EML"


# --- signals / learning / charts -------------------------------------------------------------

def test_insiders(client):
    txs = client.get("/api/signals/insiders", params={"ticker": "EML"}).json()["transactions"]
    assert txs[0]["owner_name"] == "A. Director"


def test_learning_summary(client):
    data = client.get("/api/learning/summary").json()
    assert data["totals"]["n_outcomes"] == 1
    assert data["totals"]["win_rate"] == 1.0
    assert data["per_symbol"][0]["symbol"] == "EML"


def test_chart_price_with_trades(client):
    data = client.get("/api/charts/price/EML").json()
    assert data["candles"][0]["close"] == 12.5
    assert data["trades"][0]["fill_price"] == 12.30
    assert client.get("/api/charts/symbols").json()["symbols"] == ["EML", "GME"]


# --- system ------------------------------------------------------------------------------------

def test_system(client):
    data = client.get("/api/system").json()
    assert data["db_size_bytes"] > 0
    assert data["control"]["trading_paused"] is False
    assert data["env_force_dry_run"] is True


# --- auth: mutations gated -----------------------------------------------------------------------

def test_mutations_require_token(client):
    assert client.post("/api/control/pause", json={"reason": "x"}).status_code == 401
    assert client.post(
        "/api/control/pause", json={"reason": "x"}, headers=_auth("wrong")
    ).status_code == 401


def test_pause_resume_roundtrip(client):
    res = client.post("/api/control/pause", json={"reason": "hols"}, headers=_auth())
    assert res.status_code == 200 and res.json()["trading_paused"] is True
    res = client.post("/api/control/resume", headers=_auth())
    assert res.json()["trading_paused"] is False


def test_kill_switch_one_way(client):
    res = client.post("/api/control/kill", json={"reason": ""}, headers=_auth())
    assert res.json()["force_dry_run"] is True
    res = client.post("/api/control/unkill", headers=_auth())
    data = res.json()
    assert data["force_dry_run"] is False
    assert data["env_force_dry_run"] is True  # env layer still forces paper


def test_blocklist_management(client):
    res = client.post("/api/blocklist", json={"symbol": "gme", "reason": "meme"}, headers=_auth())
    assert "GME" in res.json()["learned"]
    res = client.delete("/api/blocklist/GME", headers=_auth())
    assert "GME" not in res.json()["learned"]
    assert client.delete("/api/blocklist/GME", headers=_auth()).status_code == 404


# --- settings ---------------------------------------------------------------------------------------

def test_settings_catalog(client):
    data = client.get("/api/settings").json()
    keys = {s["key"] for s in data["settings"]}
    assert "dry_run" in keys
    dry = next(s for s in data["settings"] if s["key"] == "dry_run")
    assert dry["current"] is True
    assert dry["safety"] is True


def test_settings_write_preserves_comments(client, tmp_path):
    res = client.put(
        "/api/settings/caps.max_positions", json={"value": 7}, headers=_auth()
    )
    assert res.status_code == 200, res.text
    data = client.get("/api/settings").json()
    cap = next(s for s in data["settings"] if s["key"] == "caps.max_positions")
    assert cap["current"] == 7


def test_settings_safety_key_needs_confirm(client):
    res = client.put("/api/settings/mode", json={"value": "thesis"}, headers=_auth())
    assert res.status_code == 428
    # dry_run: false is refused outright, even confirmed
    res = client.put(
        "/api/settings/dry_run", json={"value": False, "confirm": True}, headers=_auth()
    )
    assert res.status_code == 403


def test_settings_validation_error_is_422(client):
    res = client.put(
        "/api/settings/caps.max_positions", json={"value": "not-a-number"}, headers=_auth()
    )
    assert res.status_code == 422
