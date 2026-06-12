"""Tests for the local web dashboard (FastAPI)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from investment_monitor.config import Settings
from investment_monitor.web.app import create_app


@pytest.fixture
def settings(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    return Settings(
        config_dir=cfg,
        db_path=tmp_path / "data.db",
        ollama_model="qwen2.5:7b",
        ollama_synthesis_model="qwen2.5:32b",
        llm_provider="ollama",
    )


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Investment Monitor" in r.text


def test_status_payload(client):
    with patch(
        "investment_monitor.web.app._probe_ollama",
        return_value=(True, ["qwen2.5:7b"], None),
    ):
        r = client.get("/api/status")
    assert r.status_code == 200
    s = r.json()
    assert s["fast_model"] == "qwen2.5:7b"
    assert s["synthesis_model"] == "qwen2.5:32b"
    assert s["provider"] == "ollama"
    assert s["ollama_reachable"] is True
    assert s["fast_model_ready"] is True
    assert s["synthesis_model_ready"] is False  # 32b not installed


def test_get_portfolio_empty(client):
    r = client.get("/api/portfolio")
    assert r.status_code == 200
    assert r.json() == {"holdings": [], "watchlist": []}


def test_put_then_get_portfolio_roundtrip(client, settings):
    payload = {
        "holdings": [{"ticker": "AAPL", "shares": 10, "cost_basis": 165.0, "thesis": "x"}],
        "watchlist": [{"ticker": "GOOGL", "target_price": 140.0}],
    }
    r = client.put("/api/portfolio", json=payload)
    assert r.status_code == 200
    # Persisted to disk.
    assert (settings.config_dir / "portfolio.yaml").exists()
    # And readable back.
    got = client.get("/api/portfolio").json()
    assert got["holdings"][0]["ticker"] == "AAPL"
    assert got["watchlist"][0]["ticker"] == "GOOGL"


def test_put_portfolio_invalid_returns_422(client):
    # shares must be > 0; ticker pattern enforced by the model.
    r = client.put("/api/portfolio", json={"holdings": [{"ticker": "toolongticker", "shares": -1, "cost_basis": 1}]})
    assert r.status_code == 422


def test_alerts_empty_when_no_db(client):
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert r.json() == []


def test_run_invalid_type_returns_400(client):
    r = client.post("/api/run", json={"type": "nope"})
    assert r.status_code == 400


def test_run_triggers_and_completes(client):
    fake_summary = "RunSummary(regular, ok)"
    with patch(
        "investment_monitor.web.app.run_monitor_sync", return_value=fake_summary
    ) as mock_run:
        r = client.post("/api/run", json={"type": "regular"})
        assert r.status_code == 202
        assert r.json()["status"] == "started"

        # Poll the background job to completion.
        for _ in range(50):
            state = client.get("/api/run").json()
            if state["status"] in ("done", "error"):
                break
            time.sleep(0.05)

    assert state["status"] == "done"
    assert state["run_type"] == "regular"
    assert fake_summary in state["summary"]
    mock_run.assert_called_once()
