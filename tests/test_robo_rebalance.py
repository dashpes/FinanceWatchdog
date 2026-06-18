"""Integration tests for the rebalance orchestration using a fake broker.

No network and no real money — a FakeBroker duck-types the broker interface so we
can exercise the full pipeline, including the dry-run simulate path, the live
place path, and the hard margin-account refusal.
"""

from __future__ import annotations

from decimal import Decimal

from investment_monitor.config import Settings
from investment_monitor.robo.broker import PlacedOrder, PreflightResult
from investment_monitor.robo.config import RoboCaps, RoboConfig
from investment_monitor.robo.models import AccountState, Position
from investment_monitor.robo.rebalance import rebalance_run
from investment_monitor.storage import get_recent_robo_runs, get_session, init_db


class FakeBroker:
    """Duck-typed stand-in for PublicBroker."""

    def __init__(self, account: AccountState, *, dry_run: bool = True):
        self._account = account
        self.dry_run = dry_run
        self.place_called = 0
        self.preflight_called = 0

    def get_account_state(self) -> AccountState:
        return self._account

    def get_quotes(self, symbols):
        return {"VOO": Decimal("500"), "SCHD": Decimal("80")}

    def preflight(self, order):
        self.preflight_called += 1
        return PreflightResult(ok=True, estimated_cost=order.notional or Decimal("0"),
                               margin_requirement=Decimal("0"))

    def place_order(self, order):
        self.place_called += 1
        return PlacedOrder(order_id="fake-123", status="NEW", simulated=False)


def cash_account(cash="100", positions=None) -> AccountState:
    return AccountState(
        account_id="ACC1", account_type="BROKERAGE", is_cash_account=True,
        has_margin=False, settled_cash=Decimal(cash), positions=positions or [],
    )


def margin_account() -> AccountState:
    return AccountState(
        account_id="ACC2", account_type="BROKERAGE", is_cash_account=False,
        has_margin=True, settled_cash=Decimal("100"), positions=[],
    )


def make_settings(tmp_path, *, force_dry_run=True) -> Settings:
    return Settings(
        public_api_token="test-token",
        robo_force_dry_run=force_dry_run,
        config_dir=tmp_path,
        data_dir=tmp_path,
        log_dir=tmp_path,
        db_path=tmp_path / "test.db",
    )


def make_config(*, dry_run=True, require_market_hours=False) -> RoboConfig:
    return RoboConfig(
        target_allocation={"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2},
        allowlist=["VOO", "SCHD"],
        use_llm=False,  # deterministic path; no Ollama needed
        dry_run=dry_run,
        # Default off so placement tests are deterministic regardless of wall-clock.
        require_market_hours=require_market_hours,
        caps=RoboCaps(max_order_pct=0.25, max_orders_per_run=5, max_orders_per_day=10),
    )


def test_dry_run_simulates_and_places_nothing(tmp_path):
    broker = FakeBroker(cash_account("100"))
    result = rebalance_run(make_config(), make_settings(tmp_path), broker=broker)

    assert result.status == "completed"
    assert result.dry_run is True
    assert result.num_proposed == 2  # VOO + SCHD buys
    assert result.num_accepted == 2
    assert result.num_rejected == 0
    assert result.num_placed == 2  # simulated
    assert broker.place_called == 0  # never touches the real place endpoint
    assert broker.preflight_called == 2

    # Audit log was written.
    audit_file = tmp_path / "robo_audit.jsonl"
    assert audit_file.exists()
    assert "run_summary" in audit_file.read_text()

    # Run + orders persisted.
    init_db(tmp_path / "test.db")
    with get_session() as session:
        runs = get_recent_robo_runs(session, limit=5)
        assert len(runs) == 1
        assert runs[0].num_placed == 2


def test_refuses_margin_account(tmp_path):
    broker = FakeBroker(margin_account())
    result = rebalance_run(make_config(), make_settings(tmp_path), broker=broker)

    assert result.status == "refused"
    assert result.num_placed == 0
    assert broker.place_called == 0
    audit_file = tmp_path / "robo_audit.jsonl"
    assert "safety_check" in audit_file.read_text()


def test_live_path_places_orders_when_fully_enabled(tmp_path):
    # Live requires: env kill-switch off, config dry_run off, no override.
    broker = FakeBroker(cash_account("100"), dry_run=False)
    settings = make_settings(tmp_path, force_dry_run=False)
    result = rebalance_run(make_config(dry_run=False), settings, broker=broker)

    assert result.dry_run is False
    assert result.num_placed == 2
    assert broker.place_called == 2


def test_env_kill_switch_forces_dry_run_even_if_config_live(tmp_path):
    # config says live, but ROBO_FORCE_DRY_RUN=true must win.
    broker = FakeBroker(cash_account("100"))
    settings = make_settings(tmp_path, force_dry_run=True)
    result = rebalance_run(make_config(dry_run=False), settings, broker=broker)

    assert result.dry_run is True
    assert broker.place_called == 0


def test_live_run_defers_placement_when_market_closed(tmp_path, monkeypatch):
    # Live + require_market_hours, but market closed -> orders are gate-accepted and
    # preflighted, yet NOT placed (deferred), so nothing trades off-hours.
    import investment_monitor.robo.rebalance as reb
    monkeypatch.setattr(reb, "is_market_open", lambda *a, **k: False)
    broker = FakeBroker(cash_account("100"), dry_run=False)
    settings = make_settings(tmp_path, force_dry_run=False)
    cfg = make_config(dry_run=False, require_market_hours=True)
    result = reb.rebalance_run(cfg, settings, broker=broker)
    assert result.dry_run is False
    assert result.num_accepted == 2
    assert result.num_placed == 0          # deferred
    assert broker.place_called == 0        # nothing sent to the broker


def test_live_run_places_when_market_open(tmp_path, monkeypatch):
    import investment_monitor.robo.rebalance as reb
    monkeypatch.setattr(reb, "is_market_open", lambda *a, **k: True)
    broker = FakeBroker(cash_account("100"), dry_run=False)
    settings = make_settings(tmp_path, force_dry_run=False)
    cfg = make_config(dry_run=False, require_market_hours=True)
    result = reb.rebalance_run(cfg, settings, broker=broker)
    assert result.num_placed == 2 and broker.place_called == 2


def test_per_day_cap_counts_only_real_placements(tmp_path):
    # Simulated (dry-run) and deferred orders must NOT count toward the per-day cap,
    # so a day of paper runs never exhausts the live order budget.
    from investment_monitor.storage import RoboOrder, count_placed_orders_today, save_robo_order
    init_db(tmp_path / "c.db")
    with get_session() as s:
        save_robo_order(s, RoboOrder(run_id="r1", symbol="VOO", side="buy",
                                     order_type="market", gate_accepted=True, placed=True))
        save_robo_order(s, RoboOrder(run_id="r1", symbol="MSFT", side="buy",
                                     order_type="market", gate_accepted=True,
                                     simulated=True, placed=False))
        save_robo_order(s, RoboOrder(run_id="r1", symbol="AAPL", side="buy",
                                     order_type="market", gate_accepted=True,
                                     status="deferred_market_closed", placed=False))
    with get_session() as s:
        assert count_placed_orders_today(s) == 1  # only the real placement


def test_overweight_account_generates_sell(tmp_path):
    # VOO heavily overweight -> a trim (sell) is proposed and simulated.
    acct = cash_account("10", positions=[
        Position(symbol="VOO", quantity=Decimal("0.18"), price=Decimal("500")),  # $90 of $100
    ])
    broker = FakeBroker(acct)
    result = rebalance_run(make_config(), make_settings(tmp_path), broker=broker)
    assert result.status == "completed"
    sides = {d.order.symbol: d.order.side.value for d in result.decisions}
    assert sides.get("VOO") == "sell"
