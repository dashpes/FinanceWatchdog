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
        self.get_order_called = 0

    def get_account_state(self) -> AccountState:
        return self._account

    def get_order(self, order_id):
        # Tracked so tests can assert the placement path does NOT poll right after
        # placing (Public placement is async / eventually consistent).
        self.get_order_called += 1
        return {"status": "NEW"}

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
    # Placement must not eagerly poll get_order — fills are reconciled next run.
    assert broker.get_order_called == 0


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


def test_run_snapshots_unrealized_pnl(tmp_path):
    # The run records the broker's unrealized P&L so runs form a P&L time series.
    acct = cash_account("100", positions=[
        Position(symbol="VOO", quantity=Decimal("0.1"), price=Decimal("500"),
                 unit_cost=Decimal("450"), unrealized_gain=Decimal("5")),
    ])
    rebalance_run(make_config(), make_settings(tmp_path), broker=FakeBroker(acct))
    init_db(tmp_path / "test.db")
    with get_session() as session:
        assert get_recent_robo_runs(session, limit=1)[0].unrealized_pnl == 5.0


def test_run_snapshot_unrealized_none_without_cost_basis(tmp_path):
    # No broker cost basis (paper) -> snapshot stays None, never a misleading 0.
    rebalance_run(make_config(), make_settings(tmp_path), broker=FakeBroker(cash_account("100")))
    init_db(tmp_path / "test.db")
    with get_session() as session:
        assert get_recent_robo_runs(session, limit=1)[0].unrealized_pnl is None


class _UnbuyableBroker(FakeBroker):
    """Broker whose preflight refuses everything as un-buyable (e.g. close-only)."""

    def preflight(self, order):
        self.preflight_called += 1
        return PreflightResult(
            ok=False,
            estimated_cost=order.notional or Decimal("0"),
            margin_requirement=Decimal("0"),
            message="API Error 400: This asset is currently only available for "
            "trading when closing an existing position on Public.",
        )


def test_unbuyable_preflight_failure_is_learned_into_blocklist(tmp_path):
    from investment_monitor.robo.blocklist import load_learned

    settings = make_settings(tmp_path)
    broker = _UnbuyableBroker(cash_account("100"))
    result = rebalance_run(make_config(), settings, broker=broker)

    # Proposed buys all fail preflight -> none placed, but the names are learned.
    assert result.num_placed == 0
    learned = load_learned(str(settings.db_path))
    assert {"VOO", "SCHD"} <= learned  # both refused buys recorded


# --------------------------------------------------------------------------- #
# Finding #9 — fill_cost must come from THIS thesis's own entry order, not the
# broker's blended unit cost for any held symbol.
# --------------------------------------------------------------------------- #
def _save_thesis(session, symbol, created_at, entry_price=100.0, account_id="ACC1"):
    from investment_monitor.storage import Thesis, ThesisStatus, save_thesis

    t = Thesis(
        symbol=symbol, account_id=account_id, narrative="x", conviction=0.6,
        status=ThesisStatus.ACTIVE.value,
        entry_conditions={"entry_price": entry_price},
        invalidation_conditions={}, evidence_refs={}, conviction_history=[],
    )
    save_thesis(session, t)
    t.created_at = created_at  # control the entry-order time window deterministically
    session.flush()
    return t


def _filled_buy(session, symbol, created_at, fill_price, run_id="r1"):
    from investment_monitor.storage import RoboOrder, save_robo_order

    o = RoboOrder(run_id=run_id, symbol=symbol, side="buy", order_type="market",
                  quantity=1.0, source="deterministic", placed=True,
                  broker_order_id=f"{symbol}-{fill_price}", status="placed",
                  fill_price=fill_price, fill_quantity=1.0, fill_status="FILLED")
    save_robo_order(session, o)
    o.created_at = created_at
    session.flush()
    return o


def test_fill_cost_uses_thesis_own_entry_order_not_blended_basis(tmp_path):
    # The position carries a BLENDED broker unit_cost (e.g. 450), but the order that
    # actually opened THIS thesis filled at 470. fill_cost must record 470, not 450.
    import datetime as _dt

    from investment_monitor.robo.models import Position
    from investment_monitor.robo.rebalance import _reconcile_fill_costs
    from investment_monitor.storage import get_active_theses, get_session, init_db

    init_db(tmp_path / "t.db")
    t0 = _dt.datetime(2026, 6, 1, tzinfo=_dt.timezone.utc)
    acct = cash_account("100", positions=[
        Position(symbol="VOO", quantity=Decimal("2"), price=Decimal("500"),
                 unit_cost=Decimal("450")),  # broker blended basis (NOT the thesis entry)
    ])
    with get_session() as s:
        _save_thesis(s, "VOO", created_at=t0, entry_price=480.0)
        # The thesis's own entry order filled at 470, AFTER the thesis was created.
        _filled_buy(s, "VOO", created_at=t0 + _dt.timedelta(minutes=5), fill_price=470.0)
    with get_session() as s:
        _reconcile_fill_costs(s, acct)
    with get_session() as s:
        th = get_active_theses(s, "ACC1")[0]
        assert th.entry_conditions["fill_cost"] == 470.0  # the entry fill, not 450


def test_fill_cost_not_set_for_preexisting_holding(tmp_path):
    # A symbol held BEFORE the thesis (its only filled buy predates thesis creation):
    # there is no entry-corresponding fill, so fill_cost must NOT be written and the
    # idea-time entry_price stays the basis.
    import datetime as _dt

    from investment_monitor.robo.models import Position
    from investment_monitor.robo.rebalance import _reconcile_fill_costs
    from investment_monitor.storage import get_active_theses, get_session, init_db

    init_db(tmp_path / "t.db")
    thesis_created = _dt.datetime(2026, 6, 10, tzinfo=_dt.timezone.utc)
    acct = cash_account("100", positions=[
        Position(symbol="VOO", quantity=Decimal("2"), price=Decimal("500"),
                 unit_cost=Decimal("450")),
    ])
    with get_session() as s:
        _save_thesis(s, "VOO", created_at=thesis_created, entry_price=480.0)
        # Buy filled a week BEFORE the thesis existed -> pre-existing holding.
        _filled_buy(s, "VOO", created_at=thesis_created - _dt.timedelta(days=7),
                    fill_price=470.0)
    with get_session() as s:
        _reconcile_fill_costs(s, acct)
    with get_session() as s:
        th = get_active_theses(s, "ACC1")[0]
        assert "fill_cost" not in th.entry_conditions       # not overwritten
        assert th.entry_conditions["entry_price"] == 480.0  # idea-time basis kept


def test_fill_cost_skipped_when_symbol_not_held(tmp_path):
    # A thesis whose symbol is no longer held must not get a fill_cost from a stale fill.
    import datetime as _dt

    from investment_monitor.robo.rebalance import _reconcile_fill_costs
    from investment_monitor.storage import get_active_theses, get_session, init_db

    init_db(tmp_path / "t.db")
    t0 = _dt.datetime(2026, 6, 1, tzinfo=_dt.timezone.utc)
    acct = cash_account("100")  # holds nothing
    with get_session() as s:
        _save_thesis(s, "VOO", created_at=t0, entry_price=480.0)
        _filled_buy(s, "VOO", created_at=t0 + _dt.timedelta(minutes=5), fill_price=470.0)
    with get_session() as s:
        _reconcile_fill_costs(s, acct)
    with get_session() as s:
        th = get_active_theses(s, "ACC1")[0]
        assert "fill_cost" not in th.entry_conditions


# --------------------------------------------------------------------------- #
# Drawdown breaker — true ALL-TIME peak via aggregate, not a capped window.
# --------------------------------------------------------------------------- #
def test_drawdown_halt_uses_alltime_peak_beyond_recent_window(tmp_path):
    # An old, very-high peak that would age out of any fixed recent-row window must
    # still be the high-water mark: the breaker computes MAX(total_value) over ALL runs,
    # so a real drawdown from that peak still halts buys.
    import datetime as _dt

    from investment_monitor.robo.config import RoboCaps, RoboConfig
    from investment_monitor.robo.rebalance import _drawdown_halt
    from investment_monitor.storage import RoboRun, get_session, init_db, save_robo_run

    init_db(tmp_path / "t.db")
    base = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
    with get_session() as s:
        # One very old high peak, then many newer, lower runs (the kind that would
        # crowd the old peak out of a capped recent window).
        old = RoboRun(run_id="peak", dry_run=False, account_id="ACC1",
                      total_value=200.0, status="completed")
        save_robo_run(s, old)
        old.started_at = base
        s.flush()
        for i in range(50):
            r = RoboRun(run_id=f"r{i}", dry_run=False, account_id="ACC1",
                        total_value=120.0, status="completed")
            save_robo_run(s, r)
            r.started_at = base + _dt.timedelta(days=i + 1)
            s.flush()

    cfg = RoboConfig(target_allocation={"VOO": 1.0}, allowlist=["VOO"], use_llm=False,
                     caps=RoboCaps(max_drawdown_pct=25.0))
    with get_session() as s:
        # current 120 vs all-time peak 200 -> 40% drawdown >= 25% -> halt.
        assert _drawdown_halt(s, cfg, Decimal("120"), "ACC1") is True
        # current 160 vs peak 200 -> 20% < 25% -> no halt.
        assert _drawdown_halt(s, cfg, Decimal("160"), "ACC1") is False


def test_drawdown_halt_scoped_per_account(tmp_path):
    # Another account's higher peak must not pin this account's drawdown.
    from investment_monitor.robo.config import RoboCaps, RoboConfig
    from investment_monitor.robo.rebalance import _drawdown_halt
    from investment_monitor.storage import RoboRun, get_session, init_db, save_robo_run

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_robo_run(s, RoboRun(run_id="a", dry_run=False, account_id="ACC1",
                                 total_value=100.0, status="completed"))
        save_robo_run(s, RoboRun(run_id="b", dry_run=False, account_id="OTHER",
                                 total_value=1000.0, status="completed"))
    cfg = RoboConfig(target_allocation={"VOO": 1.0}, allowlist=["VOO"], use_llm=False,
                     caps=RoboCaps(max_drawdown_pct=25.0))
    with get_session() as s:
        # ACC1 peak is 100; current 100 -> 0% drawdown, OTHER's 1000 ignored.
        assert _drawdown_halt(s, cfg, Decimal("100"), "ACC1") is False
