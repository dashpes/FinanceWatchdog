"""Go-live hardening tests (Phase 7) — regression guards for the audit blockers.

Covers the real-order path fixes:
  A. Cash mapping respects a real $0 cash-only balance (no fall-through to margin).
  B. Market-hours gate honors NYSE holidays + early closes.
  C. Placement durability: a non-BrokerError SDK/network failure on one order does
     NOT crash the run, is recorded as place_failed, and a prior successful real
     placement survives (committed, so the next run can't double-place it).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from investment_monitor.config import Settings
from investment_monitor.robo.broker import PlacedOrder, PreflightResult, account_state_from_raw
from investment_monitor.robo.config import RoboCaps, RoboConfig, SizingConfig
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.robo.models import AccountState, Position
from investment_monitor.robo.rebalance import rebalance_run
from investment_monitor.storage import (
    SimulationResult,
    Thesis,
    ThesisStatus,
    get_robo_orders_for_run,
    get_session,
    init_db,
    save_thesis,
)


# --------------------------------------------------------------------------- #
# A. Cash mapping (the falsy-Decimal bug)
# --------------------------------------------------------------------------- #
def test_cash_mapping_respects_zero_cash_only():
    # cash_only_buying_power is exactly 0 -> settled_cash must be 0, NOT the
    # marginable buying_power (which would let the advisor "spend" margin).
    acct = {"brokerageAccountType": "CASH", "accountId": "A1"}
    portfolio = {"buying_power": {"cash_only_buying_power": "0", "buying_power": "200"}}
    state = account_state_from_raw(acct, portfolio)
    assert state.settled_cash == Decimal("0")
    assert state.is_cash_account is True


def test_cash_mapping_never_falls_back_to_marginable():
    # No cash-only field at all: we must NOT read the marginable buying_power as cash.
    acct = {"brokerageAccountType": "CASH", "accountId": "A1"}
    portfolio = {"buying_power": {"buying_power": "200"}}
    state = account_state_from_raw(acct, portfolio)
    assert state.settled_cash == Decimal("0")


def test_cash_mapping_reads_real_cash_only_balance():
    acct = {"brokerageAccountType": "CASH", "accountId": "A1"}
    portfolio = {"buying_power": {"cash_only_buying_power": "52.50", "buying_power": "999"}}
    state = account_state_from_raw(acct, portfolio)
    assert state.settled_cash == Decimal("52.50")


# --------------------------------------------------------------------------- #
# B. Market-hours gate: holidays + early closes
# --------------------------------------------------------------------------- #
def test_market_closed_on_juneteenth_2026():
    # 2026-06-19 is a Friday Juneteenth holiday — the day after a typical go-live.
    assert is_market_open(datetime(2026, 6, 19, 12, 0)) is False


def test_market_open_on_normal_weekday():
    # 2026-06-18 is a normal Thursday.
    assert is_market_open(datetime(2026, 6, 18, 12, 0)) is True
    assert is_market_open(datetime(2026, 6, 18, 9, 0)) is False   # pre-open
    assert is_market_open(datetime(2026, 6, 20, 12, 0)) is False  # Saturday


def test_market_early_close_half_day():
    # 2026-11-27 (Black Friday) closes at 13:00 ET.
    assert is_market_open(datetime(2026, 11, 27, 12, 0)) is True
    assert is_market_open(datetime(2026, 11, 27, 14, 0)) is False


# --------------------------------------------------------------------------- #
# C. Placement durability
# --------------------------------------------------------------------------- #
def test_reconcile_schema_adds_missing_column(tmp_path):
    # Simulate a learning_events table created before `as_of_date` existed (the exact
    # live-DB drift the dry-run sanity check surfaced) and confirm reconcile adds it.
    from sqlalchemy import create_engine, inspect, text

    from investment_monitor.storage.database import _reconcile_schema

    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as c:
        c.execute(text(
            "CREATE TABLE learning_events ("
            "id INTEGER PRIMARY KEY, kind VARCHAR, symbol VARCHAR, created_at DATETIME)"
        ))
    _reconcile_schema(eng)
    cols = {col["name"] for col in inspect(eng).get_columns("learning_events")}
    assert "as_of_date" in cols        # the column that was silently missing
    assert "realized_return" in cols   # other additive columns reconciled too
    # Idempotent: a second pass is a no-op (doesn't raise / duplicate).
    _reconcile_schema(eng)


class _FlakyBroker:
    """Places the first order, then raises a non-BrokerError (SDK/network) error."""

    def __init__(self, account):
        self._account = account
        self.dry_run = False
        self.place_called = 0

    def get_account_state(self):
        return self._account

    def get_quotes(self, symbols):
        return {"VOO": Decimal("100"), "SCHD": Decimal("80")}

    def preflight(self, order):
        return PreflightResult(ok=True, estimated_cost=order.notional or Decimal("0"),
                               margin_requirement=Decimal("0"))

    def place_order(self, order):
        self.place_called += 1
        if self.place_called == 1:
            return PlacedOrder(order_id="OK1", status="NEW", simulated=False)
        raise RuntimeError("simulated SDK 500 / read timeout")  # NOT a BrokerError


def _live_config():
    return RoboConfig(
        mode="autonomous", target_allocation={}, allowlist=[], use_llm=False,
        dry_run=False, require_market_hours=False,  # live path, time-independent
        sizing=SizingConfig(),
        caps=RoboCaps(max_order_pct=1.0, max_orders_per_run=10, max_orders_per_day=20,
                      max_per_name_weight=1.0),
    )


def _cash_account(cash="100"):
    return AccountState(
        account_id="ACC1", account_type="BROKERAGE", is_cash_account=True,
        has_margin=False, settled_cash=Decimal(cash), positions=[],
    )


def _settings(tmp_path, db):
    return Settings(public_api_token="t", robo_force_dry_run=False,  # live kill-switch OFF
                    config_dir=tmp_path, data_dir=tmp_path, log_dir=tmp_path, db_path=db)


def _seed_two_theses(db):
    init_db(db)
    with get_session() as s:
        for sym in ("VOO", "SCHD"):
            save_thesis(s, Thesis(symbol=sym, conviction=0.9, status=ThesisStatus.ACTIVE.value,
                                  entry_conditions={"entry_price": 100.0}))
            s.add(SimulationResult(
                ticker=sym, run_date=date.today(), entry_price=100.0, composite_score=80.0,
                num_simulations=1000, lookback_days=252, volatility=0.18, drift=0.15,
                results_90d={"base_var_95": -0.15, "base_cvar_95": -0.20},
            ))


def test_sdk_error_mid_run_does_not_crash_and_keeps_prior_placement(tmp_path):
    db = tmp_path / "t.db"
    _seed_two_theses(db)
    broker = _FlakyBroker(_cash_account("100"))

    # Must not raise even though place_order #2 throws a non-BrokerError.
    result = rebalance_run(_live_config(), _settings(tmp_path, db), broker=broker)

    assert broker.place_called == 2          # both orders attempted (one bad order didn't abort)
    assert result.dry_run is False           # real live path exercised
    assert result.status == "completed"      # run finalized, not crashed
    assert result.num_placed == 1            # exactly the one that succeeded

    # The successful placement is durably recorded (committed), and the failure is
    # recorded as place_failed — so the next run won't re-derive and double-place it.
    init_db(db)
    with get_session() as s:
        rows = get_robo_orders_for_run(s, result.run_id)
        placed = [r for r in rows if r.placed]
        failed = [r for r in rows if r.status == "place_failed"]
        assert len(placed) == 1 and placed[0].broker_order_id == "OK1"
        assert len(failed) == 1
