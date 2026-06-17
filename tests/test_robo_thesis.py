"""Tests for the autonomous investor's thesis layer (Phase 3).

Tiers:
  A. PURE sizing + decay (no I/O).
  B. PURE deterministic invalidation.
  C. Thesis store CRUD (in-memory SQLite).
  D. compute_conviction_weights (sum-to-1, cash floor, scaling).
  E. Autonomous rebalance through the UNCHANGED gate (FakeBroker) — conviction
     drives weights, the gate still bounds everything, dry-run places nothing.
  F. Evaluator: pure thesis parsing + invalidation short-circuits the LLM.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from investment_monitor.analysis.thesis_evaluator import (
    ThesisEvaluator,
    parse_thesis_response,
)
from investment_monitor.robo.config import RoboCaps, RoboConfig, SizingConfig
from investment_monitor.robo.invalidation import check_invalidation
from investment_monitor.robo.models import AccountState, Position
from investment_monitor.robo.rebalance import rebalance_run
from investment_monitor.robo.sizing import (
    RiskMetrics,
    compute_conviction_weights,
    decay_conviction,
    risk_from_sim,
    size_position,
)
from investment_monitor.config import Settings
from investment_monitor.robo.broker import PlacedOrder, PreflightResult
from investment_monitor.storage import (
    SimulationResult,
    Thesis,
    ThesisStatus,
    get_active_theses,
    get_session,
    get_thesis,
    init_db,
    invalidate_thesis,
    record_conviction_update,
    save_thesis,
)

SC = SizingConfig()


# --------------------------------------------------------------------------- #
# A. Pure sizing / decay
# --------------------------------------------------------------------------- #
def test_size_position_bounds_and_monotonicity():
    risk = RiskMetrics(drift=0.15, volatility=0.18, var_95=-0.15, cvar_95=-0.20)
    low = size_position(0.3, risk, SC)
    high = size_position(0.9, risk, SC)
    assert 0.0 <= low < high <= SC.max_position_weight
    assert size_position(0.0, risk, SC) == 0.0


def test_size_position_capped():
    # Huge edge, tiny tail -> still capped at max_position_weight.
    risk = RiskMetrics(drift=2.0, volatility=0.05, var_95=-0.01, cvar_95=-0.01)
    assert size_position(1.0, risk, SC) == SC.max_position_weight


def test_size_position_no_sim_floor():
    assert size_position(0.8, None, SC) == min(0.8 * SC.no_sim_weight_per_conviction,
                                               SC.max_position_weight)


def test_size_position_tail_haircut_shrinks_size():
    mild = RiskMetrics(drift=0.15, volatility=0.18, var_95=-0.05, cvar_95=-0.05)
    severe = RiskMetrics(drift=0.15, volatility=0.18, var_95=-0.40, cvar_95=-0.50)
    assert size_position(0.8, severe, SC) < size_position(0.8, mild, SC)


def test_decay_toward_floor():
    assert decay_conviction(1.0, 0.0, SC) == 1.0
    assert decay_conviction(1.0, SC.conviction_half_life_days, SC) == 0.75  # halfway 1.0->0.5
    # Far future -> approaches the floor.
    assert abs(decay_conviction(1.0, 300.0, SC) - SC.conviction_floor) < 0.01


def test_risk_from_sim_defensive_keys():
    class FakeSim:
        drift = 0.1
        volatility = 0.2
        results_90d = {"var_95": -0.12, "cvar_95": -0.18}  # non-"base_" variant
    rm = risk_from_sim(FakeSim())
    assert rm.var_95 == -0.12 and rm.cvar_95 == -0.18
    assert risk_from_sim(None) is None


# --------------------------------------------------------------------------- #
# B. Pure invalidation
# --------------------------------------------------------------------------- #
def test_invalidation_composite_drop():
    cond = {"composite_drop": 15}
    assert check_invalidation(cond, entry_composite=80, latest_composite=60) is not None
    assert check_invalidation(cond, entry_composite=80, latest_composite=70) is None


def test_invalidation_price_drop():
    cond = {"price_drop_pct": 25}
    assert check_invalidation(cond, entry_price=100.0, latest_price=70.0) is not None
    assert check_invalidation(cond, entry_price=100.0, latest_price=80.0) is None


def test_invalidation_keywords():
    cond = {"keywords": ["fraud", "bankruptcy"]}
    assert check_invalidation(cond, recent_alert_keywords=["fraud"]) is not None
    assert check_invalidation(cond, recent_alert_keywords=["earnings"]) is None


def test_invalidation_none_when_empty():
    assert check_invalidation({}, latest_price=10, entry_price=100) is None  # no condition keys


def test_invalidation_composite_drop_boundary():
    # An EXACT drop of `composite_drop` points fires (>= semantics, matches docstring).
    cond = {"composite_drop": 10}
    assert check_invalidation(cond, entry_composite=100, latest_composite=90) is not None
    assert check_invalidation(cond, entry_composite=100, latest_composite=91) is None


# --------------------------------------------------------------------------- #
# C. Thesis store CRUD
# --------------------------------------------------------------------------- #
def _seed_thesis(db, **kw):
    init_db(db)
    with get_session() as s:
        t = Thesis(
            symbol=kw.get("symbol", "VOO"),
            account_id=kw.get("account_id"),
            narrative=kw.get("narrative", "test"),
            conviction=kw.get("conviction", 0.8),
            status=kw.get("status", ThesisStatus.ACTIVE.value),
            entry_conditions=kw.get("entry_conditions", {}),
            invalidation_conditions=kw.get("invalidation_conditions", {}),
        )
        save_thesis(s, t)


def test_thesis_crud_and_invalidate(tmp_path):
    db = tmp_path / "t.db"
    _seed_thesis(db, symbol="VOO", conviction=0.8)
    init_db(db)
    with get_session() as s:
        active = get_active_theses(s)
        assert len(active) == 1 and active[0].symbol == "VOO"
        record_conviction_update(s, active[0], 0.6, trigger="test")
        assert active[0].conviction == 0.6
        assert active[0].conviction_history[-1]["trigger"] == "test"
    with get_session() as s:
        t = get_thesis(s, "VOO")
        invalidate_thesis(s, t, "test reason")
        assert t.status == ThesisStatus.INVALIDATED.value
        assert t.conviction == 0.0
    with get_session() as s:
        assert get_active_theses(s) == []  # invalidated is no longer active


# --------------------------------------------------------------------------- #
# D. compute_conviction_weights
# --------------------------------------------------------------------------- #
def _autonomous_config(**sizing_kw):
    return RoboConfig(
        mode="autonomous",
        target_allocation={},  # autonomous: conviction-driven, no fixed allocation
        allowlist=[],
        sizing=SizingConfig(**sizing_kw),
        caps=RoboCaps(max_order_pct=0.5, max_orders_per_run=10, max_orders_per_day=20),
    )


def test_conviction_weights_sum_to_one_and_keep_cash(tmp_path):
    db = tmp_path / "t.db"
    _seed_thesis(db, symbol="VOO", conviction=0.9)
    init_db(db)
    with get_session() as s:
        # add a second active thesis
        save_thesis(s, Thesis(symbol="SCHD", conviction=0.7,
                              status=ThesisStatus.ACTIVE.value))
    with get_session() as s:
        weights = compute_conviction_weights(s, _autonomous_config())
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["CASH"] >= _autonomous_config().sizing.min_cash_weight - 1e-9
    assert all(w >= 0 for w in weights.values())


def test_conviction_weights_cap_duplicate_symbol(tmp_path):
    # Two live theses for the SAME symbol must not double-count past max_position_weight.
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=1.0, status=ThesisStatus.ACTIVE.value))
        save_thesis(s, Thesis(symbol="VOO", conviction=1.0, status=ThesisStatus.ACTIVE.value))
    cfg = _autonomous_config(max_position_weight=0.1, no_sim_weight_per_conviction=0.5,
                             min_cash_weight=0.0)
    with get_session() as s:
        weights = compute_conviction_weights(s, cfg)
    assert weights["VOO"] <= 0.1 + 1e-9  # clamped at the per-name cap, not 0.2
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_conviction_weights_scale_down_to_cash_floor(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    # Many high-conviction names with a generous cap so raw equity > 1 - min_cash.
    with get_session() as s:
        for i in range(8):
            save_thesis(s, Thesis(symbol=f"S{i}", conviction=1.0,
                                  status=ThesisStatus.ACTIVE.value))
    cfg = _autonomous_config(max_position_weight=0.5, no_sim_weight_per_conviction=0.5, min_cash_weight=0.1)
    with get_session() as s:
        weights = compute_conviction_weights(s, cfg)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["CASH"] >= 0.1 - 1e-9  # cash floor respected even when over-subscribed


# --------------------------------------------------------------------------- #
# E. Autonomous rebalance through the unchanged gate
# --------------------------------------------------------------------------- #
class FakeBroker:
    def __init__(self, account):
        self._account = account
        self.dry_run = True
        self.place_called = 0

    def get_account_state(self):
        return self._account

    def get_quotes(self, symbols):
        return {"VOO": Decimal("500"), "SCHD": Decimal("80"), "AAPL": Decimal("200")}

    def preflight(self, order):
        return PreflightResult(ok=True, estimated_cost=order.notional or Decimal("0"),
                               margin_requirement=Decimal("0"))

    def place_order(self, order):
        self.place_called += 1
        return PlacedOrder(order_id="x", status="NEW", simulated=False)


def _cash_account(cash="100", positions=None):
    return AccountState(
        account_id="ACC1", account_type="BROKERAGE", is_cash_account=True,
        has_margin=False, settled_cash=Decimal(cash), positions=positions or [],
    )


def _settings(tmp_path, db):
    return Settings(public_api_token="t", robo_force_dry_run=True,
                    config_dir=tmp_path, data_dir=tmp_path, log_dir=tmp_path, db_path=db)


def _seed_sim(session, ticker, drift=0.15, vol=0.18, cvar=-0.20):
    session.add(SimulationResult(
        ticker=ticker, run_date=date.today(), entry_price=100.0, composite_score=80.0,
        num_simulations=1000, lookback_days=252, volatility=vol, drift=drift,
        results_90d={"base_var_95": -0.15, "base_cvar_95": cvar},
    ))


def test_autonomous_rebalance_buys_toward_conviction(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
        _seed_sim(s, "VOO")
    broker = FakeBroker(_cash_account("100"))
    result = rebalance_run(_autonomous_config(), _settings(tmp_path, db), broker=broker)

    assert result.status == "completed"
    assert result.source == "autonomous"
    assert result.dry_run is True
    assert broker.place_called == 0  # dry-run: nothing real placed
    sides = {d.order.symbol: d.order.side.value for d in result.decisions}
    assert sides.get("VOO") == "buy"  # conviction drove a buy toward target


def test_autonomous_gate_blocks_non_thesis_buys(tmp_path):
    # Only VOO has a thesis. The gate's allowlist = active theses (+ holdings), so a
    # proposal to buy AAPL (no thesis) would be rejected symbol_not_allowed. Here we
    # confirm AAPL is simply never proposed AND is absent from the effective universe.
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value))
        _seed_sim(s, "VOO")
    broker = FakeBroker(_cash_account("100"))
    result = rebalance_run(_autonomous_config(), _settings(tmp_path, db), broker=broker)
    symbols = {d.order.symbol for d in result.decisions}
    assert "AAPL" not in symbols  # no thesis -> never enters the tradeable universe


def test_autonomous_sells_held_name_without_thesis(tmp_path):
    # Hold AAPL but have no thesis for it -> autonomous mode trims it to 0 (a sell),
    # and the gate allows it because held symbols are always on the allowlist.
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value))
        _seed_sim(s, "VOO")
    held = [Position(symbol="AAPL", quantity=Decimal("0.1"), price=Decimal("200"))]  # $20 held
    broker = FakeBroker(_cash_account("100", positions=held))
    result = rebalance_run(_autonomous_config(), _settings(tmp_path, db), broker=broker)
    sides = {d.order.symbol: (d.order.side.value, d.accepted) for d in result.decisions}
    assert "AAPL" in sides and sides["AAPL"][0] == "sell"
    assert sides["AAPL"][1] is True  # the exit sell is accepted by the gate


# --------------------------------------------------------------------------- #
# F. Evaluator
# --------------------------------------------------------------------------- #
def test_parse_thesis_response_pure():
    good = '{"narrative": "Strong moat", "conviction": 0.7, "invalidation_conditions": {"price_drop_pct": 25}}'
    u = parse_thesis_response(good)
    assert u is not None and u.conviction == 0.7 and u.narrative == "Strong moat"
    assert u.invalidation_conditions == {"price_drop_pct": 25}
    # Fenced + noisy still parses.
    assert parse_thesis_response('```json\n{"narrative":"x","conviction":1.5}\n```').conviction == 1.0
    # Unusable -> None (fail-safe: caller keeps prior thesis).
    assert parse_thesis_response("no json here") is None
    assert parse_thesis_response('{"conviction": 0.5}') is None  # missing narrative


class _FakeLLM:
    model = "fake"

    def __init__(self, text):
        self._text = text
        self.client = self

    def is_available(self):
        return True

    def generate(self, model, prompt, options):
        return {"response": self._text}


def test_evaluator_updates_conviction(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value))
    llm = _FakeLLM('{"narrative": "Improving fundamentals", "conviction": 0.8}')
    evaluator = ThesisEvaluator(llm, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "VOO")
        action = evaluator.evaluate(s, t)
        assert action == "updated"
        assert t.conviction == 0.8 and t.narrative == "Improving fundamentals"


def test_evaluator_invalidation_short_circuits_llm(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        # Price already crashed below the thesis's stop, and a price row proves it.
        save_thesis(s, Thesis(
            symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 500.0},
            invalidation_conditions={"price_drop_pct": 25},
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="VOO", date=date.today(), close=300.0))  # -40% from entry
    # LLM would say "still bullish" — but deterministic invalidation must win.
    llm = _FakeLLM('{"narrative": "still bullish!", "conviction": 0.95}')
    evaluator = ThesisEvaluator(llm, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "VOO")
        action = evaluator.evaluate(s, t)
        assert action == "invalidated"
        assert t.conviction == 0.0 and t.status == ThesisStatus.INVALIDATED.value
