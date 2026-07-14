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
from investment_monitor.robo.config import ExitConfig, RoboCaps, RoboConfig, SizingConfig
from investment_monitor.robo.invalidation import check_exit, check_invalidation, entry_basis
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
# B2. Pure take-profit exits (check_exit — invalidation's upside twin)
# --------------------------------------------------------------------------- #
def test_exit_profit_target_boundary_and_fail_open():
    cond = {"profit_target_pct": 40}
    assert check_exit(cond, entry_price=10.0, latest_price=14.0) is not None  # exactly +40%
    assert check_exit(cond, entry_price=10.0, latest_price=13.9) is None
    # Missing data fails OPEN (no trip) — never force an exit off absent inputs.
    assert check_exit(cond, entry_price=None, latest_price=14.0) is None
    assert check_exit(cond, entry_price=10.0, latest_price=None) is None


def test_exit_trailing_stop_arms_then_fires():
    cond = {"trailing_stop_pct": 15, "trailing_arm_pct": 10}
    # NOT armed: the peak never cleared +10% — a fall from a flat peak is
    # invalidation's (price_drop_pct) problem, not the trailing stop's.
    assert check_exit(cond, entry_price=10.0, latest_price=8.4, high_water_mark=10.5) is None
    # Armed (+35% peak) and 18.5% off the high -> exit with the gain protected.
    assert check_exit(cond, entry_price=10.0, latest_price=11.0, high_water_mark=13.5) is not None
    # Armed but only ~10% off the high -> keep riding.
    assert check_exit(cond, entry_price=10.0, latest_price=12.15, high_water_mark=13.5) is None


def test_exit_horizon_boundary():
    cond = {"max_hold_days": 90}
    assert check_exit(cond, days_held=90) is not None
    assert check_exit(cond, days_held=89) is None
    assert check_exit(cond, days_held=None) is None


def test_exit_zero_disables_and_empty_never_fires():
    # A 0/absent threshold disables that trigger (per-thesis override semantics).
    assert check_exit({"profit_target_pct": 0}, entry_price=10.0, latest_price=100.0) is None
    assert check_exit({}, entry_price=10.0, latest_price=100.0,
                      high_water_mark=100.0, days_held=999) is None
    assert check_exit(None, entry_price=10.0, latest_price=100.0) is None


def test_entry_basis_prefers_fill_cost():
    assert entry_basis({"fill_cost": 9.5, "entry_price": 10.0}) == 9.5
    assert entry_basis({"entry_price": 10.0}) == 10.0
    assert entry_basis({"fill_cost": "bad", "entry_price": 10.0}) == 10.0  # garbage skipped
    assert entry_basis({}) is None and entry_basis(None) is None


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


def test_parse_thesis_sanitizes_invalidation():
    # NEGATIVE drop magnitudes (a common LLM mistake) are coerced positive, so a
    # fresh thesis isn't invalidated on the spot.
    u = parse_thesis_response(
        '{"narrative":"x","conviction":0.7,"invalidation_conditions":'
        '{"composite_drop":-15,"price_drop_pct":-20,"keywords":["fraud",""]}}'
    )
    assert u.invalidation_conditions["composite_drop"] == 15.0
    assert u.invalidation_conditions["price_drop_pct"] == 20.0
    assert u.invalidation_conditions["keywords"] == ["fraud"]
    # Zero / non-numeric thresholds are dropped entirely.
    u2 = parse_thesis_response(
        '{"narrative":"x","conviction":0.7,"invalidation_conditions":'
        '{"composite_drop":0,"price_drop_pct":"bad"}}'
    )
    assert "composite_drop" not in u2.invalidation_conditions
    assert "price_drop_pct" not in u2.invalidation_conditions


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


# --------------------------------------------------------------------------- #
# G. Take-profit exits through the evaluator
# --------------------------------------------------------------------------- #
def test_parse_thesis_sanitizes_exit_conditions():
    # LLM-proposed exit thresholds are clamped into sane bands: a hallucinated 2%
    # target can't scalp-exit a position, 5000 days can't disable the horizon.
    u = parse_thesis_response(
        '{"narrative":"x","conviction":0.7,"exit_conditions":'
        '{"profit_target_pct":2,"max_hold_days":5000,"trailing_stop_pct":-20}}'
    )
    assert u.exit_conditions["profit_target_pct"] == 10.0   # clamped up to the floor
    assert u.exit_conditions["max_hold_days"] == 365.0      # clamped down to the cap
    assert u.exit_conditions["trailing_stop_pct"] == 20.0   # negative -> magnitude
    # Absent block -> empty dict (config defaults apply unchanged).
    assert parse_thesis_response('{"narrative":"x","conviction":0.7}').exit_conditions == {}


def test_evaluator_take_profit_short_circuits_llm(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="MOON", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 10.0},
            invalidation_conditions={"price_drop_pct": 25},
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="MOON", date=date.today(), close=15.0))  # +50% >= default 40%
    # LLM wants to let it ride — the deterministic profit target must win.
    llm = _FakeLLM('{"narrative": "to the moon!", "conviction": 0.99}')
    evaluator = ThesisEvaluator(llm, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "MOON")
        action = evaluator.evaluate(s, t)
        assert action == "exited"
        assert t.status == ThesisStatus.EXITED.value and t.conviction == 0.0
        assert "profit target" in t.conviction_history[-1]["trigger"]
    with get_session() as s:
        assert get_active_theses(s) == []          # no longer drives allocation
        assert get_thesis(s, "MOON") is None       # EXITED vanishes from get_thesis


def test_evaluator_trailing_stop_uses_high_water(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="RIDE", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 10.0},
            high_water_mark=13.5,   # peaked +35% on an earlier pass
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="RIDE", date=date.today(), close=11.0))  # 18.5% off the high
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "RIDE")
        assert evaluator.evaluate(s, t) == "exited"
        assert t.status == ThesisStatus.EXITED.value
        assert "trailing stop" in t.conviction_history[-1]["trigger"]


def test_evaluator_maintains_high_water_mark(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="HWMK", conviction=0.5, status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 10.0}))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="HWMK", date=date.today(), close=11.0))
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "HWMK")
        # +10%: armed but not fallen -> no exit; the peak must be recorded though.
        assert evaluator.evaluate(s, t) == "unchanged"
        assert t.high_water_mark == 11.0
        assert t.status == ThesisStatus.ACTIVE.value


def test_evaluator_horizon_exit_from_thesis_stamp(tmp_path):
    from datetime import datetime, timedelta, timezone

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="OLDIE", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 10.0},
            exit_conditions={"max_hold_days": 90},   # confluence-style per-thesis stamp
            created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=120),
        ))
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "OLDIE")
        assert evaluator.evaluate(s, t) == "exited"   # no price data needed for horizon
        assert "horizon" in t.conviction_history[-1]["trigger"]


def test_evaluator_exit_config_master_switch(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="HODL", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 10.0},
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="HODL", date=date.today(), close=20.0))  # +100%
    cfg = _autonomous_config().model_copy(update={"exits": ExitConfig(enabled=False)})
    evaluator = ThesisEvaluator(None, cfg)
    with get_session() as s:
        t = get_thesis(s, "HODL")
        assert evaluator.evaluate(s, t) == "unchanged"  # exits off -> position rides
        assert t.status == ThesisStatus.ACTIVE.value
