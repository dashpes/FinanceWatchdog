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
    select_top_positions,
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


def test_size_position_risk_modulates_but_never_vetoes():
    # A beaten-down name (negative trailing drift — the classic contrarian confluence
    # setup) must NOT size to zero on the Kelly term; it falls back to the conviction
    # floor, still shrunk by the tail haircut.
    down = RiskMetrics(drift=-0.05, volatility=0.45, var_95=-0.25, cvar_95=-0.30)
    got = size_position(0.9, down, SC)
    floor = 0.9 * SC.no_sim_weight_per_conviction
    assert 0 < got < floor            # alive, but tail-haircut below the raw floor
    # And a genuinely strong Sharpe still sizes ABOVE the floor (Kelly adds).
    strong = RiskMetrics(drift=0.30, volatility=0.15, var_95=-0.05, cvar_95=-0.06)
    assert size_position(0.9, strong, SC) > floor


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
# A2. Pure top-N selection with incumbent hysteresis
# --------------------------------------------------------------------------- #
def test_select_top_positions_plain_topn_and_ties():
    raw = {"A": 0.2, "B": 0.3, "C": 0.2, "D": 0.1}
    out = select_top_positions(raw, 2)
    assert out == {"B": 0.3, "A": 0.2}  # tie A/C breaks by symbol, never dict order
    assert select_top_positions(raw, 0) == raw       # no cap
    assert select_top_positions(raw, 10) == raw      # under cap


def test_select_top_positions_incumbent_survives_noise():
    # Held incumbent at 0.17; challenger at 0.19 is only ~12% stronger — rank noise
    # in a saturated conviction band. The incumbent keeps its slot.
    raw = {"INC": 0.17, "CHAL": 0.19, "TOP": 0.30}
    out = select_top_positions(raw, 2, held={"INC"}, hysteresis=0.25)
    assert set(out) == {"TOP", "INC"}
    # Without hysteresis the same numbers rotate the position out.
    out0 = select_top_positions(raw, 2, held={"INC"}, hysteresis=0.0)
    assert set(out0) == {"TOP", "CHAL"}


def test_select_top_positions_clear_winner_evicts():
    raw = {"INC": 0.17, "CHAL": 0.22, "TOP": 0.30}  # 0.22 >= 0.17 * 1.25
    out = select_top_positions(raw, 2, held={"INC"}, hysteresis=0.25)
    assert set(out) == {"TOP", "CHAL"}


def test_select_top_positions_weakest_incumbent_evicted_first():
    raw = {"I1": 0.10, "I2": 0.20, "C1": 0.30, "C2": 0.14}
    # One challenger strong enough for I1 (0.30 >= 0.125) but slots stay full:
    # I2 survives because C2 (0.14) is under its 0.25 margin (needs 0.25).
    out = select_top_positions(raw, 2, held={"I1", "I2"}, hysteresis=0.25)
    assert set(out) == {"C1", "I2"}


def test_select_top_positions_incumbent_wins_exact_ties():
    # Once the position cap flattens many strong names to the SAME weight, ties are
    # the common case — and an exact tie must never rotate a real position (the
    # challenger has not beaten the incumbent by any margin, let alone the required
    # one). Alphabetical order must not pick the book.
    raw = {"AAAA": 0.25, "BBBB": 0.25, "HELD": 0.25}
    out = select_top_positions(raw, 2, held={"HELD"}, hysteresis=0.25)
    assert "HELD" in out and len(out) == 2
    # Even with hysteresis off, a tie keeps the incumbent (rotating costs spread).
    out0 = select_top_positions(raw, 2, held={"HELD"}, hysteresis=0.0)
    assert "HELD" in out0


def test_select_top_positions_exited_incumbent_holds_no_slot():
    # A broken/exited thesis has weight 0 -> it is not in raw at all, so hysteresis
    # can never delay an exit.
    raw = {"CHAL": 0.19, "TOP": 0.30}
    out = select_top_positions(raw, 2, held={"GONE"}, hysteresis=0.25)
    assert set(out) == {"TOP", "CHAL"}


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


def test_exit_giveback_trail_banks_a_real_gain():
    # THE fix for "positions up 10-15% never took profit, then sold at a loss".
    # Peak +12%, giveback 40% -> exit once the gain falls to +7.2%: still a PROFIT.
    # The price-distance trail at the shipped 15% would have exited the identical
    # position at -4.8% (1.12 x 0.85), which is why it never banked anything.
    cond = {"trailing_giveback_pct": 40, "trailing_arm_pct": 8}
    assert check_exit(cond, entry_price=10.0, latest_price=11.3, high_water_mark=11.2) is None
    reason = check_exit(cond, entry_price=10.0, latest_price=10.72, high_water_mark=11.2)
    assert reason is not None and "trailing giveback" in reason
    assert "+7.2%" in reason  # the exit is above entry, by construction
    # Same peak under the OLD mechanism exits below entry — the bug, pinned.
    old = check_exit({"trailing_stop_pct": 15, "trailing_arm_pct": 10},
                     entry_price=10.0, latest_price=9.52, high_water_mark=11.2)
    assert old is not None  # fires at -4.8%: a loss


def test_exit_giveback_arms_and_scales_with_peak():
    cond = {"trailing_giveback_pct": 40, "trailing_arm_pct": 8}
    # Peak only +5% -> not armed, no exit (that downside is invalidation's job).
    assert check_exit(cond, entry_price=10.0, latest_price=10.0, high_water_mark=10.5) is None
    # Bigger peak -> proportionally more room: +50% peak keeps riding down to +30%.
    assert check_exit(cond, entry_price=10.0, latest_price=13.5, high_water_mark=15.0) is None
    assert check_exit(cond, entry_price=10.0, latest_price=12.9, high_water_mark=15.0) is not None
    # 0 disables; the legacy price trail still works when explicitly set.
    assert check_exit({"trailing_giveback_pct": 0}, entry_price=10.0,
                      latest_price=10.72, high_water_mark=11.2) is None


def test_exit_config_defaults_flip_to_giveback():
    from investment_monitor.robo.config import ExitConfig

    cond = ExitConfig().as_conditions()
    assert cond["trailing_giveback_pct"] == 40.0
    assert cond["trailing_stop_pct"] == 0.0   # the loss-making price trail is OFF
    assert cond["trailing_arm_pct"] == 8.0


def test_vol_scaled_target_math_and_clamps():
    from investment_monitor.robo.config import ExitConfig

    cfg = ExitConfig()  # 1.5 sigma, 60d default horizon, band [8, 60]
    # Mega cap: 20% annualized vol over 60d -> sigma_60 ~8.1% -> target ~12.2%.
    mega = cfg.vol_target_pct(0.20, 60)
    assert 11.5 < mega < 13.0
    # Small cap: 50% vol -> a far larger, still-reachable target.
    small = cfg.vol_target_pct(0.50, 60)
    assert small > mega and small < 60.0
    # Clamped both ends; disabled/missing vol -> None (flat config target stands).
    assert cfg.vol_target_pct(0.001, 60) == cfg.target_floor_pct
    assert cfg.vol_target_pct(5.0, 60) == cfg.target_ceiling_pct
    assert cfg.vol_target_pct(None, 60) is None
    assert ExitConfig(vol_scaled_target=False).vol_target_pct(0.20, 60) is None


def test_with_vol_target_takes_the_earlier_target():
    from investment_monitor.robo.invalidation import with_vol_target

    # Explicit +20% vs vol-scaled +12% -> bank at the earlier one.
    assert with_vol_target({"profit_target_pct": 20.0}, 12.0)["profit_target_pct"] == 12.0
    # A vol target LOOSER than the thesis's own does not loosen it.
    assert with_vol_target({"profit_target_pct": 20.0}, 35.0)["profit_target_pct"] == 20.0
    # No explicit target -> the vol target supplies one; None is a no-op.
    assert with_vol_target({}, 12.0)["profit_target_pct"] == 12.0
    assert with_vol_target({"profit_target_pct": 20.0}, None) == {"profit_target_pct": 20.0}


def test_invalidation_floor_raises_noise_tight_stops():
    from investment_monitor.robo.config import InvalidationConfig

    icfg = InvalidationConfig()  # floor 8%
    # The LLM's 5% and 3% stops (32 live theses had 5%) are noise for an equity.
    assert icfg.floored({"price_drop_pct": 5.0})["price_drop_pct"] == 8.0
    assert icfg.floored({"price_drop_pct": 3.0})["price_drop_pct"] == 8.0
    # A wider stop is left alone, and other keys pass through untouched.
    kept = icfg.floored({"price_drop_pct": 20.0, "keywords": ["fraud"]})
    assert kept["price_drop_pct"] == 20.0 and kept["keywords"] == ["fraud"]
    # 0 disables the floor; absent/garbage keys are no-ops.
    assert InvalidationConfig(min_price_drop_pct=0).floored(
        {"price_drop_pct": 3.0})["price_drop_pct"] == 3.0
    assert icfg.floored({}) == {} and icfg.floored(None) == {}


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
    # Rate limit off: this test pins the plain LLM-update path (the limit has its own tests).
    evaluator = ThesisEvaluator(llm, _autonomous_config(max_conviction_delta_per_day=0.0))
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
        s.add(Price(ticker="RIDE", date=date.today(), close=11.0))  # gave back 25 of 35 pts
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "RIDE")
        assert evaluator.evaluate(s, t) == "exited"
        assert t.status == ThesisStatus.EXITED.value
        # The giveback trail is now the default mechanism (the price-distance trail is off).
        assert "trailing giveback" in t.conviction_history[-1]["trigger"]


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


# --------------------------------------------------------------------------- #
# H. Book hygiene: benching + weekly re-eval + cap (run_maintenance)
# --------------------------------------------------------------------------- #
def _history(*points):
    """[(days_ago, conviction), ...] -> conviction_history entries with timestamps."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return [
        {"ts": (now - timedelta(days=d)).isoformat(), "conviction": c, "trigger": "test"}
        for d, c in points
    ]


def _aged(days: float):
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)


def test_maintenance_benches_sustained_sub_floor(tmp_path):
    from investment_monitor.analysis.thesis_evaluator import run_maintenance

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(  # weeks of sub-floor conviction: pure LLM deadweight
            symbol="ZOMBIE", conviction=0.1, status=ThesisStatus.ACTIVE.value,
            created_at=_aged(30), conviction_history=_history((8, 0.1), (2, 0.1)),
        ))
        save_thesis(s, Thesis(  # strong name: untouched
            symbol="GOOD", conviction=0.8, status=ThesisStatus.ACTIVE.value,
            created_at=_aged(30), conviction_history=_history((2, 0.8)),
        ))
    cfg = _autonomous_config()
    with get_session() as s:
        counts = run_maintenance(s, ThesisEvaluator(None, cfg), cfg)
    assert counts["benched"] == 1
    with get_session() as s:
        assert get_thesis(s, "ZOMBIE").status == ThesisStatus.WATCH.value
        assert get_thesis(s, "GOOD").status == ThesisStatus.ACTIVE.value
        assert "benched" in get_thesis(s, "ZOMBIE").conviction_history[-1]["trigger"]


def test_maintenance_never_benches_fresh_or_recovering(tmp_path):
    from investment_monitor.analysis.thesis_evaluator import run_maintenance

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(  # sub-floor but only 2 days old — give it time
            symbol="FRESH", conviction=0.1, status=ThesisStatus.ACTIVE.value,
            created_at=_aged(2), conviction_history=_history((1, 0.1)),
        ))
        save_thesis(s, Thesis(  # showed real strength inside the window
            symbol="RECOV", conviction=0.1, status=ThesisStatus.ACTIVE.value,
            created_at=_aged(30),
            conviction_history=_history((6, 0.1), (5, 0.5), (4, 0.1), (3, 0.1),
                                        (2, 0.1), (1, 0.1)),
        ))
    cfg = _autonomous_config()
    with get_session() as s:
        counts = run_maintenance(s, ThesisEvaluator(None, cfg), cfg)
    assert counts["benched"] == 0


def test_maintenance_weekly_reeval_revives_recovered_bench(tmp_path):
    from investment_monitor.analysis.thesis_evaluator import run_maintenance

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(  # benched, due for its weekly look, conviction recovered
            symbol="BACK", conviction=0.6, status=ThesisStatus.WATCH.value,
            created_at=_aged(30), last_evaluated_at=_aged(8),
        ))
        save_thesis(s, Thesis(  # benched, looked at yesterday: stays skipped
            symbol="WAIT", conviction=0.6, status=ThesisStatus.WATCH.value,
            created_at=_aged(30), last_evaluated_at=_aged(1),
        ))
    cfg = _autonomous_config()
    with get_session() as s:
        counts = run_maintenance(s, ThesisEvaluator(None, cfg), cfg)
    assert counts["revived"] == 1 and counts["skipped_benched"] == 1
    with get_session() as s:
        assert get_thesis(s, "BACK").status == ThesisStatus.ACTIVE.value
        assert get_thesis(s, "WAIT").status == ThesisStatus.WATCH.value


def test_maintenance_cap_benches_weakest(tmp_path):
    from investment_monitor.analysis.thesis_evaluator import run_maintenance
    from investment_monitor.robo.config import AutonomyConfig

    init_db(tmp_path / "t.db")
    with get_session() as s:
        for sym, conv in [("AA", 0.9), ("BB", 0.8), ("CC", 0.7), ("DD", 0.6)]:
            save_thesis(s, Thesis(symbol=sym, conviction=conv,
                                  status=ThesisStatus.ACTIVE.value, created_at=_aged(1)))
    cfg = _autonomous_config().model_copy(
        update={"autonomy": AutonomyConfig(max_active_theses=2)}
    )
    with get_session() as s:
        counts = run_maintenance(s, ThesisEvaluator(None, cfg), cfg)
    assert counts["benched"] == 2
    with get_session() as s:
        statuses = {t.symbol: t.status for t in
                    [get_thesis(s, x) for x in ("AA", "BB", "CC", "DD")]}
    assert statuses == {"AA": "active", "BB": "active", "CC": "watch", "DD": "watch"}


def test_benched_thesis_gets_no_weight(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="LIVE", conviction=0.9, status=ThesisStatus.ACTIVE.value))
        save_thesis(s, Thesis(symbol="BENCH", conviction=0.9, status=ThesisStatus.WATCH.value))
    with get_session() as s:
        weights = compute_conviction_weights(s, _autonomous_config())
    assert weights.get("LIVE", 0) > 0
    assert "BENCH" not in weights  # benched = tracked but never sized


# --------------------------------------------------------------------------- #
# H. Conviction-noise guards (2026-07 live churn root cause):
#    evidence-gated re-evals, per-day conviction rate limit, exit dwell +
#    sub-floor re-entry cooldown. Live, phi3:mini anchored on the prompt's
#    "Current conviction" and walked held names ±0.05/eval through the 0.35
#    floor and back (LLY 0.95->0.25->0.94 over six days, sold and re-bought).
# --------------------------------------------------------------------------- #
def _hist_entry(days_ago: float, conviction: float, trigger: str = "test", **extra):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return {"ts": (now - timedelta(days=days_ago)).isoformat(),
            "conviction": conviction, "trigger": trigger, **extra}


def _guard_thesis(history):
    from types import SimpleNamespace

    return SimpleNamespace(conviction_history=history)


def test_exit_dwell_pure():
    from datetime import datetime, timezone

    from investment_monitor.robo.sizing import in_exit_dwell

    now = datetime.now(timezone.utc)
    cfg = SizingConfig()  # exit_dwell_days=2 default
    # Floor-strength 1 day ago: the dip is fresh -> dwell holds.
    fresh_dip = _guard_thesis([_hist_entry(1.0, 0.9), _hist_entry(0.1, 0.2)])
    assert in_exit_dwell(fresh_dip, cfg, now)
    # Floor-strength last seen 3 days ago: dwell over -> exit proceeds.
    stale_dip = _guard_thesis([_hist_entry(3.0, 0.9), _hist_entry(0.1, 0.2)])
    assert not in_exit_dwell(stale_dip, cfg, now)
    # Timestamp-less birth entry (fresh promotion) never grants a dwell.
    birth_only = _guard_thesis([{"conviction": 0.9, "trigger": "generated"}])
    assert not in_exit_dwell(birth_only, cfg, now)
    # 0 disables.
    assert not in_exit_dwell(fresh_dip, SizingConfig(exit_dwell_days=0.0), now)


def test_reentry_cooldown_pure():
    from datetime import datetime, timezone

    from investment_monitor.robo.sizing import in_reentry_cooldown

    now = datetime.now(timezone.utc)
    cfg = SizingConfig()  # sub_floor_reentry_days=3 default
    # Walked through the floor yesterday, back above today: still cooling off.
    recent_walk = _guard_thesis([_hist_entry(1.0, 0.2), _hist_entry(0.1, 0.9)])
    assert in_reentry_cooldown(recent_walk, cfg, now)
    # The sub-floor point is outside the window: eligible again.
    old_walk = _guard_thesis([_hist_entry(5.0, 0.2), _hist_entry(0.1, 0.9)])
    assert not in_reentry_cooldown(old_walk, cfg, now)
    # A fresh confluence signal AFTER the dip overrides the cooldown (revival invariant).
    confluence = _guard_thesis([
        _hist_entry(1.0, 0.2),
        _hist_entry(0.5, 0.9, trigger="confluence-revival:insider_cluster"),
    ])
    assert not in_reentry_cooldown(confluence, cfg, now)
    # 0 disables.
    assert not in_reentry_cooldown(recent_walk, SizingConfig(sub_floor_reentry_days=0.0), now)


def test_conviction_weights_dwell_freezes_held_dip(tmp_path):
    # A HELD name whose smoothed conviction just dipped sub-floor keeps its CURRENT
    # weight (zero trades) instead of full-exiting; unheld names still drop out.
    init_db(tmp_path / "t.db")
    dip_history = [_hist_entry(1.5, 0.5), _hist_entry(0.5, 0.2), _hist_entry(0.1, 0.2)]
    with get_session() as s:
        save_thesis(s, Thesis(symbol="DIP", conviction=0.2, status=ThesisStatus.ACTIVE.value,
                              conviction_history=dip_history))
    cfg = _autonomous_config()
    with get_session() as s:
        held = compute_conviction_weights(
            s, cfg, held_symbols={"DIP"}, held_weights={"DIP": 0.11})
        unheld = compute_conviction_weights(s, cfg)
    assert held["DIP"] == 0.11        # frozen at the actual held weight
    assert "DIP" not in unheld        # never held -> no dwell, no capital


def test_conviction_weights_dwell_expires_to_full_exit(tmp_path):
    # Same dip, but floor-strength was last seen beyond exit_dwell_days: sell fires.
    init_db(tmp_path / "t.db")
    stale_history = [_hist_entry(3.0, 0.5), _hist_entry(1.0, 0.2), _hist_entry(0.1, 0.2)]
    with get_session() as s:
        save_thesis(s, Thesis(symbol="DIP", conviction=0.2, status=ThesisStatus.ACTIVE.value,
                              conviction_history=stale_history))
    with get_session() as s:
        weights = compute_conviction_weights(
            s, _autonomous_config(), held_symbols={"DIP"}, held_weights={"DIP": 0.11})
    assert "DIP" not in weights


def test_conviction_weights_reentry_cooldown_blocks_new_position(tmp_path):
    # Conviction recovered above the floor, but the name walked sub-floor 2 days ago:
    # no NEW position yet. Held names are exempt (they were never sold).
    init_db(tmp_path / "t.db")
    walk_history = [_hist_entry(2.0, 0.2), _hist_entry(1.0, 0.55), _hist_entry(0.1, 0.9)]
    with get_session() as s:
        save_thesis(s, Thesis(symbol="WALK", conviction=0.9, status=ThesisStatus.ACTIVE.value,
                              conviction_history=walk_history))
    cfg = _autonomous_config()
    with get_session() as s:
        as_new = compute_conviction_weights(s, cfg)
        as_held = compute_conviction_weights(s, cfg, held_symbols={"WALK"})
    assert "WALK" not in as_new
    assert as_held.get("WALK", 0) > 0


def test_conviction_weights_dwell_reserves_topn_slot(tmp_path):
    # The dwell stub is tiny by construction; a challenger must NOT win its slot on
    # size during the window — the rotation decision is deferred, not delegated.
    from investment_monitor.robo.config import RoboCaps

    init_db(tmp_path / "t.db")
    dip_history = [_hist_entry(1.0, 0.5), _hist_entry(0.1, 0.2)]
    with get_session() as s:
        save_thesis(s, Thesis(symbol="DIP", conviction=0.2, status=ThesisStatus.ACTIVE.value,
                              conviction_history=dip_history))
        save_thesis(s, Thesis(symbol="BIG1", conviction=0.95, status=ThesisStatus.ACTIVE.value))
        save_thesis(s, Thesis(symbol="BIG2", conviction=0.94, status=ThesisStatus.ACTIVE.value))
    cfg = _autonomous_config().model_copy(update={"caps": RoboCaps(
        max_order_pct=0.5, max_orders_per_run=10, max_orders_per_day=20, max_positions=2)})
    with get_session() as s:
        weights = compute_conviction_weights(
            s, cfg, held_symbols={"DIP"}, held_weights={"DIP": 0.05})
    assert weights["DIP"] == 0.05                      # slot reserved for the dwell
    assert ("BIG1" in weights) and ("BIG2" not in weights)  # one free slot left


def test_evaluator_skips_reeval_on_unchanged_evidence(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value))
    llm = _FakeLLM('{"narrative": "same story", "conviction": 0.6}')
    evaluator = ThesisEvaluator(llm, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "VOO")
        assert evaluator.evaluate(s, t) == "updated"       # first look: hash recorded
        assert t.conviction_history[-1].get("evidence_hash")
        n_points = len(t.conviction_history)
        assert evaluator.evaluate(s, t) == "unchanged_evidence"  # nothing new -> no LLM noise
        assert t.conviction == 0.6
        assert len(t.conviction_history) == n_points       # and no history spam
    # Master switch off -> the old always-re-eval behavior returns.
    from investment_monitor.robo.config import AutonomyConfig

    permissive = _autonomous_config().model_copy(
        update={"autonomy": AutonomyConfig(skip_reeval_unchanged_evidence=False)})
    with get_session() as s:
        t = get_thesis(s, "VOO")
        assert ThesisEvaluator(llm, permissive).evaluate(s, t) == "updated"


def test_evaluator_rate_limits_conviction_move(tmp_path):
    # A collapse bigger than max_conviction_delta_per_day is clamped against the
    # 24h-ago baseline, and the clamped update withholds its evidence hash so the
    # NEXT cycle re-runs (keeps stepping) instead of skipping as "unchanged".
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, narrative="old",
                              status=ThesisStatus.ACTIVE.value,
                              conviction_history=[_hist_entry(1.5, 0.9)]))
    llm = _FakeLLM('{"narrative": "suddenly bearish", "conviction": 0.2}')
    evaluator = ThesisEvaluator(llm, _autonomous_config())  # default cap 0.15/day
    with get_session() as s:
        t = get_thesis(s, "VOO")
        assert evaluator.evaluate(s, t) == "updated"
        assert abs(t.conviction - 0.75) < 1e-9             # 0.9 - 0.15, not 0.2
        assert not t.conviction_history[-1].get("evidence_hash")
        assert evaluator.evaluate(s, t) == "updated"       # clamped -> next eval re-runs
        assert abs(t.conviction - 0.75) < 1e-9             # still bounded by the 24h baseline


def test_evaluator_banks_a_ten_pct_winner_end_to_end(tmp_path):
    # The live complaint, end to end: a position peaks at +12% and fades. It must now
    # EXIT AT A PROFIT rather than ride down into a stop-loss.
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="WINR", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 100.0, "fill_cost": 100.0},
            invalidation_conditions={"price_drop_pct": 5.0},
            high_water_mark=112.0,            # peaked +12%
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="WINR", date=date.today(), close=107.0))  # faded to +7%
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "WINR")
        assert evaluator.evaluate(s, t) == "exited"
        assert t.status == ThesisStatus.EXITED.value
        trigger = t.conviction_history[-1]["trigger"]
        assert "trailing giveback" in trigger and "+7.0%" in trigger


def test_evaluator_noise_tight_stop_no_longer_invalidates(tmp_path):
    # A 5% stop used to realize a loss on ordinary two-week noise; the 8% floor holds
    # the position. A genuine break past the floor still invalidates.
    init_db(tmp_path / "t.db")
    for sym, close in (("NOISE", 94.0), ("BROKE", 88.0)):
        with get_session() as s:
            save_thesis(s, Thesis(
                symbol=sym, conviction=0.9, status=ThesisStatus.ACTIVE.value,
                entry_conditions={"entry_price": 100.0},
                invalidation_conditions={"price_drop_pct": 5.0},
            ))
        from investment_monitor.storage import Price
        with get_session() as s:
            s.add(Price(ticker=sym, date=date.today(), close=close))
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        assert evaluator.evaluate(s, get_thesis(s, "NOISE")) != "invalidated"   # -6%: held
        assert evaluator.evaluate(s, get_thesis(s, "BROKE")) == "invalidated"   # -12%: real


def test_evaluator_vol_scaled_target_fires_on_a_mega_cap(tmp_path):
    # A +20% thesis target is unreachable for a low-vol name; the vol-scaled target
    # (~12% at 20% annualized vol over 60d) banks the move instead.
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="MEGA", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 100.0, "fill_cost": 100.0},
            exit_conditions={"profit_target_pct": 20.0, "max_hold_days": 60.0},
        ))
        _seed_sim(s, "MEGA", drift=0.10, vol=0.20, cvar=-0.10)
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="MEGA", date=date.today(), close=113.0))  # +13%: under the 20%
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        t = get_thesis(s, "MEGA")
        assert evaluator.evaluate(s, t) == "exited"
        assert "profit target" in t.conviction_history[-1]["trigger"]


def test_evaluator_vol_target_absent_sim_keeps_flat_target(tmp_path):
    # No simulation -> the explicit thesis target stands unchanged (fail-open).
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="NOSIM", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 100.0, "fill_cost": 100.0},
            exit_conditions={"profit_target_pct": 20.0},
        ))
    from investment_monitor.storage import Price
    with get_session() as s:
        s.add(Price(ticker="NOSIM", date=date.today(), close=113.0))
    evaluator = ThesisEvaluator(None, _autonomous_config())
    with get_session() as s:
        assert evaluator.evaluate(s, get_thesis(s, "NOSIM")) != "exited"  # +13% < 20%


def test_evaluator_rate_limit_zero_disables(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, narrative="old",
                              status=ThesisStatus.ACTIVE.value,
                              conviction_history=[_hist_entry(1.5, 0.9)]))
    llm = _FakeLLM('{"narrative": "suddenly bearish", "conviction": 0.2}')
    evaluator = ThesisEvaluator(llm, _autonomous_config(max_conviction_delta_per_day=0.0))
    with get_session() as s:
        t = get_thesis(s, "VOO")
        assert evaluator.evaluate(s, t) == "updated"
        assert abs(t.conviction - 0.2) < 1e-9              # unclamped
