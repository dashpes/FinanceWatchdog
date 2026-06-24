"""Tests for the robo advisor's feedback loop (Phase 6).

Covers:
  A. PURE metrics + multiplier (no I/O).
  B. Ledger write + EWMA-smoothed accuracy stats (in-memory SQLite).
  C. Sizing tilt: a poor track record dampens size; disabling the loop restores it;
     no outcomes => byte-identical to no-learning.
  D. Evaluator: outcome capture on re-eval, invalidation still short-circuits the
     LLM while capturing the final data point, and the prompt gains the outcome block
     only when there is a real track record to show.
"""

from __future__ import annotations

import string
from datetime import date

from investment_monitor.analysis.thesis_evaluator import ThesisEvaluator
from investment_monitor.analysis.thesis_prompts import (
    THESIS_UPDATE_PROMPT,
    THESIS_UPDATE_PROMPT_WITH_OUTCOME,
)
from investment_monitor.robo.config import LearningConfig, RoboCaps, RoboConfig, SizingConfig
from investment_monitor.robo.sizing import accuracy_multiplier, compute_conviction_weights
from investment_monitor.storage import (
    Price,
    SimulationResult,
    Thesis,
    ThesisStatus,
    accuracy_stats_for_symbol,
    get_session,
    get_thesis,
    init_db,
    outcome_metrics,
    record_thesis_outcome,
    save_thesis,
)


# --------------------------------------------------------------------------- #
# A. Pure metrics + multiplier
# --------------------------------------------------------------------------- #
def test_outcome_metrics_sign_and_brier():
    # Confident + wrong => direction 0, large brier.
    assert outcome_metrics(-0.40, 0.9) == (0, 0.81)
    # Confident + right => direction 1, small brier.
    assert outcome_metrics(0.20, 0.9) == (1, (0.9 - 1) ** 2)
    # Conviction is clamped before scoring.
    assert outcome_metrics(0.1, 1.5)[1] == 0.0


def test_accuracy_multiplier_neutral_until_min_samples():
    # No data and thin samples => exactly 1.0 (no tilt).
    assert accuracy_multiplier({"n": 0}, accuracy_weight=0.5, floor=0.5, ceiling=1.0, min_samples=6) == 1.0
    assert accuracy_multiplier(
        {"n": 5, "ewma_hit_rate": 0.0}, accuracy_weight=0.5, floor=0.5, ceiling=1.0, min_samples=6
    ) == 1.0


def test_accuracy_multiplier_dampens_poor_and_clamps():
    poor = accuracy_multiplier(
        {"n": 8, "ewma_hit_rate": 0.2}, accuracy_weight=0.5, floor=0.5, ceiling=1.0, min_samples=6
    )
    assert abs(poor - 0.7) < 1e-9  # 1 + 0.5*(0.2-0.5)*2
    # Worst case is clamped to the floor, never below.
    floored = accuracy_multiplier(
        {"n": 8, "ewma_hit_rate": 0.0}, accuracy_weight=1.0, floor=0.5, ceiling=1.0, min_samples=6
    )
    assert floored == 0.5


def test_accuracy_multiplier_shrink_only_by_default():
    # A great track record cannot inflate exposure when ceiling == 1.0.
    assert accuracy_multiplier(
        {"n": 8, "ewma_hit_rate": 0.95}, accuracy_weight=0.5, floor=0.5, ceiling=1.0, min_samples=6
    ) == 1.0
    # But it can when an operator opts into amplification.
    amp = accuracy_multiplier(
        {"n": 8, "ewma_hit_rate": 1.0}, accuracy_weight=0.5, floor=0.5, ceiling=1.25, min_samples=6
    )
    assert abs(amp - 1.5) < 1e-9 or amp == 1.25  # 1.5 clamped to ceiling 1.25
    assert amp == 1.25


# --------------------------------------------------------------------------- #
# B. Ledger + accuracy stats
# --------------------------------------------------------------------------- #
def test_outcome_roundtrip_and_stats(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.8, realized_return=0.10)
        record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.8, realized_return=0.05)
        record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.8, realized_return=-0.20)
    with get_session() as s:
        stats = accuracy_stats_for_symbol(s, "VOO")
        assert stats["n"] == 3
        assert abs(stats["hit_rate"] - 2 / 3) < 1e-9  # 2 wins / 3
        # An unseen symbol returns the neutral prior.
        assert accuracy_stats_for_symbol(s, "NONE") == {
            "n": 0, "hit_rate": 0.5, "brier": 0.25, "ewma_hit_rate": 0.5
        }


def test_accuracy_stats_ewma_weights_recent_more(tmp_path):
    init_db(tmp_path / "t.db")
    with get_session() as s:
        for _ in range(5):  # older: wins
            record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.7, realized_return=0.10)
        for _ in range(5):  # newer: losses
            record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.7, realized_return=-0.10)
    with get_session() as s:
        stats = accuracy_stats_for_symbol(s, "VOO", ewma_halflife=2.0)
    assert abs(stats["hit_rate"] - 0.5) < 1e-9          # raw is balanced
    assert stats["ewma_hit_rate"] < 0.5                  # recent losses dominate


# --------------------------------------------------------------------------- #
# C. Sizing tilt
# --------------------------------------------------------------------------- #
def _autonomous_config(learning: LearningConfig | None = None, **sizing_kw) -> RoboConfig:
    return RoboConfig(
        mode="autonomous",
        target_allocation={},
        allowlist=[],
        sizing=SizingConfig(**sizing_kw),
        caps=RoboCaps(max_order_pct=0.5, max_orders_per_run=10, max_orders_per_day=20),
        learning=learning or LearningConfig(),
    )


def _seed_voo(db):
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
        s.add(SimulationResult(
            ticker="VOO", run_date=date.today(), entry_price=500.0, composite_score=80.0,
            num_simulations=1000, lookback_days=252, volatility=0.18, drift=0.15,
            results_90d={"base_var_95": -0.15, "base_cvar_95": -0.20},
        ))


def test_poor_track_record_dampens_size(tmp_path):
    db = tmp_path / "t.db"
    _seed_voo(db)
    # Baseline weight with no outcomes yet (multiplier == 1.0).
    with get_session() as s:
        w_base = compute_conviction_weights(s, _autonomous_config())["VOO"]
    assert w_base > 0
    # Record a losing track record, then re-size: the poor name must shrink.
    with get_session() as s:
        for _ in range(8):
            record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.9, realized_return=-0.10)
    with get_session() as s:
        w_learned = compute_conviction_weights(s, _autonomous_config())["VOO"]
    assert 0 < w_learned < w_base
    # Disabling the tilt restores the baseline even with the same losing history.
    off = LearningConfig(accuracy_sizing=False)
    with get_session() as s:
        w_off = compute_conviction_weights(s, _autonomous_config(learning=off))["VOO"]
    assert abs(w_off - w_base) < 1e-12


def test_no_outcomes_is_byte_identical(tmp_path):
    # With the loop ON but no ledger data, weights equal the loop-OFF weights.
    db = tmp_path / "t.db"
    _seed_voo(db)
    with get_session() as s:
        on = compute_conviction_weights(s, _autonomous_config())
    _seed_voo(tmp_path / "t2.db")
    with get_session() as s:
        off = compute_conviction_weights(s, _autonomous_config(learning=LearningConfig(enabled=False)))
    assert on == off


# --------------------------------------------------------------------------- #
# D. Evaluator integration
# --------------------------------------------------------------------------- #
class _RecordingLLM:
    model = "fake"

    def __init__(self, text):
        self._text = text
        self.client = self
        self.prompts: list[str] = []

    def is_available(self):
        return True

    def generate(self, model, prompt, options):
        self.prompts.append(prompt)
        return {"response": self._text}


def test_evaluator_captures_outcome_and_injects_block(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
        s.add(Price(ticker="VOO", date=date.today(), close=520.0))  # +4%, no invalidation
    llm = _RecordingLLM('{"narrative": "Holding up", "conviction": 0.6}')
    # min_days_held=0 so the brand-new thesis is captured/shown immediately.
    evaluator = ThesisEvaluator(llm, _autonomous_config(learning=LearningConfig(min_days_held=0)))
    with get_session() as s:
        t = get_thesis(s, "VOO")
        action = evaluator.evaluate(s, t)
        assert action == "updated"
    # The realized outcome was recorded (a win: 520 > 500).
    with get_session() as s:
        stats = accuracy_stats_for_symbol(s, "VOO")
        assert stats["n"] == 1 and stats["hit_rate"] == 1.0
    # The prompt carried the compact realized-performance line...
    assert llm.prompts and "Realized performance & track record:" in llm.prompts[0]
    assert "opened $500.00" in llm.prompts[0]
    # ...but NOT the aggregate track-record sentence (n=1 < min_samples).
    assert "Track record (last" not in llm.prompts[0]


def test_evaluator_base_prompt_when_no_entry_price(tmp_path):
    # No entry price => no outcome block => byte-for-byte the base prompt.
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        t = get_thesis(s, "VOO")
        ThesisEvaluator(llm, _autonomous_config()).evaluate(s, t)
    assert llm.prompts and "Realized performance & track record:" not in llm.prompts[0]


def test_evaluator_invalidation_still_short_circuits_with_capture(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(
            symbol="VOO", conviction=0.9, status=ThesisStatus.ACTIVE.value,
            entry_conditions={"entry_price": 500.0},
            invalidation_conditions={"price_drop_pct": 25},
        ))
        s.add(Price(ticker="VOO", date=date.today(), close=300.0))  # -40% from entry
    llm = _RecordingLLM('{"narrative": "still bullish!", "conviction": 0.95}')
    evaluator = ThesisEvaluator(llm, _autonomous_config(learning=LearningConfig(min_days_held=0)))
    with get_session() as s:
        t = get_thesis(s, "VOO")
        action = evaluator.evaluate(s, t)
        assert action == "invalidated"
        assert t.conviction == 0.0 and t.status == ThesisStatus.INVALIDATED.value
    assert llm.prompts == []  # deterministic rail short-circuited the LLM
    # The final, most-informative data point (confident + big loss) was still captured.
    with get_session() as s:
        stats = accuracy_stats_for_symbol(s, "VOO")
        assert stats["n"] == 1 and stats["hit_rate"] == 0.0


# --------------------------------------------------------------------------- #
# E. Sampling hygiene (per-day dedupe, holding age, flat-return guard)
# --------------------------------------------------------------------------- #
def _seed_voo_priced(db, close):
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
        s.add(Price(ticker="VOO", date=date.today(), close=close))


def test_one_outcome_per_symbol_per_day(tmp_path):
    # Re-evaluating the same thesis twice in a day records ONE outcome, not two —
    # so intraday runs can't flood the window with autocorrelated copies.
    _seed_voo_priced(tmp_path / "t.db", 520.0)
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    for _ in range(2):
        with get_session() as s:
            ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    with get_session() as s:
        assert accuracy_stats_for_symbol(s, "VOO")["n"] == 1


def test_min_days_held_gates_capture_and_block(tmp_path):
    # A brand-new (0-day) thesis under the default min_days_held=2 records nothing
    # and the prompt stays the byte-for-byte base template.
    _seed_voo_priced(tmp_path / "t.db", 520.0)
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        ThesisEvaluator(llm, _autonomous_config()).evaluate(s, get_thesis(s, "VOO"))
    with get_session() as s:
        assert accuracy_stats_for_symbol(s, "VOO")["n"] == 0
    assert "Realized performance & track record:" not in llm.prompts[0]


def test_flat_return_not_recorded(tmp_path):
    # A flat position (current == entry) is not a directional loss; it is skipped.
    _seed_voo_priced(tmp_path / "t.db", 500.0)  # 0% move
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    with get_session() as s:
        assert accuracy_stats_for_symbol(s, "VOO")["n"] == 0


def test_track_record_line_appears_past_min_samples(tmp_path):
    _seed_voo_priced(tmp_path / "t.db", 520.0)
    with get_session() as s:
        for _ in range(6):
            record_thesis_outcome(s, symbol="VOO", conviction_at_eval=0.6, realized_return=0.05)
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    assert "Track record (last" in llm.prompts[0]  # >= min_samples: aggregate shown


# --------------------------------------------------------------------------- #
# F. Boundary + prompt-parity
# --------------------------------------------------------------------------- #
def test_accuracy_multiplier_activates_exactly_at_min_samples():
    base = dict(accuracy_weight=1.0, floor=0.5, ceiling=1.0, min_samples=6)
    # n == min_samples - 1: still exactly neutral (no tilt).
    assert accuracy_multiplier({"n": 5, "ewma_hit_rate": 0.0}, **base) == 1.0
    # n == min_samples: the tilt engages (poor record -> floor).
    assert accuracy_multiplier({"n": 6, "ewma_hit_rate": 0.0}, **base) == 0.5


def test_with_outcome_prompt_placeholder_parity():
    def _ph(t):
        return {fn for _, fn, _, _ in string.Formatter().parse(t) if fn}
    assert _ph(THESIS_UPDATE_PROMPT_WITH_OUTCOME) == _ph(THESIS_UPDATE_PROMPT) | {"outcome_block"}
    # And it renders with exactly the kwargs the evaluator passes (no KeyError).
    THESIS_UPDATE_PROMPT_WITH_OUTCOME.format(
        symbol="X", narrative="n", conviction="0.50", score_block="s",
        news_block="nw", signals_block="sig", outcome_block="ob",
    )


# --------------------------------------------------------------------------- #
# G. Fail-open: a raising learning path degrades to a neutral no-op
# --------------------------------------------------------------------------- #
def _raise(*_a, **_k):
    raise RuntimeError("boom")


def test_outcome_capture_failure_does_not_break_evaluate(tmp_path, monkeypatch):
    _seed_voo_priced(tmp_path / "t.db", 520.0)
    import investment_monitor.analysis.thesis_evaluator as te
    monkeypatch.setattr(te, "record_thesis_outcome", _raise)
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        action = ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    assert action == "updated"  # the maintenance loop survives a ledger outage


def test_accuracy_multiplier_db_failure_is_neutral(tmp_path, monkeypatch):
    _seed_voo(tmp_path / "t.db")
    import investment_monitor.robo.sizing as sz
    monkeypatch.setattr(sz, "accuracy_stats_for_symbol", _raise)
    with get_session() as s:
        w = compute_conviction_weights(s, _autonomous_config())["VOO"]
    assert w > 0  # sizing unbroken; multiplier fell back to 1.0


def test_outcome_block_failure_falls_back_to_base_prompt(tmp_path, monkeypatch):
    _seed_voo_priced(tmp_path / "t.db", 520.0)
    import investment_monitor.analysis.thesis_evaluator as te
    monkeypatch.setattr(te, "accuracy_stats_for_symbol", _raise)
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    # The aggregate read blew up -> the whole block degrades to "" -> base prompt.
    assert llm.prompts and "Realized performance & track record:" not in llm.prompts[0]


# --------------------------------------------------------------------------- #
# F. Fill-cost reconciliation: the loop scores against the real broker fill
#    (P&L pulled from the broker), not the quote captured when the idea opened.
# --------------------------------------------------------------------------- #
def test_fill_cost_preferred_over_entry_price_for_outcome(tmp_path):
    # entry_price (idea quote) implies a WIN; the real fill cost implies a LOSS.
    # The recorded outcome — and the displayed line — must follow the fill cost.
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", conviction=0.5, narrative="old",
                              status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0, "fill_cost": 600.0}))
        s.add(Price(ticker="VOO", date=date.today(), close=560.0))  # +12% vs quote, -6.7% vs fill
    cfg = _autonomous_config(learning=LearningConfig(min_days_held=0))
    llm = _RecordingLLM('{"narrative": "x", "conviction": 0.6}')
    with get_session() as s:
        ThesisEvaluator(llm, cfg).evaluate(s, get_thesis(s, "VOO"))
    with get_session() as s:
        stats = accuracy_stats_for_symbol(s, "VOO")
        assert stats["n"] == 1 and stats["hit_rate"] == 0.0  # a loss against the real fill
    assert "opened $600.00" in llm.prompts[0]  # the line shows the fill cost, not $500


def test_reconcile_fill_costs_writes_broker_basis(tmp_path):
    # fill_cost must be the thesis's OWN entry-order fill price (450, from the
    # reconciled BUY), NOT the broker's blended unit cost (999 here) — the blend
    # folds in pre-existing/independent shares and would bias calibration.
    from datetime import datetime, timedelta
    from decimal import Decimal

    from investment_monitor.robo.models import AccountState, OrderSide, Position
    from investment_monitor.robo.rebalance import _reconcile_fill_costs
    from investment_monitor.storage import RoboOrder

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", account_id="ACC", conviction=0.5,
                              status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
        thesis_created = get_thesis(s, "VOO").created_at
        # The robo's own opening BUY, reconciled to a real fill at $450 AFTER the idea.
        s.add(RoboOrder(run_id="R1", symbol="VOO", side=OrderSide.BUY.value,
                        order_type="MARKET", quantity=2.0, placed=True,
                        broker_order_id="O1", fill_price=450.0, fill_quantity=2.0,
                        fill_status="FILLED",
                        created_at=thesis_created + timedelta(minutes=1)))
    account = AccountState(
        account_id="ACC", is_cash_account=True, has_margin=False, settled_cash=Decimal("10"),
        positions=[Position(symbol="VOO", quantity=Decimal("2"),
                            price=Decimal("560"), unit_cost=Decimal("999"))],
    )
    with get_session() as s:
        _reconcile_fill_costs(s, account)
    with get_session() as s:
        cond = get_thesis(s, "VOO").entry_conditions
        assert cond["fill_cost"] == 450.0       # entry-order fill, not blended cost
        assert cond["entry_price"] == 500.0     # idea quote left intact


def test_reconcile_fill_costs_noop_without_basis(tmp_path):
    # Paper / no broker cost basis => nothing written (path stays byte-identical).
    from decimal import Decimal

    from investment_monitor.robo.models import AccountState, Position
    from investment_monitor.robo.rebalance import _reconcile_fill_costs

    init_db(tmp_path / "t.db")
    with get_session() as s:
        save_thesis(s, Thesis(symbol="VOO", account_id="ACC", conviction=0.5,
                              status=ThesisStatus.ACTIVE.value,
                              entry_conditions={"entry_price": 500.0}))
    account = AccountState(
        account_id="ACC", is_cash_account=True, has_margin=False, settled_cash=Decimal("10"),
        positions=[Position(symbol="VOO", quantity=Decimal("2"), price=Decimal("560"))],
    )
    with get_session() as s:
        _reconcile_fill_costs(s, account)
    with get_session() as s:
        assert "fill_cost" not in (get_thesis(s, "VOO").entry_conditions or {})
