"""Tests for Phase 4 — autonomous selection + the additive gate guards.

The gate guards are the safety-critical part: each must (a) fire correctly when
enabled, (b) always permit SELLS so positions can be exited, and (c) stay inert
under default/permissive config so rebalance mode is unaffected.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from investment_monitor.robo.config import AutonomyConfig, RoboCaps, RoboConfig
from investment_monitor.robo.gate import validate, validate_orders
from investment_monitor.robo.market_hours import is_market_open
from investment_monitor.robo.models import (
    AccountState,
    OrderSide,
    OrderType,
    Position,
    ProposedOrder,
)
from investment_monitor.robo.promotion import promote_candidates
from investment_monitor.storage import (
    StockCandidate,
    Thesis,
    ThesisStatus,
    get_session,
    get_thesis,
    init_db,
    save_thesis,
)


def _acct(cash="100", positions=None) -> AccountState:
    return AccountState(
        account_id="A", account_type="B", is_cash_account=True, has_margin=False,
        settled_cash=Decimal(cash), positions=positions or [],
    )


def _cfg(**caps) -> RoboConfig:
    return RoboConfig(
        target_allocation={"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2},
        allowlist=["VOO", "SCHD", "AAPL"],
        caps=RoboCaps(**caps),
    )


def _buy(symbol, notional="10"):
    return ProposedOrder(symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET,
                         notional=Decimal(notional))


def _sell(symbol, notional="10"):
    return ProposedOrder(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
                         notional=Decimal(notional))


# --------------------------------------------------------------------------- #
# Gate guards
# --------------------------------------------------------------------------- #
def test_no_active_thesis_blocks_buy_allows_sell():
    cfg = _cfg()
    # BUY of a name with no live thesis -> rejected (autonomous).
    d = validate(_buy("AAPL"), _acct(), cfg, price=Decimal("200"), active_symbols={"VOO"})
    assert not d.accepted and d.code == "no_active_thesis"
    # BUY of a name WITH a thesis -> passes that check.
    assert validate(_buy("VOO"), _acct(), cfg, price=Decimal("500"), active_symbols={"VOO"}).accepted
    # SELL is always allowed, even without a thesis (exit path).
    held = [Position(symbol="AAPL", quantity=Decimal("1"), price=Decimal("200"))]
    d3 = validate(_sell("AAPL"), _acct("100", held), cfg, price=Decimal("200"), active_symbols={"VOO"})
    assert d3.accepted


def test_no_active_thesis_inert_in_rebalance_mode():
    # active_symbols=None (rebalance mode) -> guard skipped entirely.
    assert validate(_buy("AAPL"), _acct(), _cfg(), price=Decimal("200")).accepted


def test_drawdown_breaker_halts_buys_allows_sells():
    cfg = _cfg()
    assert validate(_buy("VOO"), _acct(), cfg, price=Decimal("500"), halt_buys=True).code == "drawdown_breaker"
    held = [Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500"))]
    assert validate(_sell("VOO"), _acct("100", held), cfg, price=Decimal("500"), halt_buys=True).accepted


def test_per_name_cap():
    cfg = _cfg(max_per_name_weight=0.1, max_order_pct=1.0)  # total 100 -> $10 cap per name
    assert validate(_buy("VOO", "20"), _acct("100"), cfg, price=Decimal("500")).code == "exceeds_per_name_cap"
    assert validate(_buy("VOO", "5"), _acct("100"), cfg, price=Decimal("500")).accepted


def test_per_name_cap_counts_existing_holding():
    cfg = _cfg(max_per_name_weight=0.2, max_order_pct=1.0)
    held = [Position(symbol="VOO", quantity=Decimal("0.03"), price=Decimal("500"))]  # $15 held
    acct = _acct("100", held)  # total = 115, cap = 23
    # already $15; buying $10 more -> $25 > $23 cap.
    assert validate(_buy("VOO", "10"), acct, cfg, price=Decimal("500")).code == "exceeds_per_name_cap"


def test_max_positions_blocks_new_name_allows_existing():
    cfg = _cfg(max_positions=2, max_order_pct=1.0)
    held = [
        Position(symbol="VOO", quantity=Decimal("0.1"), price=Decimal("500")),
        Position(symbol="SCHD", quantity=Decimal("0.1"), price=Decimal("80")),
    ]
    acct = _acct("100", held)
    assert validate(_buy("AAPL", "5"), acct, cfg, price=Decimal("200")).code == "max_positions"
    assert validate(_buy("VOO", "5"), acct, cfg, price=Decimal("500")).accepted  # adding to existing is fine


def test_turnover_budget_threaded_in_batch():
    cfg = _cfg(max_turnover_pct=0.1, max_order_pct=1.0)  # total 100 -> $10 turnover budget
    orders = [_buy("VOO", "6"), _buy("SCHD", "6")]
    decisions = validate_orders(orders, _acct("100"), cfg,
                                {"VOO": Decimal("500"), "SCHD": Decimal("80")})
    assert decisions[0].accepted
    assert not decisions[1].accepted and decisions[1].code == "exceeds_turnover"


def test_max_positions_threaded_across_batch():
    # Cap of 1 new name: two new-name buys -> only the first opens a slot.
    cfg = _cfg(max_positions=1, max_order_pct=1.0)
    orders = [_buy("VOO", "5"), _buy("SCHD", "5")]  # both new names
    decisions = validate_orders(orders, _acct("100"), cfg,
                                {"VOO": Decimal("500"), "SCHD": Decimal("80")})
    assert decisions[0].accepted
    assert not decisions[1].accepted and decisions[1].code == "max_positions"


def test_guards_inert_by_default():
    # Default caps -> none of the new guards fire.
    cfg = _cfg()
    d = validate(_buy("VOO", "10"), _acct("100"), cfg, price=Decimal("500"))
    assert d.accepted


def test_open_order_guard_blocks_both_sides():
    # A symbol with an in-flight order at the broker -> no new order (either side),
    # to avoid duplicating a queued trade. Other symbols are unaffected.
    held = [Position(symbol="VOO", quantity=Decimal("1"), price=Decimal("500"))]
    acct = AccountState(
        account_id="A", account_type="B", is_cash_account=True, has_margin=False,
        settled_cash=Decimal("100"), positions=held, open_order_symbols=["VOO"],
    )
    cfg = _cfg()
    assert validate(_buy("VOO", "10"), acct, cfg, price=Decimal("500")).code == "open_order_exists"
    assert validate(_sell("VOO", "10"), acct, cfg, price=Decimal("500")).code == "open_order_exists"
    assert validate(_buy("SCHD", "10"), acct, cfg, price=Decimal("80")).accepted  # no open order


# --------------------------------------------------------------------------- #
# Market hours
# --------------------------------------------------------------------------- #
def test_is_market_open():
    et = ZoneInfo("America/New_York")
    assert is_market_open(datetime(2026, 6, 17, 11, 0, tzinfo=et)) is True   # Wed 11:00 ET
    assert is_market_open(datetime(2026, 6, 17, 9, 0, tzinfo=et)) is False   # before 9:30
    assert is_market_open(datetime(2026, 6, 17, 16, 0, tzinfo=et)) is False  # close is exclusive
    assert is_market_open(datetime(2026, 6, 17, 15, 59, tzinfo=et)) is True
    assert is_market_open(datetime(2026, 6, 20, 11, 0, tzinfo=et)) is False  # Saturday
    assert is_market_open(datetime(2026, 6, 17, 11, 0)) is True              # naive -> assumed ET
    # Pacific 08:00 == 11:00 ET -> open (timezone conversion works).
    pt = ZoneInfo("America/Los_Angeles")
    assert is_market_open(datetime(2026, 6, 17, 8, 0, tzinfo=pt)) is True


def test_turnover_never_blocks_exit_sell():
    # Turnover budget consumed by a buy; a later SELL must still be accepted (exit path).
    cfg = _cfg(max_turnover_pct=0.05, max_order_pct=1.0)
    held = [Position(symbol="SCHD", quantity=Decimal("1"), price=Decimal("80"))]  # $80 held
    acct = _acct("100", held)  # total 180 -> turnover budget 9
    orders = [_buy("VOO", "9"), _sell("SCHD", "20")]
    decisions = validate_orders(orders, acct, cfg, {"VOO": Decimal("500"), "SCHD": Decimal("80")})
    assert decisions[0].accepted          # buy consumes the budget
    assert decisions[1].accepted          # SELL not blocked by exhausted turnover


# --------------------------------------------------------------------------- #
# Promotion (deterministic fallback, no LLM)
# --------------------------------------------------------------------------- #
def _autonomy_cfg(**kw):
    base = dict(enabled=True, score_floor=75.0, max_promotions_per_run=3)
    base.update(kw)
    return RoboConfig(mode="autonomous", target_allocation={}, allowlist=[],
                      autonomy=AutonomyConfig(**base))


def _seed_candidate(session, ticker, score, status="watchlist"):
    session.add(StockCandidate(ticker=ticker, status=status, composite_score=score))


def test_promote_high_score_candidate(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        _seed_candidate(s, "NVDA", 85.0)
        _seed_candidate(s, "LOWS", 50.0)  # below floor
    with get_session() as s:
        promoted = promote_candidates(s, _autonomy_cfg())  # no evaluator -> deterministic
    assert promoted == ["NVDA"]
    with get_session() as s:
        t = get_thesis(s, "NVDA")
        assert t is not None and t.status == ThesisStatus.ACTIVE.value
        assert 0 < t.conviction <= 1.0
        assert t.invalidation_conditions  # fallback set conservative exits


def test_promotion_respects_per_run_cap(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        for i, score in enumerate([90, 89, 88, 87, 86]):
            _seed_candidate(s, f"AA{i}", float(score))
    with get_session() as s:
        promoted = promote_candidates(s, _autonomy_cfg(max_promotions_per_run=2))
    assert len(promoted) == 2  # only the top 2 by score


def test_promotion_skips_existing_thesis(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        _seed_candidate(s, "NVDA", 90.0)
        save_thesis(s, Thesis(symbol="NVDA", conviction=0.6, status=ThesisStatus.ACTIVE.value))
    with get_session() as s:
        promoted = promote_candidates(s, _autonomy_cfg())
    assert promoted == []  # already maintained -> no duplicate


def test_promotion_disabled_does_nothing(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        _seed_candidate(s, "NVDA", 90.0)
    with get_session() as s:
        assert promote_candidates(s, _autonomy_cfg(enabled=False)) == []


def test_invalidated_name_can_be_repromoted(tmp_path):
    # A name whose prior thesis was INVALIDATED must be promotable again if it
    # re-clears the score floor (it isn't locked out forever).
    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        _seed_candidate(s, "NVDA", 90.0)
        save_thesis(s, Thesis(symbol="NVDA", conviction=0.0,
                              status=ThesisStatus.INVALIDATED.value))
    with get_session() as s:
        assert promote_candidates(s, _autonomy_cfg()) == ["NVDA"]
