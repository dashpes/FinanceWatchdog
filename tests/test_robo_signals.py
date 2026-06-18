"""Tests for the event-driven signal layer (Phase 2).

Three tiers:
  A. PURE scoring/tilt tests — hand-built SignalEvent lists, no DB.
  B. DB-read tests — seed an in-memory-ish SQLite file and assert on collected
     raw signals (out-of-window exclusion, NULL volume, relevance scale, clusters).
  C. Proposer/gate integration — signals are advisory and can never bypass the gate;
     disabled/empty signals reproduce the baseline exactly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from investment_monitor.robo.config import RoboCaps, RoboConfig, SignalConfig
from investment_monitor.robo.gate import validate_orders
from investment_monitor.robo.llm import RoboProposer, _signals_block
from investment_monitor.robo.models import (
    AccountState,
)
from investment_monitor.robo.prompts import PROPOSAL_PROMPT
from investment_monitor.robo.signals import (
    SignalEvent,
    SignalSnapshot,
    build_snapshot,
    collect_raw_signals,
    fetch_signals,
    score_symbol,
    tilt_targets,
)
from investment_monitor.storage import (
    CongressionalTrade,
    EarningsDate,
    InsiderTransaction,
    NewsItem,
    Price,
    get_session,
    init_db,
)

NOW = datetime(2026, 6, 16, 12, 0, 0)


def sc(**kw) -> SignalConfig:
    return SignalConfig(enabled=True, **kw)


def robo_config(**kw) -> RoboConfig:
    base = dict(
        target_allocation={"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2},
        allowlist=["VOO", "SCHD"],
        use_llm=False,
        caps=RoboCaps(max_order_pct=0.25, max_orders_per_run=5, max_orders_per_day=10),
    )
    base.update(kw)
    return RoboConfig(**base)


# --------------------------------------------------------------------------- #
# A. PURE scoring / tilt
# --------------------------------------------------------------------------- #
def test_insider_buy_scores_strongly_positive():
    ev = SignalEvent("insider", direction=1, magnitude=1.0, detail="CFO $500k", age_days=2.0)
    sig = score_symbol("AAPL", [ev], sc())
    assert sig.score > 0.7
    assert sig.has_caution is False
    assert "INSIDER" in sig.summary


def test_recency_decay_reduces_score():
    fresh = score_symbol("AAPL", [SignalEvent("insider", 1, 1.0, "x", age_days=2.0)], sc())
    stale = score_symbol("AAPL", [SignalEvent("insider", 1, 1.0, "x", age_days=20.0)], sc())
    assert fresh.score > stale.score
    assert stale.score > 0  # still positive, just smaller


def test_neutral_events_do_not_move_score():
    # News + volume are attention-only (direction 0): score stays 0.
    events = [
        SignalEvent("news", 0, 0.8, "headline", age_days=0.0),
        SignalEvent("volume", 0, 0.9, "3x", age_days=0.0),
    ]
    sig = score_symbol("AAPL", events, sc())
    assert sig.score == 0.0
    assert sig.has_caution is False
    assert "NEWS" in sig.summary and "VOLUME" in sig.summary


def test_earnings_sets_caution_without_driving_score():
    events = [
        SignalEvent("insider", 1, 1.0, "buy", age_days=0.0),
        SignalEvent("earnings", 0, 0.8, "earnings in 1d", age_days=0.0, caution=True),
    ]
    sig = score_symbol("AAPL", events, sc())
    assert sig.has_caution is True
    assert sig.score == 1.0  # insider drives it; earnings (dir 0) does not
    assert "CAUTION" in sig.summary


def test_conflicting_directional_signals_net_out():
    events = [
        SignalEvent("insider", 1, 1.0, "buy", age_days=0.0),
        SignalEvent("insider", -1, 1.0, "sell", age_days=0.0),
    ]
    sig = score_symbol("AAPL", events, sc())
    assert sig.score == 0.0


def test_strongly_bearish_score_flags_caution():
    sig = score_symbol("AAPL", [SignalEvent("insider", -1, 1.0, "exec dumping", age_days=0.0)], sc())
    assert sig.score < 0
    assert sig.has_caution is True


def test_tilt_targets_preserves_sum_and_bounds():
    snap = SignalSnapshot(
        as_of=NOW, lookback_days=90,
        symbols={"VOO": score_symbol("VOO", [SignalEvent("insider", 1, 1.0, "buy", 0.0)], sc())},
    )
    tilted = tilt_targets({"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2}, snap, max_event_tilt=0.05)
    assert abs(sum(tilted.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in tilted.values())
    assert tilted["VOO"] > 0.5  # bullish signal tilted VOO up
    # Bound is exact (CASH absorbs the change, so no renormalization inflation).
    assert tilted["VOO"] <= 0.5 + 0.05 + 1e-9
    assert abs(tilted["SCHD"] - 0.3) < 1e-9   # no-signal name untouched
    assert abs(tilted["CASH"] - 0.15) < 1e-9  # CASH funded the buy


def test_tilt_cash_absorbs_downtilt_no_signal_names_flat():
    # Only SCHD is bearish; VOO has no signal and MUST stay flat; CASH absorbs.
    snap = SignalSnapshot(
        as_of=NOW, lookback_days=90,
        symbols={"SCHD": score_symbol("SCHD", [SignalEvent("insider", -1, 1.0, "exec sold", 0.0)], sc())},
    )
    alloc = {"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2}
    tilted = tilt_targets(alloc, snap, max_event_tilt=0.05)
    assert abs(sum(tilted.values()) - 1.0) < 1e-9
    assert abs(tilted["VOO"] - 0.5) < 1e-9   # no-signal equity unchanged (no pro-rata leak)
    assert tilted["SCHD"] < 0.3              # bearish -> trimmed
    assert tilted["CASH"] > 0.2              # proceeds went to CASH, not VOO
    assert abs((0.3 - tilted["SCHD"]) - (tilted["CASH"] - 0.2)) < 1e-9  # exactly absorbed


def test_tilt_scales_buys_to_available_cash():
    # Two bullish names want +0.05 each (0.10) but only 0.04 cash is available.
    snap = SignalSnapshot(
        as_of=NOW, lookback_days=90,
        symbols={
            "VOO": score_symbol("VOO", [SignalEvent("insider", 1, 1.0, "buy", 0.0)], sc()),
            "SCHD": score_symbol("SCHD", [SignalEvent("insider", 1, 1.0, "buy", 0.0)], sc()),
        },
    )
    alloc = {"VOO": 0.48, "SCHD": 0.48, "CASH": 0.04}
    tilted = tilt_targets(alloc, snap, max_event_tilt=0.05)
    assert abs(sum(tilted.values()) - 1.0) < 1e-9
    assert all(w >= -1e-12 for w in tilted.values())
    assert tilted["CASH"] < 1e-9  # all available cash deployed, none left negative
    # Net deployment uses exactly the available cash (0.04), split across the buys.
    assert abs((tilted["VOO"] - 0.48) + (tilted["SCHD"] - 0.48) - 0.04) < 1e-9


def test_tilt_caution_never_increases_exposure():
    caution_sig = score_symbol(
        "VOO",
        [
            SignalEvent("insider", 1, 1.0, "buy", 0.0),
            SignalEvent("earnings", 0, 0.9, "earnings in 1d", 0.0, caution=True),
        ],
        sc(),
    )
    snap = SignalSnapshot(as_of=NOW, lookback_days=90, symbols={"VOO": caution_sig})
    tilted = tilt_targets({"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2}, snap, max_event_tilt=0.05)
    assert abs(sum(tilted.values()) - 1.0) < 1e-9
    # Bullish insider but CAUTION → must not add into VOO.
    assert tilted["VOO"] <= 0.5 + 1e-9


def test_tilt_empty_snapshot_is_identity():
    empty = SignalSnapshot(as_of=NOW, lookback_days=0, symbols={})
    alloc = {"VOO": 0.5, "SCHD": 0.3, "CASH": 0.2}
    assert tilt_targets(alloc, empty, 0.05) == alloc


def test_build_snapshot_empty_input():
    snap = build_snapshot({}, robo_config(signals=sc()), now=NOW)
    assert snap.is_empty
    assert snap.prompt_block() == ""


# --------------------------------------------------------------------------- #
# B. DB-read collection
# --------------------------------------------------------------------------- #
def _seed(db, rows):
    init_db(db)
    with get_session() as session:
        for r in rows:
            session.add(r)


def test_collect_insider_buy(tmp_path):
    db = tmp_path / "sig.db"
    _seed(db, [
        InsiderTransaction(
            ticker="AAPL", filing_date=date.today(), trade_date=date.today() - timedelta(days=2),
            owner_name="Jane CFO", owner_title="CFO", transaction_type="P",
            shares=1000, total_value=500_000.0,
        ),
    ])
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
    assert "AAPL" in raw
    insider = [e for e in raw["AAPL"] if e.category == "insider"]
    assert len(insider) == 1
    assert insider[0].direction == 1
    assert insider[0].magnitude == 1.0  # $500k -> saturation, +exec, capped


def test_insider_out_of_window_excluded(tmp_path):
    db = tmp_path / "sig.db"
    _seed(db, [
        InsiderTransaction(
            ticker="AAPL", filing_date=date.today(), trade_date=date.today() - timedelta(days=60),
            owner_name="Old Buyer", owner_title="CEO", transaction_type="P",
            shares=1000, total_value=500_000.0,
        ),
    ])
    init_db(db)
    with get_session() as session:
        # insider_days default 30 -> the 60-day-old txn is filtered out by the query.
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
    assert "AAPL" not in raw


def test_congress_cluster_threshold(tmp_path):
    db = tmp_path / "sig.db"
    rows = [
        CongressionalTrade(
            ticker="AAPL", politician=f"Rep {i}", trade_type="buy",
            amount_range="$1,001-$15,000", trade_date=date.today() - timedelta(days=5),
        )
        for i in range(3)  # 3 distinct politicians == cluster
    ]
    _seed(db, rows)
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
    congress = [e for e in raw.get("AAPL", []) if e.category == "congress"]
    assert len(congress) == 1
    assert congress[0].direction == 1


def test_congress_below_cluster_emits_nothing(tmp_path):
    db = tmp_path / "sig.db"
    rows = [
        CongressionalTrade(
            ticker="AAPL", politician=f"Rep {i}", trade_type="buy",
            amount_range="$1,001-$15,000", trade_date=date.today() - timedelta(days=5),
        )
        for i in range(2)  # below cluster_min_unique=3
    ]
    _seed(db, rows)
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
    assert "AAPL" not in raw


def test_volume_spike_detected_and_nulls_skipped(tmp_path):
    db = tmp_path / "sig.db"
    rows = [Price(ticker="AAPL", date=date.today(), close=100.0, volume=10_000_000)]
    # One NULL-volume historical row must be skipped without crashing.
    rows.append(Price(ticker="AAPL", date=date.today() - timedelta(days=1), close=100.0, volume=None))
    for i in range(2, 21):
        rows.append(Price(ticker="AAPL", date=date.today() - timedelta(days=i),
                          close=100.0, volume=1_000_000))
    _seed(db, rows)
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
    volume = [e for e in raw["AAPL"] if e.category == "volume"]
    assert len(volume) == 1
    assert volume[0].direction == 0  # non-directional
    assert 0 < volume[0].magnitude <= 1.0


def test_news_relevance_scale(tmp_path):
    db = tmp_path / "sig.db"
    _seed(db, [
        NewsItem(ticker="AAPL", headline="Apple beats", source="x", url="u1", relevance_score=8.0),
        NewsItem(ticker="MSFT", headline="meh", source="x", url="u2", relevance_score=4.0),
    ])
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL", "MSFT"], robo_config(signals=sc()))
    # relevance 8 -> emitted with magnitude (8-5)/5 = 0.6; relevance 4 (< min 5) -> nothing.
    news = [e for e in raw["AAPL"] if e.category == "news"]
    assert len(news) == 1
    assert abs(news[0].magnitude - 0.6) < 1e-9
    assert "MSFT" not in raw


def test_earnings_imminent_flags_caution(tmp_path):
    db = tmp_path / "sig.db"
    _seed(db, [
        EarningsDate(ticker="AAPL", earnings_date=date.today() + timedelta(days=2), confirmed=True),
    ])
    init_db(db)
    with get_session() as session:
        raw = collect_raw_signals(session, ["AAPL"], robo_config(signals=sc()))
        snap = build_snapshot(raw, robo_config(signals=sc()))
    assert snap.for_symbol("AAPL").has_caution is True


def test_empty_db_yields_empty_snapshot(tmp_path):
    db = tmp_path / "sig.db"
    init_db(db)
    with get_session() as session:
        snap = fetch_signals(session, robo_config(signals=sc()), _account())
    assert snap.is_empty


def test_fetch_signals_disabled_returns_empty(tmp_path):
    db = tmp_path / "sig.db"
    # Seed a strong signal, but signals are disabled -> snapshot must be empty.
    _seed(db, [
        InsiderTransaction(
            ticker="VOO", filing_date=date.today(), trade_date=date.today(),
            owner_name="CEO", owner_title="CEO", transaction_type="P",
            shares=1, total_value=1_000_000.0,
        ),
    ])
    init_db(db)
    with get_session() as session:
        snap = fetch_signals(session, robo_config(), _account())  # signals disabled by default
    assert snap.is_empty


# --------------------------------------------------------------------------- #
# C. Proposer / gate integration — signals are advisory, never authoritative
# --------------------------------------------------------------------------- #
def _account(cash="100", positions=None) -> AccountState:
    return AccountState(
        account_id="ACC1", account_type="BROKERAGE", is_cash_account=True,
        has_margin=False, settled_cash=Decimal(cash), positions=positions or [],
    )


def test_empty_signals_reproduce_baseline_exactly():
    account = _account("100")
    proposer = RoboProposer(None, robo_config())  # use_llm False
    baseline, src_base = proposer.propose(account, signals=None)
    empty = SignalSnapshot(as_of=NOW, lookback_days=0, symbols={})
    with_empty, src_empty = proposer.propose(account, signals=empty)
    assert baseline == with_empty
    assert src_base == "deterministic"
    assert src_empty == "deterministic"  # empty snapshot is NOT tagged +sig


def test_signal_tilts_deterministic_orders_and_tags_source():
    account = _account("100")
    snap = SignalSnapshot(
        as_of=NOW, lookback_days=90,
        symbols={"VOO": score_symbol("VOO", [SignalEvent("insider", 1, 1.0, "buy", 0.0)], sc())},
    )
    # Loosen the per-order cap so the tilt is observable (with the 25% cap both
    # buys would be cap-bound at $25 — see test_gate_still_rejects_over_cap_order).
    config = robo_config(caps=RoboCaps(max_order_pct=1.0, max_orders_per_run=5, max_orders_per_day=10))
    proposer = RoboProposer(None, config)
    baseline, _ = proposer.propose(account, signals=None)
    tilted, source = proposer.propose(account, signals=snap)
    assert source == "deterministic+sig"
    base_voo = next(o.notional for o in baseline if o.symbol == "VOO")
    tilt_voo = next(o.notional for o in tilted if o.symbol == "VOO")
    assert tilt_voo > base_voo  # bullish insider -> bigger VOO buy


def test_gate_still_rejects_over_cap_order_even_with_signal():
    """A signal-justified, over-cap LLM order must still be rejected by the gate."""

    class FakeLLM:
        model = "fake"

        def __init__(self, text):
            self._text = text
            self.client = self
            self.last_prompt = ""

        def is_available(self):
            return True

        def generate(self, model, prompt, options):
            self.last_prompt = prompt
            return {"response": self._text}

    account = _account("100")  # total value $100, max_order_pct 0.25 -> cap $25
    config = robo_config(use_llm=True)
    snap = SignalSnapshot(
        as_of=NOW, lookback_days=90,
        symbols={"VOO": score_symbol("VOO", [SignalEvent("insider", 1, 1.0, "CFO bought", 0.0)], sc())},
    )
    # LLM cites the signal to justify a $90 buy — 90% of the portfolio, far over cap.
    fake = FakeLLM('[{"symbol": "VOO", "side": "buy", "notional": 90, "reason": "insider buy signal"}]')
    proposer = RoboProposer(fake, config)
    orders, source = proposer.propose(account, signals=snap)

    assert source == "llm+sig"
    assert "RECENT EVENT SIGNALS" in fake.last_prompt  # signals reached the prompt
    assert "VOO" in fake.last_prompt

    prices = {"VOO": Decimal("500"), "SCHD": Decimal("80")}
    decisions = validate_orders(orders, account, config, prices, orders_today=0)
    voo = [d for d in decisions if d.order.symbol == "VOO"][0]
    assert voo.accepted is False
    assert voo.code == "exceeds_max_order_pct"


def test_rebalance_run_end_to_end_with_signals(tmp_path):
    """Full pipeline with signals enabled: tilts, audits, tags source, places nothing."""
    from investment_monitor.config import Settings
    from investment_monitor.robo.broker import PlacedOrder, PreflightResult
    from investment_monitor.robo.rebalance import rebalance_run

    class FakeBroker:
        def __init__(self, account):
            self._account = account
            self.dry_run = True
            self.place_called = 0

        def get_account_state(self):
            return self._account

        def get_quotes(self, symbols):
            return {"VOO": Decimal("500"), "SCHD": Decimal("80")}

        def preflight(self, order):
            return PreflightResult(ok=True, estimated_cost=order.notional or Decimal("0"),
                                   margin_requirement=Decimal("0"))

        def place_order(self, order):
            self.place_called += 1
            return PlacedOrder(order_id="fake", status="NEW", simulated=False)

    db = tmp_path / "test.db"
    _seed(db, [
        InsiderTransaction(
            ticker="VOO", filing_date=date.today(), trade_date=date.today(),
            owner_name="CEO Jane", owner_title="CEO", transaction_type="P",
            shares=1000, total_value=500_000.0,
        ),
    ])
    settings = Settings(
        public_api_token="t", robo_force_dry_run=True,
        config_dir=tmp_path, data_dir=tmp_path, log_dir=tmp_path, db_path=db,
    )
    config = robo_config(signals=sc())  # signals enabled, use_llm False
    broker = FakeBroker(_account("100"))
    result = rebalance_run(config, settings, broker=broker)

    assert result.status == "completed"
    assert result.dry_run is True
    assert result.source == "deterministic+sig"  # signal was active
    assert broker.place_called == 0  # dry-run: nothing real placed
    audit_text = (tmp_path / "robo_audit.jsonl").read_text()
    assert '"event": "signals"' in audit_text  # the snapshot was audited


def test_signals_block_empty_keeps_prompt_identical():
    # signals=None -> empty block -> no EVENT SIGNALS section; GOAL block intact.
    assert _signals_block(None) == ""
    rendered = PROPOSAL_PROMPT.format(
        settled_cash=Decimal("100"), total_value=Decimal("100"),
        allowlist="VOO, SCHD", max_order_pct="25%", rebalance_threshold="5%",
        positions_block="  VOO: 0.0% -> 50.0%", signals_block="",
    )
    assert "RECENT EVENT SIGNALS" not in rendered
    assert "GOAL" in rendered
    assert "  VOO: 0.0% -> 50.0%\n\nGOAL" in rendered  # blank line preserved
