"""Configuration for the robo advisor.

Non-secret settings live in ``config/robo.yaml`` and are validated by
``RoboConfig``. The Public.com personal access token is a *secret* and is read
from the environment via the shared ``Settings`` (``PUBLIC_API_TOKEN``), never
from YAML.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from investment_monitor.robo.models import CASH_SYMBOL


class ConfigError(ValueError):
    """A robo.yaml that fails validation, surfaced as a single clear message.

    Raised by ``RoboConfig.from_yaml`` instead of letting a raw pydantic
    ``ValidationError`` (or a YAML parse error) escape. A bad config file must
    never crash a CLI command or a launchd daemon with an unhandled traceback —
    the caller logs this message and exits cleanly instead of silently halting
    the autonomous trader. Subclasses ``ValueError`` so existing ``except
    ValueError`` handlers (e.g. the ``config set`` path) keep working.
    """


class Mode(str, Enum):
    """Operating mode (categorical — a real enum so a CLI/GUI gets a dropdown).

    ``str``-valued so existing ``config.mode == "autonomous"`` comparisons and YAML
    round-trips keep working unchanged.
    """

    rebalance = "rebalance"   # fixed target_allocation
    autonomous = "autonomous"  # conviction-driven weights from the thesis store

# Sum of target weights must be within this tolerance of 1.0.
_ALLOCATION_TOLERANCE = Decimal("0.001")


class RoboCaps(BaseModel):
    """Rate and size caps enforced by the guardrail gate.

    Fields carrying ``json_schema_extra={"x_ui": ...}`` are user-tunable and are
    surfaced by ``robo.tunables`` to the ``config`` CLI (and, later, a GUI). The hard
    validation bounds live in the pydantic constraints; ``x_ui`` only adds rendering
    hints (group/control/slider range/step/unit).
    """

    max_order_pct: float = Field(
        default=0.25, gt=0, le=1.0,
        title="Max order size",
        description="Largest single order as a fraction of portfolio value.",
        json_schema_extra={"x_ui": {"group": "Trading", "control": "slider",
                                    "min": 0.05, "max": 1.0, "step": 0.05, "unit": "fraction"}},
    )
    max_orders_per_run: int = Field(
        default=5, ge=0, le=100,
        title="Max orders per run",
        description="Cap on orders placed in a single trade run.",
        json_schema_extra={"x_ui": {"group": "Trading", "control": "stepper",
                                    "min": 1, "max": 20, "step": 1}},
    )
    max_orders_per_day: int = Field(
        default=10, ge=0, le=1000,
        title="Max orders per day",
        description="Cap on orders placed across all runs in a day.",
        json_schema_extra={"x_ui": {"group": "Trading", "control": "stepper",
                                    "min": 1, "max": 50, "step": 1}},
    )
    # Fraction of an order's cost reserved for fees/slippage when checking affordability.
    fee_buffer: float = Field(default=0.01, ge=0, lt=1.0)

    # --- autonomous-mode safety guards (Phase 4). Permissive defaults = disabled,
    # so rebalance mode and existing behavior are unchanged unless these are set. ---
    max_positions: int = Field(
        default=0, ge=0, le=50,
        title="Maximum positions",
        description="Most distinct holdings the portfolio may carry (0 = unlimited).",
        json_schema_extra={"x_ui": {"group": "Risk", "control": "stepper",
                                    "min": 0, "max": 50, "step": 1}},
    )
    max_per_name_weight: float = Field(
        default=1.0, gt=0, le=1.0,
        title="Max weight per name",
        description="Concentration cap: no single name above this fraction (1.0 = no cap).",
        json_schema_extra={"x_ui": {"group": "Risk", "control": "slider",
                                    "min": 0.05, "max": 1.0, "step": 0.05, "unit": "fraction"}},
    )
    max_turnover_pct: float = Field(
        default=0.0, ge=0, le=10.0,
        title="Max turnover per run",
        description="Cap on gross buys per run as a fraction of portfolio value (0 = unlimited).",
        json_schema_extra={"x_ui": {"group": "Trading", "control": "slider",
                                    "min": 0.0, "max": 2.0, "step": 0.1, "unit": "fraction"}},
    )
    max_drawdown_pct: float = Field(
        default=0.0, ge=0, le=100.0,
        title="Drawdown breaker",
        description="Halt new buys when down this % from the prior peak (0 = off; sells always allowed).",
        json_schema_extra={"x_ui": {"group": "Risk", "control": "slider",
                                    "min": 0, "max": 90, "step": 5, "unit": "percent"}},
    )


# Event-signal categories the proposer understands. Weights must key into this set.
SIGNAL_CATEGORIES = ("insider", "congress", "volume", "news", "earnings")


class SignalConfig(BaseModel):
    """Event-driven signal settings (Phase 2).

    Signals only *inform* proposals — they are NEVER seen by the guardrail gate and
    can never relax a cap. When ``enabled`` is False the proposer behaves exactly as
    the pure drift-to-target rebalancer. Event-driven exposure is bounded by
    ``max_event_tilt``: a holding's effective target weight may move by at most that
    fraction on events, well under the gate's per-order ``max_order_pct``.
    """

    # Master switch. OFF by default: behavior is identical to the baseline rebalancer.
    enabled: bool = False

    # --- lookback windows (per source) ---
    news_hours: int = Field(default=24, ge=1)
    insider_days: int = Field(default=30, ge=1)
    earnings_days_ahead: int = Field(default=14, ge=1)
    congress_days: int = Field(default=90, ge=1)
    volume_lookback: int = Field(default=20, ge=5)
    # Magnitude of an event halves every this-many days (recency decay).
    recency_half_life_days: float = Field(default=7.0, gt=0)

    # --- per-category relative weight in the net directional score ---
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "insider": 1.0,
            "congress": 0.5,
            "volume": 0.4,
            "news": 0.3,
            "earnings": 0.6,
        }
    )

    # --- detection thresholds (mirror the existing alert rules) ---
    news_relevance_min: float = Field(default=5.0, ge=0, le=10)  # 1-10 scale
    insider_buy_min_value: float = Field(default=100_000, ge=0)
    insider_sell_min_value: float = Field(default=100_000, ge=0)
    cluster_min_unique: int = Field(default=3, ge=2)
    volume_spike_multiplier: float = Field(default=2.5, gt=1)
    # Earnings within this many days flags a holding CAUTION (suppress buying into it).
    earnings_caution_days: int = Field(default=3, ge=0)

    # --- risk bound: max fraction a name's target weight may move on events ---
    max_event_tilt: float = Field(default=0.05, ge=0, le=0.25)

    @field_validator("weights")
    @classmethod
    def _known_weight_keys(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(SIGNAL_CATEGORIES)
        if unknown:
            raise ValueError(
                f"unknown signal weight categories {sorted(unknown)}; "
                f"valid: {list(SIGNAL_CATEGORIES)}"
            )
        for cat, w in v.items():
            if w < 0:
                raise ValueError(f"signal weight for {cat} must be >= 0, got {w}")
        return v


class SizingConfig(BaseModel):
    """Risk-adjusted conviction -> target-weight sizing (Phase 3, autonomous mode).

    Pure deterministic sizing (see ``robo/sizing.py``). The LLM sets *conviction*;
    this config governs how conviction + Monte-Carlo risk metrics become a bounded
    target weight. The guardrail gate still re-checks every resulting order.
    """

    # Hard ceiling on any single name's target weight.
    max_position_weight: float = Field(
        default=0.15, gt=0, le=1.0,
        title="Target weight ceiling",
        description="Hard ceiling on any single name's conviction-sized target weight.",
        json_schema_extra={"x_ui": {"group": "Sizing", "control": "slider",
                                    "min": 0.05, "max": 1.0, "step": 0.05, "unit": "fraction"}},
    )
    # Fractional-Kelly multiplier on the risk-adjusted (Sharpe) signal.
    kelly_fraction: float = Field(
        default=0.25, gt=0, le=1.0,
        title="Kelly fraction",
        description="Fractional-Kelly multiplier on the risk-adjusted signal (lower = more conservative).",
        json_schema_extra={"x_ui": {"group": "Sizing", "control": "slider",
                                    "min": 0.1, "max": 1.0, "step": 0.05}},
    )
    # How hard to shrink size as 90d CVaR (tail loss) grows.
    cvar_aversion: float = Field(default=2.0, ge=0)
    # Annualized risk-free rate used in the Sharpe numerator.
    risk_free: float = Field(default=0.04, ge=0, le=1.0)
    # Volatility floor so a near-zero-vol sim can't explode the Sharpe ratio.
    min_vol: float = Field(default=0.05, gt=0)
    # When no simulation exists, weight = conviction * this (conservative).
    no_sim_weight_per_conviction: float = Field(default=0.05, ge=0, le=1.0)
    # Conviction time-decay: decays toward `conviction_floor` with this half-life,
    # reset whenever a thesis is re-evaluated. Prevents stale max-conviction.
    conviction_half_life_days: float = Field(default=30.0, gt=0)
    conviction_floor: float = Field(default=0.5, ge=0, le=1.0)
    # Anti-churn: EWMA-smooth conviction over its recent re-eval history (half-life in
    # points) BEFORE sizing, so a one-off intraday LLM wobble (e.g. an overnight 0.7->0.4
    # that reverts) barely moves the target while a SUSTAINED move is still followed. A
    # broken/invalidated thesis (conviction 0) is never smoothed back up, so exits stay
    # prompt. 0 disables smoothing (raw latest conviction).
    conviction_smoothing_halflife: float = Field(default=3.0, ge=0)
    # Anti-averaging-up: block a BUY that ADDS to an existing position if it would raise
    # the cost basis (buy price above avg cost beyond the tolerance) UNLESS the thesis has
    # strengthened — conviction now >= entry + `add_strengthen_margin`, or already >=
    # `strong_add_conviction`. Averaging DOWN and opening NEW positions are always allowed.
    block_average_up: bool = Field(default=True)
    average_up_tolerance: float = Field(default=0.03, ge=0)      # buying up to 3% over cost is fine
    add_strengthen_margin: float = Field(default=0.15, ge=0, le=1.0)
    strong_add_conviction: float = Field(default=0.7, ge=0, le=1.0)
    # Concentration: hold FEWER, STRONGER names. A thesis below this (effective) conviction
    # gets NO capital — capital isn't spread across every marginal idea. Combined with the
    # gate's caps.max_positions (also applied at sizing, keeping the top-N by size) and the
    # per-name max_position_weight cap, this yields a handful of meaningful positions with a
    # bounded top, and the rest in cash / the cash ETF. 0 keeps every positive-weight name.
    min_conviction_to_hold: float = Field(default=0.35, ge=0, le=1.0)
    # Anti-churn hysteresis at the top-N selection cliff: a HELD incumbent keeps its slot
    # unless a challenger's sized weight beats it by this fraction (0.25 = 25% larger).
    # Without it, ±0.02 of LLM conviction wobble inside a saturated 0.9+ band rotated
    # real positions daily (live account: every buy of a 21-day stretch was fully sold
    # again within 1-6 days on "target changed", not on any broken thesis). 0 = plain
    # top-N. Exits are never delayed: an invalidated/exited/sub-floor name has weight 0
    # and is no incumbent at all.
    selection_hysteresis: float = Field(
        default=0.25, ge=0,
        title="Selection hysteresis",
        description="A held name keeps its top-N slot unless a challenger beats it by this margin.",
        json_schema_extra={"x_ui": {"group": "Sizing", "control": "slider",
                                    "min": 0.0, "max": 1.0, "step": 0.05, "unit": "fraction"}},
    )
    # Always leave at least this fraction of the portfolio in cash (autonomous mode).
    min_cash_weight: float = Field(
        default=0.05, ge=0, lt=1.0,
        title="Minimum cash",
        description="Always keep at least this fraction of the portfolio in cash.",
        json_schema_extra={"x_ui": {"group": "Sizing", "control": "slider",
                                    "min": 0.0, "max": 0.9, "step": 0.05, "unit": "fraction"}},
    )


class ExitConfig(BaseModel):
    """Deterministic take-profit policy (the upside twin of invalidation).

    These are the DEFAULT ``check_exit`` thresholds for every live thesis; a thesis's
    own ``exit_conditions`` (stamped at promotion or proposed by the LLM, clamped)
    are merged OVER them. A value of 0 disables that trigger. Checked in the same
    passes as invalidation (twice-daily re-eval + hourly sentinel); a trip sets the
    thesis EXITED -> conviction 0 -> the next gated trade run sells the position.
    """

    enabled: bool = Field(
        default=True,
        title="Take-profit exits",
        description="Deterministically realize gains (profit target / trailing stop / horizon).",
        json_schema_extra={"x_ui": {"group": "Exits", "control": "toggle"}},
    )
    # Exit once the gain from entry reaches this percent. Default mirrors the promotion
    # run-up guard's max_run_pct=40: if +40% means "priced in, don't enter", the same
    # move on a HELD name means "realized, take it". 0 = off.
    profit_target_pct: float = Field(
        default=40.0, ge=0,
        title="Profit target %",
        description="Sell once a position is up this % from entry (0 = off).",
        json_schema_extra={"x_ui": {"group": "Exits", "control": "slider",
                                    "min": 0, "max": 200, "step": 5, "unit": "%"}},
    )
    # Exit this percent below the post-entry high — protects an open gain from round-
    # tripping (the entry-based price_drop_pct invalidation only catches it far lower).
    trailing_stop_pct: float = Field(
        default=15.0, ge=0,
        title="Trailing stop %",
        description="Sell this % below the post-entry high once armed (0 = off).",
        json_schema_extra={"x_ui": {"group": "Exits", "control": "slider",
                                    "min": 0, "max": 50, "step": 1, "unit": "%"}},
    )
    # The trailing stop arms only once the high-water gain reaches this percent, so a
    # fresh flat position can't be noise-stopped (downside stays with invalidation).
    trailing_arm_pct: float = Field(
        default=10.0, ge=0,
        title="Trailing stop arms at %",
        description="Gain required before the trailing stop activates.",
        json_schema_extra={"x_ui": {"group": "Exits", "control": "slider",
                                    "min": 0, "max": 100, "step": 5, "unit": "%"}},
    )
    # Time-boxed horizon exit, in days. 0 = off GLOBALLY by default: a horizon suits
    # event-driven confluence bets (whose promotion stamps 90d per thesis, the backtest-
    # validated default) but would churn a still-high-scoring discovery name straight
    # into re-promotion.
    max_hold_days: float = Field(
        default=0.0, ge=0,
        title="Max hold (days)",
        description="Sell after this many days regardless (0 = off; confluence theses carry their own 90d).",
        json_schema_extra={"x_ui": {"group": "Exits", "control": "slider",
                                    "min": 0, "max": 365, "step": 5, "unit": "days"}},
    )

    def as_conditions(self) -> dict:
        """The config defaults as a ``check_exit`` conditions dict (empty when disabled)."""
        if not self.enabled:
            return {}
        return {
            "profit_target_pct": self.profit_target_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "trailing_arm_pct": self.trailing_arm_pct,
            "max_hold_days": self.max_hold_days,
        }


class AutonomyConfig(BaseModel):
    """Autonomous stock selection (Phase 4): promote discovery-funnel names to theses.

    'Fully auto behind a score floor': a candidate whose composite score clears
    ``score_floor`` is promoted to an active thesis automatically (no human in the
    loop), capped at ``max_promotions_per_run`` per run and biased toward maintaining
    existing theses over churning into new names. Disabled by default.
    """

    enabled: bool = False
    # Run the research discovery pipeline (find + score candidates) at the start of
    # each autonomous run, so the agent sources its OWN universe rather than relying
    # on discovery being run separately. Heavy (LLM-scores the universe) — off by default.
    discover: bool = False
    score_floor: float = Field(default=75.0, ge=0, le=100)        # composite score to promote
    max_promotions_per_run: int = Field(default=3, ge=0)
    # Also require a fresh research report recommending buy/strong_buy (extra key).
    require_buy_recommendation: bool = False
    # After a take-profit/horizon EXIT, don't re-promote the same name for this many
    # days. Without it, a still-high-scoring candidate re-promotes on the very next
    # run — either churning the sell straight back into a buy, or (when the exit
    # tripped intraday via the sentinel) re-inflating the target weight BEFORE the
    # sell so the take-profit silently never executes. 0 = off.
    reentry_cooldown_days: float = Field(default=30.0, ge=0)
    # --- Book hygiene: the active set is a WORKING set, not an archive. -----------
    # Promotion only ever ADDED theses (83 active after 4 weeks live; 36 below the
    # conviction floor — never sizeable, yet each still cost a full LLM re-eval every
    # cycle, ~4h/night on a Pi). Benching demotes them to WATCH: thesis + history kept,
    # deterministic checks only, one LLM re-look per bench_reeval_days, auto-revived
    # when conviction recovers or a fresh confluence finding lands.
    # Hard cap on ACTIVE theses; the weakest (smoothed conviction) are benched. 0 = off.
    max_active_theses: int = Field(
        default=30, ge=0,
        title="Max active theses",
        description="Cap the daily-maintained book; weakest are benched to WATCH (0 = uncapped).",
        json_schema_extra={"x_ui": {"group": "Autonomy", "control": "slider",
                                    "min": 0, "max": 100, "step": 5}},
    )
    # Bench an (unsizeable) thesis once its conviction has stayed below
    # sizing.min_conviction_to_hold for this many days. 0 disables sub-floor benching.
    bench_after_days: float = Field(default=7.0, ge=0)
    # Benched theses get one LLM re-evaluation this often (they keep hourly-cheap
    # deterministic coverage only while benched).
    bench_reeval_days: float = Field(default=7.0, gt=0)


class LearningConfig(BaseModel):
    """Feedback-loop settings: the robo learns from its own realized outcomes (Phase 6).

    The DB table ``learning_events`` is the system of record — full history lives
    there at zero cost. Only COMPACT, EWMA-smoothed aggregates ever enter the LLM
    prompt or the sizing math, so the feedback loop cannot rot the context window.

    Enabled by default (paper trading), but learning is a *no-op until outcomes
    accrue*: the accuracy multiplier is exactly 1.0 and the prompt is unchanged until
    a symbol has ``min_samples`` recorded outcomes, so existing runs are unaffected
    until the ledger has real data.
    """

    # Master switch for the whole feedback loop.
    enabled: bool = True
    # Record realized thesis outcomes into the ledger on every re-evaluation.
    record_outcomes: bool = True
    # Inject a compact realized-performance + track-record block into the re-eval prompt.
    outcome_aware_reeval: bool = True
    # Apply a bounded accuracy multiplier to conviction-driven sizing.
    accuracy_sizing: bool = True

    # Minimum recorded outcomes for a symbol before its aggregate stats influence
    # sizing or the prompt's track-record line.
    min_samples: int = Field(default=6, ge=1)
    # Minimum whole days a thesis must be held before an outcome is recorded or the
    # realized-performance line is shown — suppresses just-opened "~0d ago, +0.0%"
    # noise and de-duplicates intraday re-evals to one outcome per symbol per day.
    min_days_held: int = Field(default=2, ge=0)
    # Strength of the tilt: multiplier = 1 + accuracy_weight*(hit_rate-0.5)*2, clamped.
    accuracy_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Multiplier clamp band. Symmetric by default (0.5..1.5): a consistently WRONG name
    # gets sized down to 0.5x and a consistently RIGHT one up to 1.5x — so the loop both
    # cuts losers AND rewards proven winners (add more when the track record justifies).
    # Set ceiling to 1.0 for the old shrink-only behaviour.
    modifier_floor: float = Field(default=0.5, gt=0.0, le=1.0)
    modifier_ceiling: float = Field(default=1.5, ge=1.0)
    # Recency: hit-rate half-life in events, and how many recent events to aggregate.
    ewma_halflife: float = Field(default=10.0, gt=0)
    recent_window: int = Field(default=20, ge=1)

    # Benchmark for the realized-excess-return line; display clamp for the prompt.
    benchmark_symbol: str = "SPY"
    max_abs_return_pct: float = Field(default=500.0, gt=0)

    @model_validator(mode="after")
    def _check_band(self) -> "LearningConfig":
        if self.modifier_ceiling < self.modifier_floor:
            raise ValueError(
                f"modifier_ceiling ({self.modifier_ceiling}) must be >= "
                f"modifier_floor ({self.modifier_floor})"
            )
        return self


class RoboConfig(BaseModel):
    """Validated robo-advisor configuration (from ``config/robo.yaml``)."""

    # symbol -> target weight (0..1). May include the pseudo-symbol "CASH".
    target_allocation: dict[str, float] = Field(default_factory=dict)
    # Only trade if a holding drifts more than this fraction from its target.
    rebalance_threshold: float = Field(
        default=0.05, ge=0, le=1.0,
        title="Rebalance threshold",
        description="Only trade a name once it drifts more than this fraction from target.",
        json_schema_extra={"x_ui": {"group": "Trading", "control": "slider",
                                    "min": 0.0, "max": 0.5, "step": 0.01, "unit": "fraction"}},
    )
    # Symbols the gate will allow trading. Defaults to target_allocation keys.
    allowlist: list[str] = Field(default_factory=list)
    # Symbols the gate must never BUY (sells/exits always allowed). Operator-curated;
    # the system also auto-learns broker-refused, un-buyable names into a separate
    # persisted learned blocklist (see robo/blocklist.py). Union of both is enforced.
    blocklist: list[str] = Field(default_factory=list)
    # Cash sleeve: when set (e.g. "SGOV"/"BIL"), the intended cash weight (above the raw
    # min_cash_weight buffer) is parked in this short-term Treasury ETF instead of sitting
    # idle — so uninvested capital earns ~T-bill yield while staying liquid to fund buys.
    # Blank = hold raw cash (original behaviour). Autonomous mode only.
    cash_etf: str = ""

    caps: RoboCaps = Field(default_factory=RoboCaps)

    # Operating mode. "rebalance" (default) = fixed target_allocation, exactly
    # today's behavior. "autonomous" = conviction-driven weights from the thesis
    # store. The mode never affects dry-run/gate safety.
    mode: Mode = Field(
        default=Mode.rebalance,
        title="Operating mode",
        description="rebalance = fixed target_allocation; autonomous = conviction-driven from theses.",
        json_schema_extra={"x_ui": {"group": "Strategy", "control": "select"}},
    )

    # Event-driven signal settings (Phase 2). Disabled by default.
    signals: SignalConfig = Field(default_factory=SignalConfig)

    # Risk-adjusted conviction sizing (Phase 3, used in autonomous mode).
    sizing: SizingConfig = Field(default_factory=SizingConfig)

    # Deterministic take-profit exits (profit target / trailing stop / horizon).
    exits: ExitConfig = Field(default_factory=ExitConfig)

    # Autonomous stock selection from the discovery funnel (Phase 4). Disabled by default.
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)

    # Feedback loop: learn from realized outcomes (Phase 6). On by default, but inert
    # until the ledger accrues outcomes (min_samples gate keeps the multiplier at 1.0).
    learning: LearningConfig = Field(default_factory=LearningConfig)

    # Safety: simulate everything and place no real orders when True. Default True.
    dry_run: bool = Field(
        default=True,
        title="Dry run (paper)",
        description="Simulate everything and place NO real orders when on.",
        json_schema_extra={"x_ui": {"group": "Safety", "control": "toggle"}},
    )
    # Only PLACE live orders during US market hours (research/maintenance still run
    # 24/7; off-hours live runs propose + gate but defer placement). Default True.
    require_market_hours: bool = Field(
        default=True,
        title="Trade only in market hours",
        description="Defer live placement outside US market hours (research still runs 24/7).",
        json_schema_extra={"x_ui": {"group": "Safety", "control": "toggle"}},
    )
    # Whether to consult the local LLM for proposals (it is always re-checked by code).
    use_llm: bool = True
    # Optional override of which Ollama model to use; falls back to Settings.ollama_model.
    ollama_model: str = ""
    # Optional specific Public account id to use (required if more than one exists).
    account_id: str = ""

    @field_validator("target_allocation")
    @classmethod
    def _upper_symbols(cls, v: dict[str, float]) -> dict[str, float]:
        return {sym.upper(): weight for sym, weight in v.items()}

    @field_validator("allowlist", "blocklist")
    @classmethod
    def _upper_symbol_lists(cls, v: list[str]) -> list[str]:
        return [s.upper() for s in v]

    @model_validator(mode="after")
    def _validate_allocation(self) -> "RoboConfig":
        if self.target_allocation:
            for sym, weight in self.target_allocation.items():
                if weight < 0 or weight > 1:
                    raise ValueError(f"target weight for {sym} must be in [0, 1], got {weight}")
            total = sum(Decimal(str(w)) for w in self.target_allocation.values())
            if abs(total - Decimal("1")) > _ALLOCATION_TOLERANCE:
                raise ValueError(
                    f"target_allocation weights must sum to 1.0 (got {float(total):.4f})"
                )
        # Default the trading allowlist to the non-cash target symbols.
        if not self.allowlist:
            self.allowlist = [s for s in self.target_allocation if s != CASH_SYMBOL]
        return self

    @property
    def tradeable_symbols(self) -> list[str]:
        """Non-cash symbols in the target allocation."""
        return [s for s in self.target_allocation if s != CASH_SYMBOL]

    @property
    def cash_target_weight(self) -> float:
        """Target fraction of the portfolio to hold in cash."""
        return self.target_allocation.get(CASH_SYMBOL, 0.0)

    @classmethod
    def from_yaml(cls, path: Path) -> "RoboConfig":
        """Load and validate robo config from a YAML file (defaults if missing).

        Any validation or parse failure is re-raised as a single, actionable
        :class:`ConfigError` (which key, the bound, the offending value) so a
        bad file can never crash a daemon with a raw traceback.
        """
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"{path} must contain a YAML mapping at the top level, "
                f"got {type(data).__name__}"
            )
        try:
            return cls(**data)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(path, exc)) from exc


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    """Render a pydantic ValidationError as one clear, actionable message.

    Names the dotted setting key, the offending value, and the violated bound
    for each problem so the operator can fix robo.yaml without reading a
    traceback.
    """
    lines = [f"invalid robo config in {path}:"]
    for err in exc.errors():
        key = ".".join(str(p) for p in err["loc"]) or "(root)"
        msg = err["msg"]
        if "input" in err:
            lines.append(f"  - {key}: {msg} (got {err['input']!r})")
        else:
            lines.append(f"  - {key}: {msg}")
    return "\n".join(lines)
