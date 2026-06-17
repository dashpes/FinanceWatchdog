"""Configuration for the robo advisor.

Non-secret settings live in ``config/robo.yaml`` and are validated by
``RoboConfig``. The Public.com personal access token is a *secret* and is read
from the environment via the shared ``Settings`` (``PUBLIC_API_TOKEN``), never
from YAML.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from investment_monitor.robo.models import CASH_SYMBOL

# Sum of target weights must be within this tolerance of 1.0.
_ALLOCATION_TOLERANCE = Decimal("0.001")


class RoboCaps(BaseModel):
    """Rate and size caps enforced by the guardrail gate."""

    max_order_pct: float = Field(default=0.25, gt=0, le=1.0)
    max_orders_per_run: int = Field(default=5, ge=0)
    max_orders_per_day: int = Field(default=10, ge=0)
    # Fraction of an order's cost reserved for fees/slippage when checking affordability.
    fee_buffer: float = Field(default=0.01, ge=0, lt=1.0)

    # --- autonomous-mode safety guards (Phase 4). Permissive defaults = disabled,
    # so rebalance mode and existing behavior are unchanged unless these are set. ---
    max_positions: int = Field(default=0, ge=0)              # 0 = unlimited distinct holdings
    max_per_name_weight: float = Field(default=1.0, gt=0, le=1.0)  # 1.0 = no concentration cap
    max_turnover_pct: float = Field(default=0.0, ge=0)       # 0 = unlimited gross turnover per run
    max_drawdown_pct: float = Field(default=0.0, ge=0)       # 0 = drawdown circuit-breaker disabled


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
    max_position_weight: float = Field(default=0.15, gt=0, le=1.0)
    # Fractional-Kelly multiplier on the risk-adjusted (Sharpe) signal.
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0)
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
    # Always leave at least this fraction of the portfolio in cash (autonomous mode).
    min_cash_weight: float = Field(default=0.05, ge=0, lt=1.0)


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


class RoboConfig(BaseModel):
    """Validated robo-advisor configuration (from ``config/robo.yaml``)."""

    # symbol -> target weight (0..1). May include the pseudo-symbol "CASH".
    target_allocation: dict[str, float] = Field(default_factory=dict)
    # Only trade if a holding drifts more than this fraction from its target.
    rebalance_threshold: float = Field(default=0.05, ge=0, le=1.0)
    # Symbols the gate will allow trading. Defaults to target_allocation keys.
    allowlist: list[str] = Field(default_factory=list)

    caps: RoboCaps = Field(default_factory=RoboCaps)

    # Operating mode. "rebalance" (default) = fixed target_allocation, exactly
    # today's behavior. "autonomous" = conviction-driven weights from the thesis
    # store. The mode never affects dry-run/gate safety.
    mode: Literal["rebalance", "autonomous"] = "rebalance"

    # Event-driven signal settings (Phase 2). Disabled by default.
    signals: SignalConfig = Field(default_factory=SignalConfig)

    # Risk-adjusted conviction sizing (Phase 3, used in autonomous mode).
    sizing: SizingConfig = Field(default_factory=SizingConfig)

    # Autonomous stock selection from the discovery funnel (Phase 4). Disabled by default.
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)

    # Safety: simulate everything and place no real orders when True. Default True.
    dry_run: bool = True
    # Only PLACE live orders during US market hours (research/maintenance still run
    # 24/7; off-hours live runs propose + gate but defer placement). Default True.
    require_market_hours: bool = True
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

    @field_validator("allowlist")
    @classmethod
    def _upper_allowlist(cls, v: list[str]) -> list[str]:
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
        """Load and validate robo config from a YAML file (defaults if missing)."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
