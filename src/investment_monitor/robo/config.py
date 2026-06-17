"""Configuration for the robo advisor.

Non-secret settings live in ``config/robo.yaml`` and are validated by
``RoboConfig``. The Public.com personal access token is a *secret* and is read
from the environment via the shared ``Settings`` (``PUBLIC_API_TOKEN``), never
from YAML.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

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


class RoboConfig(BaseModel):
    """Validated robo-advisor configuration (from ``config/robo.yaml``)."""

    # symbol -> target weight (0..1). May include the pseudo-symbol "CASH".
    target_allocation: dict[str, float] = Field(default_factory=dict)
    # Only trade if a holding drifts more than this fraction from its target.
    rebalance_threshold: float = Field(default=0.05, ge=0, le=1.0)
    # Symbols the gate will allow trading. Defaults to target_allocation keys.
    allowlist: list[str] = Field(default_factory=list)

    caps: RoboCaps = Field(default_factory=RoboCaps)

    # Event-driven signal settings (Phase 2). Disabled by default.
    signals: SignalConfig = Field(default_factory=SignalConfig)

    # Safety: simulate everything and place no real orders when True. Default True.
    dry_run: bool = True
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
