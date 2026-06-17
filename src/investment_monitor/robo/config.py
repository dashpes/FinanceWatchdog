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


class RoboConfig(BaseModel):
    """Validated robo-advisor configuration (from ``config/robo.yaml``)."""

    # symbol -> target weight (0..1). May include the pseudo-symbol "CASH".
    target_allocation: dict[str, float] = Field(default_factory=dict)
    # Only trade if a holding drifts more than this fraction from its target.
    rebalance_threshold: float = Field(default=0.05, ge=0, le=1.0)
    # Symbols the gate will allow trading. Defaults to target_allocation keys.
    allowlist: list[str] = Field(default_factory=list)

    caps: RoboCaps = Field(default_factory=RoboCaps)

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
