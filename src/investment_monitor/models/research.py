"""Research configuration models."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ScoringWeights(BaseModel):
    """Weights for composite score calculation.

    Each weight represents the importance of that factor in the overall
    investment decision. All weights must be between 0 and 1, and the
    total must sum to 1.0 (with small floating point tolerance).
    """

    value: float = Field(default=0.2, ge=0, le=1)
    growth: float = Field(default=0.2, ge=0, le=1)
    quality: float = Field(default=0.2, ge=0, le=1)
    momentum: float = Field(default=0.2, ge=0, le=1)
    sentiment: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def check_weights_sum(self) -> "ScoringWeights":
        """Validate that all weights sum to 1.0."""
        total = self.value + self.growth + self.quality + self.momentum + self.sentiment
        if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self


class UniverseConfig(BaseModel):
    """Configuration for the stock universe to screen.

    Controls which stocks are included in the research universe
    for discovery and screening.
    """

    include_sp500: bool = True
    include_nasdaq100: bool = True
    etf_tickers: list[str] = Field(default_factory=list)  # e.g., ["QQQ", "SPY", "VTI"]
    min_market_cap: int = Field(default=1_000_000_000, ge=0)  # $1B minimum
    excluded_sectors: list[str] = Field(default_factory=list)
    excluded_tickers: list[str] = Field(default_factory=list)


class ResearchThresholds(BaseModel):
    """Score thresholds for automated actions.

    Controls when stocks are automatically added to watchlist,
    rejected, or require manual review.
    """

    min_composite_score: float = Field(default=60.0, ge=0, le=100)
    auto_watchlist_score: float = Field(default=75.0, ge=0, le=100)
    auto_reject_score: float = Field(default=30.0, ge=0, le=100)


class ClaudeBudgetConfig(BaseModel):
    """Configuration for Claude API usage limits.

    Controls spending limits and rate limiting for AI-powered
    research report generation.
    """

    enabled: bool = True
    monthly_limit_usd: float = Field(default=50.0, ge=0)
    max_reports_per_day: int = Field(default=10, ge=1)


class ResearchConfig(BaseModel):
    """Main research configuration containing all research-related settings.

    This configuration controls:
    - Scoring weights for investment analysis
    - Universe of stocks to screen
    - Thresholds for automated decisions
    - Claude API budget limits
    """

    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    thresholds: ResearchThresholds = Field(default_factory=ResearchThresholds)
    claude_budget: ClaudeBudgetConfig = Field(default_factory=ClaudeBudgetConfig)
    discovery_batch_size: int = Field(default=50, ge=1, le=500)

    @classmethod
    def from_yaml(cls, path: Path) -> "ResearchConfig":
        """Load research config from YAML file.

        The YAML can contain partial configuration - any missing fields
        will use their default values.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
