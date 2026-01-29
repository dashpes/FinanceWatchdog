"""Alert configuration models."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PriceAlertSettings(BaseModel):
    """Configuration for price-based alerts."""

    enabled: bool = True
    daily_drop_pct: float = Field(default=3.0, ge=0, le=100)
    daily_rise_pct: float = Field(default=5.0, ge=0, le=100)
    weekly_drop_pct: float = Field(default=7.0, ge=0, le=100)
    below_cost_basis: bool = True


class VolumeAlertSettings(BaseModel):
    """Configuration for volume-based alerts."""

    enabled: bool = True
    lookback_days: int = Field(default=20, ge=5, le=60)
    multiplier: float = Field(default=2.5, ge=1.0)


class InsiderAlertSettings(BaseModel):
    """Configuration for insider trading alerts."""

    enabled: bool = True
    min_buy_value: int = Field(default=100_000, ge=0)
    min_sell_value: int = Field(default=500_000, ge=0)
    alert_ceo_cfo_any: bool = True
    cluster_threshold: int = Field(default=3, ge=2)
    cluster_days: int = Field(default=7, ge=1)


class EarningsAlertSettings(BaseModel):
    """Configuration for earnings-related alerts."""

    enabled: bool = True
    lookahead_days: int = Field(default=7, ge=1, le=30)


class NewsAlertSettings(BaseModel):
    """Configuration for news-based alerts."""

    enabled: bool = True
    keywords: list[str] = Field(
        default_factory=lambda: [
            "lawsuit",
            "SEC",
            "investigation",
            "guidance",
            "acquisition",
            "merger",
            "layoffs",
            "dividend",
            "buyback",
        ]
    )
    min_relevance_score: float = Field(default=5.0, ge=0, le=10)


class ETFAlertSettings(BaseModel):
    """Configuration for ETF-related alerts."""

    enabled: bool = True
    holdings_change: bool = True
    weight_change_pct: float = Field(default=1.0, ge=0)


class AlertsConfig(BaseModel):
    """Main alerts configuration containing all alert type settings."""

    price: PriceAlertSettings = Field(default_factory=PriceAlertSettings)
    volume: VolumeAlertSettings = Field(default_factory=VolumeAlertSettings)
    insider: InsiderAlertSettings = Field(default_factory=InsiderAlertSettings)
    earnings: EarningsAlertSettings = Field(default_factory=EarningsAlertSettings)
    news: NewsAlertSettings = Field(default_factory=NewsAlertSettings)
    etf: ETFAlertSettings = Field(default_factory=ETFAlertSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "AlertsConfig":
        """Load alerts config from YAML file.

        The YAML can contain partial configuration - any missing fields
        will use their default values.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
