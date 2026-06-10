"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MonteCarloSettings(BaseModel):
    """Monte Carlo simulation settings."""

    score_threshold: float = 80.0
    horizons: list[int] = Field(default_factory=lambda: [30, 90, 252])
    min_paths: int = 1000
    max_paths: int = 50000
    ci_width_threshold: float = 0.15
    min_lookback_days: int = 252
    max_lookback_days: int = 1260
    volatility_multipliers: list[float] = Field(
        default_factory=lambda: [0.5, 0.8, 1.0, 1.2, 1.5]
    )
    drift_scenarios: list[str] = Field(
        default_factory=lambda: ["pessimistic", "neutral", "optimistic"]
    )
    include_in_reports: bool = True
    disclaimer: str = "Simulation based on historical returns. Not a prediction."

    # Scenario toggles
    scenarios: dict[str, bool] = Field(
        default_factory=lambda: {
            "base_gbm": True,
            "crisis_2008": True,
            "dotcom_crash": True,
            "covid_crash": True,
            "stagflation_1970s": True,
            "black_monday_1987": True,
            "rising_rates_2022": True,
            "regime_democrat": True,
            "regime_republican": True,
        }
    )


class Settings(BaseSettings):
    """Main application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (optional)
    sendgrid_api_key: str = ""
    slack_webhook_url: str = ""
    anthropic_api_key: str = ""
    finnhub_api_key: str = ""

    # Discord - supports separate channels for daily vs weekly
    # If only discord_webhook_url is set, it's used for both
    # If daily/weekly specific URLs are set, they take precedence
    discord_webhook_url: str = ""
    discord_daily_webhook_url: str = ""
    discord_weekly_webhook_url: str = ""

    # Ollama (local LLM)
    ollama_host: str = "http://localhost:11434"
    # Model names may be set to "auto" to let the system pick a model that fits
    # the host's RAM (see analysis.hardware.recommend_models). Set an explicit
    # tag (e.g. "qwen2.5:7b") to override.
    ollama_model: str = "auto"  # fast/tier-1 model: news relevance, sentiment, scoring
    ollama_synthesis_model: str = "auto"  # tier-2 model: weekly synthesis, research reports

    # Which provider handles tier-2 synthesis/reports:
    #   "auto"      -> use Claude when an Anthropic API key is set, else local Ollama
    #   "ollama"    -> always local (completely free, no API key needed)
    #   "anthropic" -> always Claude (requires anthropic_api_key)
    llm_provider: str = "auto"

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # Database
    db_path: Path = Path("data/portfolio.db")

    def resolved_ollama_model(self) -> str:
        """Return the tier-1 (fast) Ollama model, resolving "auto" by RAM."""
        if self.ollama_model and self.ollama_model.lower() != "auto":
            return self.ollama_model
        from .analysis.hardware import recommend_models

        return recommend_models().fast

    def resolved_synthesis_model(self) -> str:
        """Return the tier-2 (synthesis) Ollama model, resolving "auto" by RAM."""
        if self.ollama_synthesis_model and self.ollama_synthesis_model.lower() != "auto":
            return self.ollama_synthesis_model
        from .analysis.hardware import recommend_models

        return recommend_models().synthesis

    def prefer_anthropic_synthesis(self) -> bool:
        """Whether tier-2 synthesis/reports should use Claude instead of Ollama.

        Honors ``llm_provider``: "anthropic" forces Claude, "ollama" forces local,
        and "auto" uses Claude only when an Anthropic API key is configured.
        Local Ollama is the free default whenever Claude is not selected.
        """
        provider = (self.llm_provider or "auto").lower()
        if provider == "ollama":
            return False
        if provider == "anthropic":
            return True
        return bool(self.anthropic_api_key)


def load_yaml_config(config_dir: Path, filename: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = config_dir / filename
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
