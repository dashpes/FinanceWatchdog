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

    # iMessage (macOS only) — the robo advisor texts you on live trades, run
    # errors, and a daily summary. Recipient is your phone number (e.g.
    # +15551234567) or the Apple ID email registered with iMessage. Blank = off.
    imessage_to: str = ""
    # Also notify on paper/dry-run placements. Off by default to avoid noise; live
    # trades and run errors always notify regardless of this flag.
    imessage_notify_paper: bool = False

    # Email (SMTP) notifications — a headless transport for the robo advisor that
    # works from a background launchd daemon (no GUI, no Messages.app, no Automation
    # grant), unlike iMessage. Used for live trades, run errors, and the daily
    # summary, and PREFERRED over iMessage when configured. Blank smtp_host or
    # email_to disables it. Gmail: SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, and an
    # App Password (not your login password) as SMTP_PASSWORD.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""  # falls back to smtp_username when blank
    email_to: str = ""
    email_use_tls: bool = True

    # SEC EDGAR requires a real contact in the User-Agent of every request (a name
    # or app plus an email). Set to YOUR email at onboarding. Blank falls back to a
    # generic contact so a fresh install still works, but you should set it — SEC may
    # rate-limit or block an anonymous/placeholder agent.
    sec_contact_email: str = ""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # Per-role model routing (Phase 5). A small/fast model triages; a stronger
    # model synthesizes theses; a dedicated model produces embeddings. Any role left
    # blank falls back to ollama_model. Override individual roles in .env, e.g.
    # MODEL_ROLES__SYNTHESIS=qwen2.5:14b
    model_roles: dict[str, str] = Field(
        default_factory=lambda: {
            "triage": "phi3:mini",
            "synthesis": "qwen2.5:14b",
            "scoring": "phi3:mini",
            "embedding": "nomic-embed-text",
        }
    )

    # Public.com robo advisor (trading) — secret token from env only
    public_api_token: str = ""
    public_api_base_url: str = ""  # blank = SDK/library default (production)
    # Hard kill-switch for the robo advisor's live trading, independent of robo.yaml.
    # When True, no real orders are ever placed regardless of config. Default True.
    robo_force_dry_run: bool = True

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # Database
    db_path: Path = Path("data/portfolio.db")

    # Data-retention windows (days) for the robo `prune` job — 0 keeps everything.
    # Broad collection is universe-independent and never drops rows on its own, so on
    # an always-on box (the Pi) these bound SQLite growth. Overridable per source in
    # .env, e.g. RETENTION_NEWS_DAYS=90. The prune timer applies them weekly.
    retention_insider_days: int = 180
    retention_news_days: int = 180
    retention_price_days: int = 180
    retention_findings_days: int = 365


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
