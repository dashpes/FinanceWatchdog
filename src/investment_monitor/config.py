"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Discord
    discord_webhook_url: str = ""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # Database
    db_path: Path = Path("data/portfolio.db")


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
