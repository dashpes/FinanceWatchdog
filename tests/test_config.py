"""Tests for configuration loading."""

import pytest
from investment_monitor.config import Settings, get_settings


def test_settings_loads_defaults():
    """Settings should load with default values."""
    settings = Settings()
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "phi3:mini"


def test_get_settings_returns_settings():
    """get_settings should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_discord_webhook_url_default():
    """Test discord_webhook_url defaults to empty string."""
    settings = Settings()
    assert settings.discord_webhook_url == ""


def test_discord_webhook_url_from_env(monkeypatch):
    """Test discord_webhook_url can be set from environment."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/abc")
    settings = Settings()
    assert settings.discord_webhook_url == "https://discord.com/api/webhooks/123/abc"
