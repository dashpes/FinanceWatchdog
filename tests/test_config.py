"""Tests for configuration loading."""

import pytest
from investment_monitor.config import Settings, get_settings


def test_settings_loads_defaults():
    """Settings should load with default values."""
    settings = Settings()
    assert settings.ollama_host == "http://localhost:11434"
    # Models default to "auto" (RAM-based selection); provider defaults to "auto".
    assert settings.ollama_model == "auto"
    assert settings.ollama_synthesis_model == "auto"
    assert settings.llm_provider == "auto"


def test_resolved_ollama_model_uses_explicit_override():
    """An explicit model tag should be returned verbatim."""
    settings = Settings(ollama_model="qwen2.5:7b", ollama_synthesis_model="qwen2.5:32b")
    assert settings.resolved_ollama_model() == "qwen2.5:7b"
    assert settings.resolved_synthesis_model() == "qwen2.5:32b"


def test_resolved_models_auto_returns_real_tags():
    """When set to "auto", resolution returns a concrete model tag, not "auto"."""
    settings = Settings()  # defaults to auto
    assert settings.resolved_ollama_model() != "auto"
    assert ":" in settings.resolved_ollama_model()
    assert settings.resolved_synthesis_model() != "auto"


def test_prefer_anthropic_synthesis_logic():
    """Provider selection should honor llm_provider and key presence."""
    # auto + no key -> local (free)
    assert Settings(llm_provider="auto", anthropic_api_key="").prefer_anthropic_synthesis() is False
    # auto + key -> Claude
    assert Settings(llm_provider="auto", anthropic_api_key="sk-ant-x").prefer_anthropic_synthesis() is True
    # ollama -> always local, even with a key
    assert Settings(llm_provider="ollama", anthropic_api_key="sk-ant-x").prefer_anthropic_synthesis() is False
    # anthropic -> always Claude
    assert Settings(llm_provider="anthropic", anthropic_api_key="").prefer_anthropic_synthesis() is True


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
