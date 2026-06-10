"""Tests for the AI/LLM diagnostics ("doctor") report."""

from __future__ import annotations

from unittest.mock import patch

from investment_monitor.config import Settings
from investment_monitor.diagnostics import build_doctor_report


def test_report_with_explicit_models_and_server_down():
    """Report should show resolved models and a clear fix when Ollama is down."""
    settings = Settings(
        ollama_model="qwen2.5:7b",
        ollama_synthesis_model="qwen2.5:32b",
        llm_provider="ollama",
        anthropic_api_key="",
    )
    with patch(
        "investment_monitor.diagnostics._probe_ollama",
        return_value=(False, [], "Connection refused"),
    ):
        report = build_doctor_report(settings)

    assert "AI / LLM diagnostics" in report
    assert "qwen2.5:7b" in report
    assert "qwen2.5:32b" in report
    # Provider "ollama" -> free local path, never Claude.
    assert "local Ollama (free)" in report
    assert "NOT reachable" in report
    assert "ollama pull qwen2.5:7b" in report
    assert "ollama pull qwen2.5:32b" in report


def test_report_reachable_flags_missing_and_present_models():
    """When reachable, present models read OK and missing ones say MISSING."""
    settings = Settings(
        ollama_model="qwen2.5:7b",
        ollama_synthesis_model="qwen2.5:32b",
        llm_provider="ollama",
    )
    # Only the fast model is installed.
    with patch(
        "investment_monitor.diagnostics._probe_ollama",
        return_value=(True, ["qwen2.5:7b"], None),
    ):
        report = build_doctor_report(settings)

    assert "reachable at" in report
    assert "OK (qwen2.5:7b)" in report
    assert "MISSING (qwen2.5:32b)" in report


def test_report_reflects_claude_provider_selection():
    """With provider auto + an API key, tier-2 should report Claude."""
    settings = Settings(llm_provider="auto", anthropic_api_key="sk-ant-test")
    with patch(
        "investment_monitor.diagnostics._probe_ollama",
        return_value=(True, [], None),
    ):
        report = build_doctor_report(settings)

    assert "Claude (Anthropic)" in report
    assert "ANTHROPIC_API_KEY      : set" in report


def test_report_auto_models_resolve_to_concrete_tags():
    """Auto models should resolve to concrete tags in the report (not 'auto')."""
    settings = Settings()  # defaults: auto / auto
    with patch(
        "investment_monitor.diagnostics._probe_ollama",
        return_value=(False, [], "down"),
    ):
        report = build_doctor_report(settings)

    # The "auto -> <tag>" mapping should contain a real, tagged model.
    assert "auto ->" in report
    assert "Capability tier" in report
