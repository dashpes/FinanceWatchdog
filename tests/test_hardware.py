"""Tests for cross-platform hardware detection and model recommendation."""

from __future__ import annotations

from unittest.mock import patch

from investment_monitor.analysis.hardware import (
    ModelRecommendation,
    recommend_models,
    total_ram_gb,
)


class TestRecommendModels:
    """Test the RAM-based model recommendation tiers."""

    def test_workstation_tier(self):
        rec = recommend_models(ram_gb=64.0)
        assert rec.tier == "workstation"
        assert rec.synthesis == "qwen2.5:72b"
        assert rec.ram_gb == 64.0

    def test_high_tier_48gb(self):
        # The reference M5 Pro / 48GB box should land in the "high" tier.
        rec = recommend_models(ram_gb=48.0)
        assert rec.tier == "high"
        assert rec.fast == "qwen2.5:7b"
        assert rec.synthesis == "qwen2.5:32b"

    def test_medium_tier(self):
        rec = recommend_models(ram_gb=16.0)
        assert rec.tier == "medium"
        assert rec.synthesis == "qwen2.5:14b"

    def test_low_tier(self):
        rec = recommend_models(ram_gb=8.0)
        assert rec.tier == "low"
        assert rec.fast == "llama3.1:8b"
        assert rec.synthesis == "qwen2.5:7b"

    def test_minimal_tier(self):
        rec = recommend_models(ram_gb=4.0)
        assert rec.tier == "minimal"
        assert rec.fast == "phi3:mini"
        assert rec.synthesis == "phi3:mini"

    def test_boundary_is_inclusive(self):
        # Exactly at a tier threshold should select that tier.
        assert recommend_models(ram_gb=28.0).tier == "high"
        assert recommend_models(ram_gb=27.9).tier == "medium"

    def test_returns_dataclass(self):
        rec = recommend_models(ram_gb=32.0)
        assert isinstance(rec, ModelRecommendation)

    def test_uses_fallback_when_detection_fails(self):
        # When detection returns None and no override is given, fall back safely.
        with patch(
            "investment_monitor.analysis.hardware.total_ram_gb", return_value=None
        ):
            rec = recommend_models()
            assert rec.tier == "low"  # _FALLBACK_RAM_GB == 8.0

    def test_uses_detected_ram_when_no_override(self):
        with patch(
            "investment_monitor.analysis.hardware.total_ram_gb", return_value=64.0
        ):
            rec = recommend_models()
            assert rec.tier == "workstation"


class TestTotalRamGb:
    """Test RAM detection returns a sane value on the host running the tests."""

    def test_detects_positive_ram(self):
        ram = total_ram_gb()
        # On any real CI/dev machine this should be a positive, plausible number.
        assert ram is None or ram > 0.5

    def test_handles_all_methods_failing(self):
        # If psutil import, platform detection, and sysconf all fail, return None.
        with patch("investment_monitor.analysis.hardware.platform.system",
                   return_value="Plan9"):
            with patch("investment_monitor.analysis.hardware.os.sysconf",
                       side_effect=OSError("nope")):
                # psutil may or may not be installed; force its path to fail too.
                import builtins

                real_import = builtins.__import__

                def fake_import(name, *args, **kwargs):
                    if name == "psutil":
                        raise ImportError("no psutil")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    assert total_ram_gb() is None
