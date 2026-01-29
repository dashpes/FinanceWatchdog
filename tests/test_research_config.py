"""Tests for research configuration models."""

import tempfile
from pathlib import Path

import pytest

from investment_monitor.models import (
    ClaudeBudgetConfig,
    ResearchConfig,
    ResearchThresholds,
    ScoringWeights,
    UniverseConfig,
)


# ============================================================================
# ScoringWeights Tests
# ============================================================================


class TestScoringWeights:
    """Tests for ScoringWeights model."""

    def test_defaults(self):
        """Test default values are balanced (equal weights)."""
        weights = ScoringWeights()
        assert weights.value == 0.2
        assert weights.growth == 0.2
        assert weights.quality == 0.2
        assert weights.momentum == 0.2
        assert weights.sentiment == 0.2

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        weights = ScoringWeights()
        total = weights.value + weights.growth + weights.quality + weights.momentum + weights.sentiment
        assert total == 1.0

    def test_custom_weights_valid(self):
        """Test custom weights that sum to 1.0 are accepted."""
        weights = ScoringWeights(
            value=0.3,
            growth=0.25,
            quality=0.2,
            momentum=0.15,
            sentiment=0.1,
        )
        assert weights.value == 0.3
        assert weights.growth == 0.25
        assert weights.quality == 0.2
        assert weights.momentum == 0.15
        assert weights.sentiment == 0.1

    def test_weights_sum_validation_too_high(self):
        """Test that weights summing to more than 1.0 are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ScoringWeights(
                value=0.3,
                growth=0.3,
                quality=0.3,
                momentum=0.3,
                sentiment=0.3,
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_weights_sum_validation_too_low(self):
        """Test that weights summing to less than 1.0 are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ScoringWeights(
                value=0.1,
                growth=0.1,
                quality=0.1,
                momentum=0.1,
                sentiment=0.1,
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_weights_floating_point_tolerance(self):
        """Test that small floating point errors are tolerated."""
        # These might not sum to exactly 1.0 due to floating point
        weights = ScoringWeights(
            value=0.33,
            growth=0.17,
            quality=0.17,
            momentum=0.17,
            sentiment=0.16,
        )
        # Should not raise - within tolerance
        assert weights is not None

    def test_individual_weight_negative_rejected(self):
        """Test that negative individual weights are rejected."""
        with pytest.raises(ValueError):
            ScoringWeights(value=-0.1, growth=0.3, quality=0.3, momentum=0.3, sentiment=0.2)

    def test_individual_weight_over_one_rejected(self):
        """Test that individual weights over 1.0 are rejected."""
        with pytest.raises(ValueError):
            ScoringWeights(value=1.1)

    def test_boundary_values(self):
        """Test boundary values (0 and 1) for individual weights."""
        weights = ScoringWeights(
            value=1.0,
            growth=0.0,
            quality=0.0,
            momentum=0.0,
            sentiment=0.0,
        )
        assert weights.value == 1.0
        assert weights.growth == 0.0

    def test_value_investor_profile(self):
        """Test a value investor weight profile."""
        weights = ScoringWeights(
            value=0.4,
            growth=0.1,
            quality=0.3,
            momentum=0.1,
            sentiment=0.1,
        )
        assert weights.value == 0.4
        assert weights.quality == 0.3

    def test_growth_investor_profile(self):
        """Test a growth investor weight profile."""
        weights = ScoringWeights(
            value=0.1,
            growth=0.4,
            quality=0.2,
            momentum=0.2,
            sentiment=0.1,
        )
        assert weights.growth == 0.4


# ============================================================================
# UniverseConfig Tests
# ============================================================================


class TestUniverseConfig:
    """Tests for UniverseConfig model."""

    def test_defaults(self):
        """Test default values are sensible."""
        config = UniverseConfig()
        assert config.include_sp500 is True
        assert config.include_nasdaq100 is True
        assert config.etf_tickers == []
        assert config.min_market_cap == 1_000_000_000
        assert config.excluded_sectors == []
        assert config.excluded_tickers == []

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = UniverseConfig(
            include_sp500=False,
            include_nasdaq100=True,
            etf_tickers=["QQQ", "SPY", "VTI"],
            min_market_cap=10_000_000_000,
            excluded_sectors=["Energy", "Utilities"],
            excluded_tickers=["XOM", "CVX"],
        )
        assert config.include_sp500 is False
        assert config.include_nasdaq100 is True
        assert config.etf_tickers == ["QQQ", "SPY", "VTI"]
        assert config.min_market_cap == 10_000_000_000
        assert config.excluded_sectors == ["Energy", "Utilities"]
        assert config.excluded_tickers == ["XOM", "CVX"]

    def test_zero_market_cap_allowed(self):
        """Test that zero market cap is allowed."""
        config = UniverseConfig(min_market_cap=0)
        assert config.min_market_cap == 0

    def test_negative_market_cap_rejected(self):
        """Test that negative market cap is rejected."""
        with pytest.raises(ValueError):
            UniverseConfig(min_market_cap=-1)

    def test_disable_both_indices(self):
        """Test that both indices can be disabled."""
        config = UniverseConfig(include_sp500=False, include_nasdaq100=False)
        assert config.include_sp500 is False
        assert config.include_nasdaq100 is False

    def test_empty_lists_default(self):
        """Test that list fields default to empty lists."""
        config = UniverseConfig()
        # Ensure they're actually mutable (not shared)
        config.etf_tickers.append("TEST")
        another_config = UniverseConfig()
        assert "TEST" not in another_config.etf_tickers


# ============================================================================
# ResearchThresholds Tests
# ============================================================================


class TestResearchThresholds:
    """Tests for ResearchThresholds model."""

    def test_defaults(self):
        """Test default values are sensible."""
        thresholds = ResearchThresholds()
        assert thresholds.min_composite_score == 60.0
        assert thresholds.auto_watchlist_score == 75.0
        assert thresholds.auto_reject_score == 30.0

    def test_custom_values(self):
        """Test custom values are accepted."""
        thresholds = ResearchThresholds(
            min_composite_score=50.0,
            auto_watchlist_score=80.0,
            auto_reject_score=25.0,
        )
        assert thresholds.min_composite_score == 50.0
        assert thresholds.auto_watchlist_score == 80.0
        assert thresholds.auto_reject_score == 25.0

    def test_negative_score_rejected(self):
        """Test that negative scores are rejected."""
        with pytest.raises(ValueError):
            ResearchThresholds(min_composite_score=-1.0)

    def test_score_over_100_rejected(self):
        """Test that scores over 100 are rejected."""
        with pytest.raises(ValueError):
            ResearchThresholds(auto_watchlist_score=101.0)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        thresholds = ResearchThresholds(
            min_composite_score=0.0,
            auto_watchlist_score=100.0,
            auto_reject_score=0.0,
        )
        assert thresholds.min_composite_score == 0.0
        assert thresholds.auto_watchlist_score == 100.0
        assert thresholds.auto_reject_score == 0.0


# ============================================================================
# ClaudeBudgetConfig Tests
# ============================================================================


class TestClaudeBudgetConfig:
    """Tests for ClaudeBudgetConfig model."""

    def test_defaults(self):
        """Test default values are sensible."""
        config = ClaudeBudgetConfig()
        assert config.enabled is True
        assert config.monthly_limit_usd == 50.0
        assert config.max_reports_per_day == 10

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = ClaudeBudgetConfig(
            enabled=False,
            monthly_limit_usd=100.0,
            max_reports_per_day=20,
        )
        assert config.enabled is False
        assert config.monthly_limit_usd == 100.0
        assert config.max_reports_per_day == 20

    def test_zero_budget_allowed(self):
        """Test that zero budget is allowed."""
        config = ClaudeBudgetConfig(monthly_limit_usd=0.0)
        assert config.monthly_limit_usd == 0.0

    def test_negative_budget_rejected(self):
        """Test that negative budget is rejected."""
        with pytest.raises(ValueError):
            ClaudeBudgetConfig(monthly_limit_usd=-1.0)

    def test_zero_reports_rejected(self):
        """Test that zero reports per day is rejected."""
        with pytest.raises(ValueError):
            ClaudeBudgetConfig(max_reports_per_day=0)

    def test_minimum_reports_boundary(self):
        """Test that 1 report per day is allowed."""
        config = ClaudeBudgetConfig(max_reports_per_day=1)
        assert config.max_reports_per_day == 1

    def test_disabled_still_validates(self):
        """Test that disabled config still validates other fields."""
        with pytest.raises(ValueError):
            ClaudeBudgetConfig(enabled=False, max_reports_per_day=0)


# ============================================================================
# ResearchConfig Tests
# ============================================================================


class TestResearchConfig:
    """Tests for ResearchConfig model."""

    def test_defaults(self):
        """Test default values create valid nested settings."""
        config = ResearchConfig()
        assert config.scoring_weights.value == 0.2
        assert config.universe.include_sp500 is True
        assert config.thresholds.min_composite_score == 60.0
        assert config.claude_budget.enabled is True
        assert config.discovery_batch_size == 50

    def test_custom_batch_size(self):
        """Test custom batch size is accepted."""
        config = ResearchConfig(discovery_batch_size=100)
        assert config.discovery_batch_size == 100

    def test_batch_size_minimum(self):
        """Test batch size minimum of 1."""
        config = ResearchConfig(discovery_batch_size=1)
        assert config.discovery_batch_size == 1

    def test_batch_size_maximum(self):
        """Test batch size maximum of 500."""
        config = ResearchConfig(discovery_batch_size=500)
        assert config.discovery_batch_size == 500

    def test_batch_size_below_minimum_rejected(self):
        """Test that batch size below 1 is rejected."""
        with pytest.raises(ValueError):
            ResearchConfig(discovery_batch_size=0)

    def test_batch_size_above_maximum_rejected(self):
        """Test that batch size above 500 is rejected."""
        with pytest.raises(ValueError):
            ResearchConfig(discovery_batch_size=501)

    def test_partial_override(self):
        """Test that partial configuration merges with defaults."""
        config = ResearchConfig(
            scoring_weights=ScoringWeights(value=0.3, growth=0.2, quality=0.2, momentum=0.15, sentiment=0.15),
            universe=UniverseConfig(include_sp500=False),
        )
        # Overridden values
        assert config.scoring_weights.value == 0.3
        assert config.universe.include_sp500 is False

        # Default values preserved
        assert config.universe.include_nasdaq100 is True  # default
        assert config.thresholds.min_composite_score == 60.0  # entire section defaults
        assert config.claude_budget.enabled is True  # entire section defaults

    def test_from_yaml_full_config(self):
        """Test loading a full configuration from YAML."""
        yaml_content = """
scoring_weights:
  value: 0.3
  growth: 0.25
  quality: 0.2
  momentum: 0.15
  sentiment: 0.1

universe:
  include_sp500: true
  include_nasdaq100: false
  etf_tickers:
    - QQQ
    - VTI
  min_market_cap: 5000000000
  excluded_sectors:
    - Energy
  excluded_tickers:
    - XOM

thresholds:
  min_composite_score: 55.0
  auto_watchlist_score: 80.0
  auto_reject_score: 25.0

claude_budget:
  enabled: true
  monthly_limit_usd: 75.0
  max_reports_per_day: 15

discovery_batch_size: 75
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ResearchConfig.from_yaml(Path(f.name))

            # Scoring weights
            assert config.scoring_weights.value == 0.3
            assert config.scoring_weights.growth == 0.25
            assert config.scoring_weights.quality == 0.2
            assert config.scoring_weights.momentum == 0.15
            assert config.scoring_weights.sentiment == 0.1

            # Universe settings
            assert config.universe.include_sp500 is True
            assert config.universe.include_nasdaq100 is False
            assert config.universe.etf_tickers == ["QQQ", "VTI"]
            assert config.universe.min_market_cap == 5_000_000_000
            assert config.universe.excluded_sectors == ["Energy"]
            assert config.universe.excluded_tickers == ["XOM"]

            # Thresholds
            assert config.thresholds.min_composite_score == 55.0
            assert config.thresholds.auto_watchlist_score == 80.0
            assert config.thresholds.auto_reject_score == 25.0

            # Claude budget
            assert config.claude_budget.enabled is True
            assert config.claude_budget.monthly_limit_usd == 75.0
            assert config.claude_budget.max_reports_per_day == 15

            # Batch size
            assert config.discovery_batch_size == 75

    def test_from_yaml_partial_config(self):
        """Test loading a partial configuration from YAML merges with defaults."""
        yaml_content = """
scoring_weights:
  value: 0.4
  growth: 0.15
  quality: 0.15
  momentum: 0.15
  sentiment: 0.15

thresholds:
  min_composite_score: 70.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ResearchConfig.from_yaml(Path(f.name))

            # Specified values
            assert config.scoring_weights.value == 0.4
            assert config.thresholds.min_composite_score == 70.0

            # Defaults for other sections
            assert config.universe.include_sp500 is True
            assert config.universe.min_market_cap == 1_000_000_000
            assert config.claude_budget.enabled is True
            assert config.claude_budget.monthly_limit_usd == 50.0
            assert config.discovery_batch_size == 50

    def test_from_yaml_empty_file(self):
        """Test loading from an empty YAML file uses all defaults."""
        yaml_content = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ResearchConfig.from_yaml(Path(f.name))

            # All defaults
            assert config.scoring_weights.value == 0.2
            assert config.universe.include_sp500 is True
            assert config.thresholds.min_composite_score == 60.0
            assert config.claude_budget.enabled is True
            assert config.discovery_batch_size == 50

    def test_from_yaml_invalid_weights_rejected(self):
        """Test that invalid weights in YAML are rejected."""
        yaml_content = """
scoring_weights:
  value: 0.5
  growth: 0.5
  quality: 0.5
  momentum: 0.5
  sentiment: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError) as exc_info:
                ResearchConfig.from_yaml(Path(f.name))
            assert "must sum to 1.0" in str(exc_info.value)

    def test_from_yaml_invalid_threshold_rejected(self):
        """Test that invalid threshold values in YAML are rejected."""
        yaml_content = """
thresholds:
  min_composite_score: 150.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError):
                ResearchConfig.from_yaml(Path(f.name))

    def test_from_yaml_invalid_market_cap_rejected(self):
        """Test that negative market cap in YAML is rejected."""
        yaml_content = """
universe:
  min_market_cap: -1000000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError):
                ResearchConfig.from_yaml(Path(f.name))
