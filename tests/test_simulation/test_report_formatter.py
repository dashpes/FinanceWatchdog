"""Tests for SimulationReportFormatter - formatting simulation output for reports."""

import pytest

from investment_monitor.simulation.models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationOutput,
)
from investment_monitor.simulation.report_formatter import SimulationReportFormatter


@pytest.fixture
def sample_simulation_output() -> SimulationOutput:
    """Create a realistic SimulationOutput for testing."""
    # Create scenario results for stress testing
    scenarios = {
        "2008 Financial Crisis": ScenarioResult(
            name="2008 Financial Crisis",
            mean=124.0,
            median=122.0,
            std=25.0,
            ci_80=(98.0, 142.0),
            ci_95=(85.0, 155.0),
            var_95=-0.305,
            cvar_95=-0.38,
            prob_loss_20pct=0.68,
        ),
        "Dot-com Crash": ScenarioResult(
            name="Dot-com Crash",
            mean=131.0,
            median=128.0,
            std=22.0,
            ci_80=(105.0, 152.0),
            ci_95=(90.0, 165.0),
            var_95=-0.265,
            cvar_95=-0.33,
            prob_loss_20pct=0.54,
        ),
        "COVID Crash": ScenarioResult(
            name="COVID Crash",
            mean=156.0,
            median=154.0,
            std=18.0,
            ci_80=(134.0, 178.0),
            ci_95=(118.0, 190.0),
            var_95=-0.185,
            cvar_95=-0.24,
            prob_loss_20pct=0.31,
        ),
    }

    # 30-day horizon
    horizon_30 = HorizonResult(
        days=30,
        base_mean=182.0,
        base_median=181.5,
        base_std=12.0,
        base_skewness=0.15,
        base_percentiles={5: 165.0, 25: 174.0, 50: 181.5, 75: 189.0, 95: 202.0},
        base_ci_80=(171.0, 188.0),
        base_ci_95=(165.0, 195.0),
        base_var_95=-0.08,
        base_cvar_95=-0.11,
        scenarios=scenarios,
    )

    # 90-day horizon
    horizon_90 = HorizonResult(
        days=90,
        base_mean=189.0,
        base_median=188.0,
        base_std=18.0,
        base_skewness=0.2,
        base_percentiles={5: 152.0, 25: 175.0, 50: 188.0, 75: 202.0, 95: 220.0},
        base_ci_80=(162.0, 201.0),
        base_ci_95=(152.0, 215.0),
        base_var_95=-0.15,
        base_cvar_95=-0.20,
        scenarios=scenarios,
    )

    # 252-day (1 year) horizon
    horizon_252 = HorizonResult(
        days=252,
        base_mean=198.0,
        base_median=195.0,
        base_std=28.0,
        base_skewness=0.25,
        base_percentiles={5: 138.0, 25: 170.0, 50: 195.0, 75: 218.0, 95: 252.0},
        base_ci_80=(149.0, 224.0),
        base_ci_95=(138.0, 245.0),
        base_var_95=-0.189,
        base_cvar_95=-0.242,
        scenarios=scenarios,
    )

    sensitivity = SensitivityResult(
        volatility_impact=85.0,
        drift_impact=32.0,
        lookback_impact=22.0,
        primary_driver="volatility",
        volatility_range={0.5: 190.0, 0.8: 195.0, 1.0: 198.0, 1.2: 202.0, 1.5: 210.0},
        drift_range={"pessimistic": 175.0, "neutral": 198.0, "optimistic": 220.0},
        lookback_range={252: 198.0, 504: 195.0, 756: 192.0},
    )

    return SimulationOutput(
        ticker="AAPL",
        entry_price=178.50,
        composite_score=85.0,
        num_simulations=10000,
        lookback_days=806,  # ~3.2 years
        volatility=0.25,
        drift=0.12,
        results={30: horizon_30, 90: horizon_90, 252: horizon_252},
        sensitivity=sensitivity,
    )


@pytest.fixture
def formatter() -> SimulationReportFormatter:
    """Create a SimulationReportFormatter instance."""
    return SimulationReportFormatter()


class TestFormatMarkdownContainsHeader:
    """Test that markdown output contains the expected header."""

    def test_format_markdown_contains_header(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes the main header."""
        result = formatter.format_markdown(sample_simulation_output)

        assert "## Risk Analysis (Monte Carlo Simulation)" in result


class TestFormatMarkdownContainsEntryInfo:
    """Test that markdown output contains entry point information."""

    def test_format_markdown_contains_entry_info(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes entry price and simulation count."""
        result = formatter.format_markdown(sample_simulation_output)

        # Entry price should be formatted with dollar sign
        assert "$178.50" in result
        # Simulation count should be formatted with commas
        assert "10,000" in result


class TestFormatMarkdownContainsHorizons:
    """Test that markdown output contains all time horizons."""

    def test_format_markdown_contains_horizons(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes 30 days and 1 year horizons."""
        result = formatter.format_markdown(sample_simulation_output)

        # Should contain readable horizon labels
        assert "30 days" in result
        assert "1 year" in result


class TestFormatMarkdownContainsStressScenarios:
    """Test that markdown output contains stress test scenarios."""

    def test_format_markdown_contains_stress_scenarios(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes 2008 Financial Crisis scenario."""
        result = formatter.format_markdown(sample_simulation_output)

        assert "2008 Financial Crisis" in result


class TestFormatMarkdownContainsRiskMetrics:
    """Test that markdown output contains risk metrics."""

    def test_format_markdown_contains_risk_metrics(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes VaR (Value at Risk)."""
        result = formatter.format_markdown(sample_simulation_output)

        assert "VaR" in result or "Value at Risk" in result


class TestFormatMarkdownContainsSensitivity:
    """Test that markdown output contains sensitivity analysis."""

    def test_format_markdown_contains_sensitivity(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes volatility sensitivity."""
        result = formatter.format_markdown(sample_simulation_output)

        assert "volatility" in result.lower() or "Volatility" in result


class TestFormatMarkdownContainsDisclaimer:
    """Test that markdown output contains disclaimer."""

    def test_format_markdown_contains_disclaimer(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown includes a disclaimer."""
        result = formatter.format_markdown(sample_simulation_output)

        # Default disclaimer should contain typical financial disclaimer text
        assert "simulation" in result.lower() or "Simulation" in result
        assert "historical" in result.lower() or "not a prediction" in result.lower()

    def test_format_markdown_contains_custom_disclaimer(
        self, sample_simulation_output: SimulationOutput
    ):
        """Test that format_markdown uses a custom disclaimer when provided."""
        custom_disclaimer = "Custom disclaimer: Past performance is not indicative of future results."
        formatter = SimulationReportFormatter(disclaimer=custom_disclaimer)

        result = formatter.format_markdown(sample_simulation_output)

        assert custom_disclaimer in result


class TestFormatCompact:
    """Test that compact format meets requirements."""

    def test_format_compact_under_1000_chars(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_compact produces output under 1000 characters."""
        result = formatter.format_compact(sample_simulation_output)

        assert len(result) < 1000, f"Compact format is {len(result)} chars, should be < 1000"

    def test_format_compact_contains_essential_info(
        self, formatter: SimulationReportFormatter, sample_simulation_output: SimulationOutput
    ):
        """Test that format_compact contains essential information."""
        result = formatter.format_compact(sample_simulation_output)

        # Should include ticker
        assert "AAPL" in result
        # Should include some price information
        assert "$" in result


class TestHorizonLabelFormatting:
    """Test horizon label formatting helper method."""

    def test_format_horizon_label_30_days(self, formatter: SimulationReportFormatter):
        """Test that 30 days is formatted as '30 days'."""
        label = formatter._format_horizon_label(30)
        assert label == "30 days"

    def test_format_horizon_label_90_days(self, formatter: SimulationReportFormatter):
        """Test that 90 days is formatted as '90 days'."""
        label = formatter._format_horizon_label(90)
        assert label == "90 days"

    def test_format_horizon_label_252_days_as_1_year(self, formatter: SimulationReportFormatter):
        """Test that 252 days is formatted as '1 year'."""
        label = formatter._format_horizon_label(252)
        assert label == "1 year"


class TestImpactLabelFormatting:
    """Test impact label formatting helper method."""

    def test_impact_label_high(self, formatter: SimulationReportFormatter):
        """Test that high impact score returns HIGH."""
        label = formatter._impact_label(80.0)
        assert label == "HIGH"

    def test_impact_label_medium(self, formatter: SimulationReportFormatter):
        """Test that medium impact score returns MEDIUM."""
        label = formatter._impact_label(50.0)
        assert label == "MEDIUM"

    def test_impact_label_low(self, formatter: SimulationReportFormatter):
        """Test that low impact score returns LOW."""
        label = formatter._impact_label(25.0)
        assert label == "LOW"


class TestFormatterInit:
    """Test formatter initialization."""

    def test_init_default_disclaimer(self):
        """Test that default disclaimer is set."""
        formatter = SimulationReportFormatter()
        assert formatter._disclaimer is not None

    def test_init_custom_disclaimer(self):
        """Test that custom disclaimer is used."""
        custom = "My custom disclaimer"
        formatter = SimulationReportFormatter(disclaimer=custom)
        assert formatter._disclaimer == custom
