"""Tests for crisis data loader module."""

from pathlib import Path

import numpy as np
import pytest

from investment_monitor.simulation import CrisisDataLoader, CrisisScenario


class TestCrisisScenarioEnum:
    """Tests for CrisisScenario enumeration."""

    def test_all_scenarios_defined(self):
        """Verify that all 8 required scenarios are defined in the enum."""
        scenarios = list(CrisisScenario)

        # Should have exactly 8 scenarios
        assert len(scenarios) == 8, f"Expected 8 scenarios, got {len(scenarios)}"

        # Verify each expected scenario exists
        expected_scenarios = [
            "CRISIS_2008",
            "DOTCOM_CRASH",
            "COVID_CRASH",
            "STAGFLATION_1970S",
            "BLACK_MONDAY_1987",
            "RISING_RATES_2022",
            "REGIME_DEMOCRAT",
            "REGIME_REPUBLICAN",
        ]

        for name in expected_scenarios:
            assert hasattr(CrisisScenario, name), f"Missing scenario: {name}"

    def test_scenario_values(self):
        """Verify scenario values match expected CSV filenames."""
        expected_values = {
            CrisisScenario.CRISIS_2008: "sp500_2008_crisis",
            CrisisScenario.DOTCOM_CRASH: "sp500_dotcom_crash",
            CrisisScenario.COVID_CRASH: "sp500_covid_crash",
            CrisisScenario.STAGFLATION_1970S: "sp500_stagflation_1970s",
            CrisisScenario.BLACK_MONDAY_1987: "sp500_black_monday_1987",
            CrisisScenario.RISING_RATES_2022: "sp500_rising_rates_2022",
            CrisisScenario.REGIME_DEMOCRAT: "regime_democrat_returns",
            CrisisScenario.REGIME_REPUBLICAN: "regime_republican_returns",
        }

        for scenario, expected_value in expected_values.items():
            assert scenario.value == expected_value, (
                f"Scenario {scenario.name} has value '{scenario.value}', "
                f"expected '{expected_value}'"
            )


class TestCrisisDataLoader:
    """Tests for CrisisDataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a CrisisDataLoader instance."""
        return CrisisDataLoader()

    def test_init_default_data_dir(self, loader):
        """Test initialization with default data directory."""
        assert loader.data_dir.exists()
        assert loader.data_dir.name == "crisis_data"

    def test_init_custom_data_dir(self, tmp_path):
        """Test initialization with custom data directory."""
        custom_dir = tmp_path / "custom_crisis_data"
        custom_dir.mkdir()

        loader = CrisisDataLoader(data_dir=custom_dir)
        assert loader.data_dir == custom_dir

    def test_init_invalid_data_dir(self, tmp_path):
        """Test initialization with non-existent directory raises error."""
        invalid_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Crisis data directory not found"):
            CrisisDataLoader(data_dir=invalid_dir)

    def test_load_crisis_returns(self, loader):
        """Test loading crisis returns returns numpy array with float64 dtype."""
        returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)

        assert isinstance(returns, np.ndarray), "Returns should be a numpy array"
        assert returns.dtype == np.float64, "Returns should have float64 dtype"
        assert len(returns) > 0, "Returns array should not be empty"

    def test_all_crisis_data_loads(self, loader):
        """Test that all 8 scenarios load with more than 5 data points."""
        for scenario in CrisisScenario:
            returns = loader.load_crisis_returns(scenario)

            assert isinstance(returns, np.ndarray), (
                f"Scenario {scenario.name}: Returns should be a numpy array"
            )
            assert len(returns) > 5, (
                f"Scenario {scenario.name}: Expected >5 data points, got {len(returns)}"
            )
            assert returns.dtype == np.float64, (
                f"Scenario {scenario.name}: Returns should have float64 dtype"
            )

    def test_returns_caching(self, loader):
        """Test that returns are cached after first load."""
        # First load
        returns1 = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)

        # Verify it's in cache
        assert CrisisScenario.CRISIS_2008 in loader._cache

        # Second load should return same object (cached)
        returns2 = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)

        assert returns1 is returns2, "Cached returns should be the same object"

    def test_clear_cache(self, loader):
        """Test cache clearing."""
        # Load a scenario to populate cache
        loader.load_crisis_returns(CrisisScenario.CRISIS_2008)
        assert len(loader._cache) > 0

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_preload_all(self, loader):
        """Test preloading all scenarios."""
        assert len(loader._cache) == 0

        loader.preload_all()

        assert len(loader._cache) == len(CrisisScenario)

    def test_apply_beta_adjustment(self, loader):
        """Test beta adjustment correctly scales returns."""
        base_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        beta = 1.5

        adjusted = loader.apply_beta_adjustment(base_returns, beta)

        expected = beta * base_returns
        np.testing.assert_array_almost_equal(adjusted, expected)

    def test_apply_beta_adjustment_with_real_data(self, loader):
        """Test beta adjustment with actual crisis data."""
        market_returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)
        beta = 1.3

        adjusted = loader.apply_beta_adjustment(market_returns, beta)

        # Check array properties
        assert adjusted.shape == market_returns.shape
        assert adjusted.dtype == np.float64

        # Check values are scaled correctly
        np.testing.assert_array_almost_equal(adjusted, beta * market_returns)

    def test_apply_beta_adjustment_low_beta(self, loader):
        """Test beta adjustment with low beta (less volatile stock)."""
        base_returns = np.array([0.10, -0.10])
        beta = 0.5

        adjusted = loader.apply_beta_adjustment(base_returns, beta)

        expected = np.array([0.05, -0.05])
        np.testing.assert_array_almost_equal(adjusted, expected)

    def test_get_scenario_metadata(self, loader):
        """Test getting scenario metadata returns correct structure."""
        for scenario in CrisisScenario:
            metadata = loader.get_scenario_metadata(scenario)

            # Verify required keys exist
            assert "name" in metadata, f"Missing 'name' for {scenario.name}"
            assert "start_date" in metadata, f"Missing 'start_date' for {scenario.name}"
            assert "end_date" in metadata, f"Missing 'end_date' for {scenario.name}"
            assert "description" in metadata, f"Missing 'description' for {scenario.name}"

            # Verify types
            assert isinstance(metadata["name"], str)
            assert isinstance(metadata["start_date"], str)
            assert isinstance(metadata["end_date"], str)
            assert isinstance(metadata["description"], str)

            # Verify date format (YYYY-MM-DD)
            for date_key in ["start_date", "end_date"]:
                date_value = metadata[date_key]
                parts = date_value.split("-")
                assert len(parts) == 3, f"Invalid date format for {scenario.name}.{date_key}"
                assert len(parts[0]) == 4, f"Invalid year in {scenario.name}.{date_key}"
                assert len(parts[1]) == 2, f"Invalid month in {scenario.name}.{date_key}"
                assert len(parts[2]) == 2, f"Invalid day in {scenario.name}.{date_key}"

    def test_get_scenario_metadata_returns_copy(self, loader):
        """Test that metadata returns a copy, not the original dict."""
        metadata1 = loader.get_scenario_metadata(CrisisScenario.CRISIS_2008)
        metadata2 = loader.get_scenario_metadata(CrisisScenario.CRISIS_2008)

        # Modify metadata1
        metadata1["name"] = "Modified"

        # metadata2 should be unchanged
        assert metadata2["name"] != "Modified"

    def test_get_all_scenarios(self):
        """Test getting list of all scenarios."""
        scenarios = CrisisDataLoader.get_all_scenarios()

        assert isinstance(scenarios, list)
        assert len(scenarios) == 8

        for scenario in scenarios:
            assert isinstance(scenario, CrisisScenario)

    def test_crisis_returns_are_log_returns(self, loader):
        """Test that crisis returns are log returns in reasonable range.

        Log returns should typically be in the range of roughly -0.25 to +0.15
        for most days, with extreme days potentially reaching -0.25 or lower.
        The vast majority should be within +/- 0.10 (10%).
        """
        for scenario in CrisisScenario:
            returns = loader.load_crisis_returns(scenario)

            # Check that returns are not percentage values (would be too large)
            # Log returns should typically be < 0.20 in absolute value for most days
            median_abs_return = np.median(np.abs(returns))
            assert median_abs_return < 0.20, (
                f"Scenario {scenario.name}: Median absolute return {median_abs_return:.4f} "
                "seems too large for log returns. Values may be percentages."
            )

            # Check there are no NaN or infinite values
            assert not np.any(np.isnan(returns)), (
                f"Scenario {scenario.name}: Contains NaN values"
            )
            assert not np.any(np.isinf(returns)), (
                f"Scenario {scenario.name}: Contains infinite values"
            )

            # Log returns should sum approximately to ln(P_end/P_start)
            # Just check they're finite and reasonable
            cumulative = np.sum(returns)
            assert np.isfinite(cumulative), (
                f"Scenario {scenario.name}: Cumulative return is not finite"
            )

            # Most extreme daily log return in history was about -0.23 (Black Monday)
            # Allow some margin for calculation differences
            max_abs_return = np.max(np.abs(returns))
            assert max_abs_return < 0.30, (
                f"Scenario {scenario.name}: Max absolute return {max_abs_return:.4f} "
                "is unreasonably large for a daily log return"
            )

    def test_crisis_2008_has_severe_returns(self, loader):
        """Test that 2008 crisis data contains some severe negative returns."""
        returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)

        min_return = np.min(returns)
        assert min_return < -0.05, (
            f"2008 crisis should have severe drops. Min return: {min_return:.4f}"
        )

    def test_black_monday_has_extreme_return(self, loader):
        """Test that Black Monday data contains the extreme -22.6% day."""
        returns = loader.load_crisis_returns(CrisisScenario.BLACK_MONDAY_1987)

        # Black Monday had a ~22.6% drop, log return of about -0.256
        min_return = np.min(returns)
        assert min_return < -0.20, (
            f"Black Monday should have the historic crash. Min return: {min_return:.4f}"
        )

    def test_covid_crash_is_short_period(self, loader):
        """Test that COVID crash data is a short, intense period."""
        returns = loader.load_crisis_returns(CrisisScenario.COVID_CRASH)

        # COVID crash was Feb-Mar 2020, about 39-45 trading days
        assert len(returns) < 60, (
            f"COVID crash should be ~40 trading days. Got {len(returns)}"
        )
        assert len(returns) > 20, (
            f"COVID crash should have at least 20 trading days. Got {len(returns)}"
        )

    def test_regime_scenarios_are_large(self, loader):
        """Test that regime scenarios have large sample sizes."""
        democrat_returns = loader.load_crisis_returns(CrisisScenario.REGIME_DEMOCRAT)
        republican_returns = loader.load_crisis_returns(CrisisScenario.REGIME_REPUBLICAN)

        # Each regime covers multiple presidential terms, should have thousands of days
        assert len(democrat_returns) > 3000, (
            f"Democrat regime should have >3000 data points. Got {len(democrat_returns)}"
        )
        assert len(republican_returns) > 3000, (
            f"Republican regime should have >3000 data points. Got {len(republican_returns)}"
        )

    def test_get_combined_returns(self, loader):
        """Test combining returns from multiple scenarios."""
        scenarios = [CrisisScenario.CRISIS_2008, CrisisScenario.COVID_CRASH]
        combined = loader.get_combined_returns(scenarios)

        # Should be sum of individual lengths
        expected_len = sum(
            len(loader.load_crisis_returns(s)) for s in scenarios
        )
        assert len(combined) == expected_len

        # Should still be float64
        assert combined.dtype == np.float64


class TestCrisisDataFiles:
    """Tests to verify the actual CSV data files exist and are valid."""

    @pytest.fixture
    def data_dir(self):
        """Get the crisis data directory path."""
        return Path(__file__).parent.parent.parent / "src/investment_monitor/simulation/crisis_data"

    def test_data_directory_exists(self, data_dir):
        """Test that the crisis data directory exists."""
        assert data_dir.exists(), f"Data directory not found: {data_dir}"
        assert data_dir.is_dir(), f"Path is not a directory: {data_dir}"

    def test_all_csv_files_exist(self, data_dir):
        """Test that all required CSV files exist."""
        expected_files = [
            "sp500_2008_crisis.csv",
            "sp500_dotcom_crash.csv",
            "sp500_covid_crash.csv",
            "sp500_stagflation_1970s.csv",
            "sp500_black_monday_1987.csv",
            "sp500_rising_rates_2022.csv",
            "regime_democrat_returns.csv",
            "regime_republican_returns.csv",
        ]

        for filename in expected_files:
            filepath = data_dir / filename
            assert filepath.exists(), f"Missing CSV file: {filename}"
            assert filepath.stat().st_size > 0, f"Empty CSV file: {filename}"

    def test_readme_exists(self, data_dir):
        """Test that README.md exists in crisis_data directory."""
        readme = data_dir / "README.md"
        assert readme.exists(), "README.md not found in crisis_data directory"
