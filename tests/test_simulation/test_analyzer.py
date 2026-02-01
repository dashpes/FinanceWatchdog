"""Tests for MonteCarloAnalyzer - the main orchestrator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from investment_monitor.simulation.analyzer import MonteCarloAnalyzer
from investment_monitor.simulation.crisis_loader import CrisisScenario
from investment_monitor.simulation.models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationConfig,
    SimulationOutput,
)


class TestCalculateHistoricalParameters:
    """Tests for calculate_historical_parameters method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance with fixed seed."""
        return MonteCarloAnalyzer(seed=42)

    def test_calculate_historical_parameters(self, analyzer):
        """Test calculation of annualized drift and volatility from price history."""
        # Create synthetic price data: 252 days of prices with known characteristics
        rng = np.random.default_rng(seed=42)

        # Start at 100 and simulate with known drift and volatility
        daily_drift = 0.08 / 252  # ~8% annual
        daily_vol = 0.20 / np.sqrt(252)  # ~20% annual

        prices = [100.0]
        for _ in range(251):  # 252 total prices
            daily_return = daily_drift + daily_vol * rng.standard_normal()
            new_price = prices[-1] * np.exp(daily_return)
            prices.append(new_price)

        prices = np.array(prices)

        mu, sigma = analyzer.calculate_historical_parameters(prices)

        # Check that we get reasonable annualized values
        # Drift should be in a reasonable range (allowing for randomness)
        assert -0.5 < mu < 0.5, f"Drift {mu} outside reasonable range"
        # Volatility should be positive and reasonable
        assert 0.05 < sigma < 0.6, f"Volatility {sigma} outside reasonable range"

    def test_calculate_historical_parameters_returns_float_tuple(self, analyzer):
        """Test that calculate_historical_parameters returns tuple of floats."""
        prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0])

        mu, sigma = analyzer.calculate_historical_parameters(prices)

        assert isinstance(mu, float), f"Expected float, got {type(mu)}"
        assert isinstance(sigma, float), f"Expected float, got {type(sigma)}"


class TestRunBaseCaseSimulation:
    """Tests for run_base_case_simulation method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance with fixed seed."""
        return MonteCarloAnalyzer(seed=42)

    def test_run_base_case_simulation_returns_dict_with_required_keys(self, analyzer):
        """Test that run_base_case_simulation returns dict with all required keys."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 30
        n_paths = 1000

        result = analyzer.run_base_case_simulation(S0, mu, sigma, days, n_paths)

        # Check all required keys are present
        required_keys = [
            "mean", "median", "std", "percentiles",
            "ci_80", "ci_95", "var_95", "cvar_95", "skewness"
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check types
        assert isinstance(result["mean"], float)
        assert isinstance(result["median"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["percentiles"], dict)
        assert isinstance(result["ci_80"], tuple)
        assert isinstance(result["ci_95"], tuple)
        assert isinstance(result["var_95"], float)
        assert isinstance(result["cvar_95"], float)
        assert isinstance(result["skewness"], float)

    def test_run_base_case_simulation_percentiles(self, analyzer):
        """Test that percentiles dictionary has expected keys."""
        result = analyzer.run_base_case_simulation(100.0, 0.08, 0.2, 30, 1000)

        # Should have standard percentiles
        percentiles = result["percentiles"]
        assert 5 in percentiles
        assert 25 in percentiles
        assert 50 in percentiles
        assert 75 in percentiles
        assert 95 in percentiles


class TestRunStressScenario:
    """Tests for run_stress_scenario method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance with fixed seed."""
        return MonteCarloAnalyzer(seed=42)

    def test_run_stress_scenario_returns_dict_with_name(self, analyzer):
        """Test that run_stress_scenario returns dict with name and prob_loss_20pct."""
        S0 = 100.0
        scenario = CrisisScenario.CRISIS_2008
        days = 30
        n_paths = 1000
        beta = 1.0

        result = analyzer.run_stress_scenario(S0, scenario, days, n_paths, beta)

        # Check required keys
        assert "name" in result, "Missing 'name' key"
        assert "prob_loss_20pct" in result, "Missing 'prob_loss_20pct' key"
        assert "mean" in result, "Missing 'mean' key"

        # Name should be set
        assert isinstance(result["name"], str)
        assert len(result["name"]) > 0

        # prob_loss_20pct should be a probability (0 to 1)
        assert 0.0 <= result["prob_loss_20pct"] <= 1.0, (
            f"prob_loss_20pct {result['prob_loss_20pct']} not in [0, 1]"
        )

    def test_run_stress_scenario_with_beta_adjustment(self, analyzer):
        """Test that beta adjustment affects results."""
        S0 = 100.0
        scenario = CrisisScenario.COVID_CRASH
        days = 30
        n_paths = 5000

        result_beta_1 = analyzer.run_stress_scenario(S0, scenario, days, n_paths, beta=1.0)
        result_beta_15 = analyzer.run_stress_scenario(S0, scenario, days, n_paths, beta=1.5)

        # Higher beta should lead to more extreme results (lower mean in crisis)
        # With 1.5x beta, expected loss is higher
        assert result_beta_15["mean"] != result_beta_1["mean"], (
            "Beta adjustment should affect mean"
        )


class TestBuildHorizonResult:
    """Tests for build_horizon_result method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance."""
        return MonteCarloAnalyzer(seed=42)

    def test_build_horizon_result(self, analyzer):
        """Test that build_horizon_result constructs HorizonResult correctly."""
        days = 30

        base_stats = {
            "mean": 105.0,
            "median": 104.5,
            "std": 15.0,
            "skewness": 0.3,
            "percentiles": {5: 85.0, 25: 95.0, 50: 104.5, 75: 115.0, 95: 130.0},
            "ci_80": (90.0, 120.0),
            "ci_95": (80.0, 130.0),
            "var_95": -0.15,
            "cvar_95": -0.20,
        }

        scenario_results = [
            {
                "name": "2008 Financial Crisis",
                "mean": 90.0,
                "median": 88.0,
                "std": 20.0,
                "ci_80": (75.0, 105.0),
                "ci_95": (65.0, 115.0),
                "var_95": -0.25,
                "cvar_95": -0.30,
                "prob_loss_20pct": 0.45,
            }
        ]

        result = analyzer.build_horizon_result(days, base_stats, scenario_results)

        # Verify it's a HorizonResult
        assert isinstance(result, HorizonResult)

        # Verify days
        assert result.days == 30

        # Verify base case stats
        assert result.base_mean == 105.0
        assert result.base_median == 104.5
        assert result.base_std == 15.0
        assert result.base_skewness == 0.3
        assert result.base_ci_80 == (90.0, 120.0)
        assert result.base_ci_95 == (80.0, 130.0)
        assert result.base_var_95 == -0.15
        assert result.base_cvar_95 == -0.20

        # Verify scenarios
        assert "2008 Financial Crisis" in result.scenarios
        scenario = result.scenarios["2008 Financial Crisis"]
        assert isinstance(scenario, ScenarioResult)
        assert scenario.prob_loss_20pct == 0.45


class TestAnalyze:
    """Tests for analyze method - main entry point."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance with fixed seed."""
        return MonteCarloAnalyzer(seed=42)

    @pytest.fixture
    def mock_price_history(self):
        """Create mock price history."""
        # Generate 500 days of synthetic price data
        rng = np.random.default_rng(seed=42)
        prices = [100.0]
        for _ in range(499):
            daily_return = 0.08/252 + (0.20/np.sqrt(252)) * rng.standard_normal()
            prices.append(prices[-1] * np.exp(daily_return))
        return np.array(prices)

    def test_analyze_returns_simulation_output(self, analyzer, mock_price_history):
        """Test that analyze returns SimulationOutput with mocked dependencies."""
        ticker = "AAPL"
        entry_price = 150.0
        composite_score = 85.0

        # Mock _fetch_price_history and _calculate_beta
        with patch.object(
            analyzer, "_fetch_price_history", return_value=mock_price_history
        ), patch.object(
            analyzer, "_calculate_beta", return_value=1.1
        ):
            result = analyzer.analyze(ticker, entry_price, composite_score, force=False)

        # Verify the result is a SimulationOutput
        assert isinstance(result, SimulationOutput)

        # Verify basic fields
        assert result.ticker == ticker
        assert result.entry_price == entry_price
        assert result.composite_score == composite_score

        # Verify results dict has horizons
        assert isinstance(result.results, dict)
        # Should have at least one horizon
        assert len(result.results) > 0

        # Verify each result is a HorizonResult
        for days, horizon_result in result.results.items():
            assert isinstance(horizon_result, HorizonResult)
            assert horizon_result.days == days

        # Verify sensitivity is present
        assert isinstance(result.sensitivity, SensitivityResult)

    def test_analyze_with_force_flag(self, analyzer, mock_price_history):
        """Test that analyze works with force=True even below threshold."""
        ticker = "TSLA"
        entry_price = 200.0
        composite_score = 50.0  # Below default threshold of 80

        with patch.object(
            analyzer, "_fetch_price_history", return_value=mock_price_history
        ), patch.object(
            analyzer, "_calculate_beta", return_value=1.5
        ):
            # Should run because force=True
            result = analyzer.analyze(ticker, entry_price, composite_score, force=True)

        assert isinstance(result, SimulationOutput)
        assert result.ticker == ticker


class TestShouldRunSimulation:
    """Tests for should_run_simulation gating method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance."""
        return MonteCarloAnalyzer()

    def test_should_run_simulation_above_threshold(self, analyzer):
        """Test that should_run_simulation returns True when score >= threshold."""
        # Default threshold is 80
        assert analyzer.should_run_simulation(85.0) is True
        assert analyzer.should_run_simulation(80.0) is True
        assert analyzer.should_run_simulation(100.0) is True

    def test_should_run_simulation_below_threshold(self, analyzer):
        """Test that should_run_simulation returns False when score < threshold."""
        # Default threshold is 80
        assert analyzer.should_run_simulation(79.9) is False
        assert analyzer.should_run_simulation(50.0) is False
        assert analyzer.should_run_simulation(0.0) is False

    def test_should_run_simulation_with_override(self, analyzer):
        """Test that should_run_simulation returns True when force=True."""
        # Even with low score, force=True should return True
        assert analyzer.should_run_simulation(50.0, force=True) is True
        assert analyzer.should_run_simulation(0.0, force=True) is True
        assert analyzer.should_run_simulation(85.0, force=True) is True

    def test_should_run_simulation_with_custom_config(self):
        """Test should_run_simulation with custom threshold."""
        config = SimulationConfig(score_threshold=90.0)
        analyzer = MonteCarloAnalyzer(config=config)

        assert analyzer.should_run_simulation(95.0) is True
        assert analyzer.should_run_simulation(89.0) is False
        assert analyzer.should_run_simulation(89.0, force=True) is True


class TestFetchPriceHistory:
    """Tests for _fetch_price_history method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance."""
        return MonteCarloAnalyzer(seed=42)

    @patch("investment_monitor.simulation.analyzer.yf.download")
    def test_fetch_price_history_returns_numpy_array(self, mock_download, analyzer):
        """Test that _fetch_price_history returns numpy array of prices."""
        import pandas as pd

        # Create mock DataFrame with Close prices
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        mock_df = pd.DataFrame({
            "Close": np.linspace(100, 110, 100)
        }, index=dates)
        mock_download.return_value = mock_df

        prices = analyzer._fetch_price_history("AAPL", days=100)

        assert isinstance(prices, np.ndarray)
        assert len(prices) > 0
        mock_download.assert_called_once()


class TestCalculateBeta:
    """Tests for _calculate_beta method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance."""
        return MonteCarloAnalyzer(seed=42)

    @patch("investment_monitor.simulation.analyzer.yf.download")
    def test_calculate_beta_returns_float(self, mock_download, analyzer):
        """Test that _calculate_beta returns a float."""
        import pandas as pd

        # Create correlated mock data for stock and market
        rng = np.random.default_rng(seed=42)
        market_returns = rng.normal(0.0004, 0.01, 252)
        # Stock moves with market (beta ~= 1.2)
        stock_returns = 1.2 * market_returns + rng.normal(0, 0.005, 252)

        # Convert to prices
        market_prices = 100 * np.exp(np.cumsum(market_returns))
        stock_prices = 100 * np.exp(np.cumsum(stock_returns))

        dates = pd.date_range(start="2024-01-01", periods=252, freq="D")

        def mock_download_side_effect(tickers, **kwargs):
            mock_df = pd.DataFrame({
                ("AAPL", "Close"): stock_prices,
                ("SPY", "Close"): market_prices,
            }, index=dates)
            mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)
            return mock_df

        mock_download.side_effect = mock_download_side_effect

        beta = analyzer._calculate_beta("AAPL", lookback_days=252)

        assert isinstance(beta, float)
        # Beta should be reasonable
        assert 0.0 < beta < 3.0, f"Beta {beta} outside reasonable range"


class TestDetermineLookbackDays:
    """Tests for _determine_lookback_days method."""

    @pytest.fixture
    def analyzer(self):
        """Create a MonteCarloAnalyzer instance."""
        return MonteCarloAnalyzer()

    def test_determine_lookback_days_with_sufficient_data(self, analyzer):
        """Test lookback determination with plenty of data."""
        # Default max_lookback_days is 1260
        available_days = 2000

        lookback = analyzer._determine_lookback_days(available_days)

        # Should return max_lookback from config
        assert lookback <= analyzer.config.max_lookback_days
        assert lookback >= analyzer.config.min_lookback_days

    def test_determine_lookback_days_with_limited_data(self, analyzer):
        """Test lookback determination with limited data."""
        # Less than min_lookback_days
        available_days = 200

        lookback = analyzer._determine_lookback_days(available_days)

        # Should return available days
        assert lookback == available_days

    def test_determine_lookback_days_respects_config(self):
        """Test that lookback respects configuration."""
        config = SimulationConfig(min_lookback_days=100, max_lookback_days=500)
        analyzer = MonteCarloAnalyzer(config=config)

        # With plenty of data, should use max
        lookback = analyzer._determine_lookback_days(1000)
        assert lookback == 500


class TestMonteCarloAnalyzerInit:
    """Tests for MonteCarloAnalyzer initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        analyzer = MonteCarloAnalyzer()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, SimulationConfig)
        assert analyzer.config.score_threshold == 80.0

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = SimulationConfig(score_threshold=90.0, min_paths=2000)
        analyzer = MonteCarloAnalyzer(config=config)

        assert analyzer.config.score_threshold == 90.0
        assert analyzer.config.min_paths == 2000

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        analyzer1 = MonteCarloAnalyzer(seed=42)
        analyzer2 = MonteCarloAnalyzer(seed=42)

        # Both should produce same results
        result1 = analyzer1.run_base_case_simulation(100.0, 0.08, 0.2, 30, 1000)
        result2 = analyzer2.run_base_case_simulation(100.0, 0.08, 0.2, 30, 1000)

        assert result1["mean"] == result2["mean"]

    def test_init_creates_engine_and_loader(self):
        """Test that init creates SimulationEngine, CrisisDataLoader, and SensitivityAnalyzer."""
        analyzer = MonteCarloAnalyzer()

        # These should exist as attributes (private or public)
        assert hasattr(analyzer, "_engine") or hasattr(analyzer, "engine")
        assert hasattr(analyzer, "_crisis_loader") or hasattr(analyzer, "crisis_loader")
        assert hasattr(analyzer, "_sensitivity_analyzer") or hasattr(analyzer, "sensitivity_analyzer")
