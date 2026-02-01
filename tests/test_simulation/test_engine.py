"""Tests for Monte Carlo simulation engine."""

import numpy as np
import pytest

from investment_monitor.simulation.engine import SimulationEngine


class TestSimulateGBM:
    """Tests for Geometric Brownian Motion simulation."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed for reproducibility."""
        return SimulationEngine(seed=42)

    def test_simulate_gbm_returns_correct_shape(self, engine):
        """Test that simulate_gbm returns array with correct shape (n_paths,)."""
        S0 = 100.0
        mu = 0.08  # 8% annual drift
        sigma = 0.2  # 20% annual volatility
        days = 30
        n_paths = 1000

        terminal_prices = engine.simulate_gbm(S0, mu, sigma, days, n_paths)

        assert isinstance(terminal_prices, np.ndarray)
        assert terminal_prices.shape == (n_paths,)
        assert terminal_prices.dtype == np.float64

    def test_simulate_gbm_positive_prices(self, engine):
        """Test that all simulated prices are positive (GBM property)."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 252  # Full year
        n_paths = 10000

        terminal_prices = engine.simulate_gbm(S0, mu, sigma, days, n_paths)

        assert np.all(terminal_prices > 0), "GBM should always produce positive prices"

    def test_simulate_gbm_mean_near_expected(self, engine):
        """Test that mean terminal price is within 5% of E[S_T] = S0 * exp(mu * T)."""
        S0 = 100.0
        mu = 0.08  # 8% annual drift
        sigma = 0.2  # 20% annual volatility
        days = 252  # 1 year
        n_paths = 50000  # Large sample for accuracy

        terminal_prices = engine.simulate_gbm(S0, mu, sigma, days, n_paths)

        T = days / 252  # Time in years
        expected_mean = S0 * np.exp(mu * T)
        actual_mean = np.mean(terminal_prices)

        # Within 5% of expected
        relative_error = abs(actual_mean - expected_mean) / expected_mean
        assert relative_error < 0.05, (
            f"Mean {actual_mean:.2f} not within 5% of expected {expected_mean:.2f} "
            f"(relative error: {relative_error:.2%})"
        )

    def test_simulate_gbm_reproducible_with_seed(self):
        """Test that simulations are reproducible with the same seed."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 30
        n_paths = 100

        engine1 = SimulationEngine(seed=12345)
        engine2 = SimulationEngine(seed=12345)

        prices1 = engine1.simulate_gbm(S0, mu, sigma, days, n_paths)
        prices2 = engine2.simulate_gbm(S0, mu, sigma, days, n_paths)

        np.testing.assert_array_equal(prices1, prices2)

    def test_simulate_gbm_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 30
        n_paths = 100

        engine1 = SimulationEngine(seed=111)
        engine2 = SimulationEngine(seed=222)

        prices1 = engine1.simulate_gbm(S0, mu, sigma, days, n_paths)
        prices2 = engine2.simulate_gbm(S0, mu, sigma, days, n_paths)

        # Should not be equal
        assert not np.array_equal(prices1, prices2)


class TestSimulateBootstrap:
    """Tests for block bootstrap simulation."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def crisis_returns(self):
        """Create synthetic crisis returns for testing."""
        # Simulate realistic crisis returns: mean -0.002, std 0.03
        rng = np.random.default_rng(seed=99)
        return rng.normal(-0.002, 0.03, size=100)

    def test_simulate_bootstrap_returns_correct_shape(self, engine, crisis_returns):
        """Test that simulate_bootstrap returns array with correct shape."""
        S0 = 100.0
        days = 30
        n_paths = 1000

        terminal_prices = engine.simulate_bootstrap(S0, crisis_returns, days, n_paths)

        assert isinstance(terminal_prices, np.ndarray)
        assert terminal_prices.shape == (n_paths,)
        assert terminal_prices.dtype == np.float64

    def test_simulate_bootstrap_preserves_return_distribution(self, engine, crisis_returns):
        """Test that bootstrap preserves the overall return distribution characteristics."""
        S0 = 100.0
        days = 50
        n_paths = 10000
        block_size = 5

        terminal_prices = engine.simulate_bootstrap(
            S0, crisis_returns, days, n_paths, block_size=block_size
        )

        # Calculate realized log returns
        log_returns = np.log(terminal_prices / S0) / days

        # The mean daily return should be roughly similar to crisis_returns mean
        # Allow for some variation due to sampling
        crisis_mean = np.mean(crisis_returns)
        realized_mean = np.mean(log_returns)

        # Should be within 50% of original mean (bootstrap has variance)
        if abs(crisis_mean) > 0.0001:  # Avoid division by zero
            relative_diff = abs(realized_mean - crisis_mean) / abs(crisis_mean)
            assert relative_diff < 1.0, (
                f"Mean daily return {realized_mean:.6f} differs too much from "
                f"crisis mean {crisis_mean:.6f}"
            )

    def test_simulate_bootstrap_handles_short_crisis_period(self, engine):
        """Test that bootstrap handles short crisis periods gracefully."""
        S0 = 100.0
        # Very short crisis period (fewer than block_size)
        short_crisis = np.array([-0.05, -0.03, -0.02])
        days = 30
        n_paths = 100
        block_size = 5

        # Should not raise an error
        terminal_prices = engine.simulate_bootstrap(
            S0, short_crisis, days, n_paths, block_size=block_size
        )

        assert terminal_prices.shape == (n_paths,)
        assert np.all(terminal_prices > 0)

    def test_simulate_bootstrap_positive_prices(self, engine, crisis_returns):
        """Test that bootstrap simulation produces positive prices."""
        S0 = 100.0
        days = 100
        n_paths = 5000

        terminal_prices = engine.simulate_bootstrap(S0, crisis_returns, days, n_paths)

        assert np.all(terminal_prices > 0), "Bootstrap should produce positive prices"


class TestCalculateVaR:
    """Tests for Value at Risk calculation."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance."""
        return SimulationEngine(seed=42)

    def test_calculate_var(self, engine):
        """Test VaR calculation returns correct percentile."""
        entry_price = 100.0
        # Create terminal prices with known distribution
        # 95% VaR should be the 5th percentile return
        rng = np.random.default_rng(seed=42)
        terminal_prices = entry_price * np.exp(rng.normal(-0.05, 0.2, size=10000))

        var_95 = engine.calculate_var(terminal_prices, entry_price, confidence=0.95)

        # VaR should be negative (indicating a loss)
        assert var_95 < 0, "VaR should be negative for this distribution"

        # VaR should be a return (fraction), not a dollar amount
        assert -1 < var_95 < 1, "VaR should be a return fraction"

        # Manually calculate 5th percentile return
        returns = terminal_prices / entry_price - 1
        expected_var = np.percentile(returns, 5)

        np.testing.assert_almost_equal(var_95, expected_var, decimal=4)

    def test_calculate_var_different_confidence(self, engine):
        """Test VaR at different confidence levels."""
        entry_price = 100.0
        rng = np.random.default_rng(seed=42)
        terminal_prices = entry_price * np.exp(rng.normal(-0.02, 0.15, size=10000))

        var_90 = engine.calculate_var(terminal_prices, entry_price, confidence=0.90)
        var_95 = engine.calculate_var(terminal_prices, entry_price, confidence=0.95)
        var_99 = engine.calculate_var(terminal_prices, entry_price, confidence=0.99)

        # Higher confidence should give more extreme (lower) VaR
        assert var_99 < var_95 < var_90


class TestCalculateCVaR:
    """Tests for Conditional Value at Risk (Expected Shortfall) calculation."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance."""
        return SimulationEngine(seed=42)

    def test_calculate_cvar(self, engine):
        """Test that CVaR is less than or equal to VaR (CVaR is more extreme)."""
        entry_price = 100.0
        rng = np.random.default_rng(seed=42)
        terminal_prices = entry_price * np.exp(rng.normal(-0.05, 0.2, size=10000))

        var_95 = engine.calculate_var(terminal_prices, entry_price, confidence=0.95)
        cvar_95 = engine.calculate_cvar(terminal_prices, entry_price, confidence=0.95)

        # CVaR should be less than (more negative than) VaR
        assert cvar_95 <= var_95, (
            f"CVaR ({cvar_95:.4f}) should be <= VaR ({var_95:.4f})"
        )

        # Both should be returns (fractions)
        assert -1 < cvar_95 < 1, "CVaR should be a return fraction"

    def test_calculate_cvar_is_mean_of_tail(self, engine):
        """Test that CVaR equals the mean of returns below VaR."""
        entry_price = 100.0
        rng = np.random.default_rng(seed=42)
        terminal_prices = entry_price * np.exp(rng.normal(-0.03, 0.15, size=10000))

        returns = terminal_prices / entry_price - 1
        var_threshold = np.percentile(returns, 5)
        expected_cvar = np.mean(returns[returns <= var_threshold])

        cvar_95 = engine.calculate_cvar(terminal_prices, entry_price, confidence=0.95)

        np.testing.assert_almost_equal(cvar_95, expected_cvar, decimal=4)


class TestDeterminePathCount:
    """Tests for adaptive path count determination."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance."""
        return SimulationEngine(seed=42)

    def test_determine_path_count_narrow_ci(self, engine):
        """Test that narrow CI returns minimum paths (1000)."""
        # Pilot results with tight confidence interval
        pilot_results = {
            "mean": 105.0,
            "std": 10.0,
            "ci_width": 0.10,  # 10% CI width, below threshold
            "n_paths": 500,
        }

        path_count = engine.determine_path_count(
            pilot_results,
            ci_width_threshold=0.15,
            min_paths=1000,
            max_paths=50000,
        )

        assert path_count == 1000, f"Expected 1000 for narrow CI, got {path_count}"

    def test_determine_path_count_wide_ci(self, engine):
        """Test that wide CI returns more than minimum paths."""
        # Pilot results with wide confidence interval
        pilot_results = {
            "mean": 105.0,
            "std": 30.0,
            "ci_width": 0.25,  # 25% CI width, above threshold
            "n_paths": 500,
        }

        path_count = engine.determine_path_count(
            pilot_results,
            ci_width_threshold=0.15,
            min_paths=1000,
            max_paths=50000,
        )

        assert path_count > 1000, f"Expected >1000 for wide CI, got {path_count}"

    def test_determine_path_count_respects_max(self, engine):
        """Test that path count doesn't exceed max_paths."""
        # Very wide CI
        pilot_results = {
            "mean": 100.0,
            "std": 100.0,
            "ci_width": 0.90,  # Very wide
            "n_paths": 500,
        }

        path_count = engine.determine_path_count(
            pilot_results,
            ci_width_threshold=0.15,
            min_paths=1000,
            max_paths=50000,
        )

        assert path_count <= 50000, f"Path count {path_count} exceeds max"

    def test_determine_path_count_at_threshold(self, engine):
        """Test behavior when CI width is exactly at threshold."""
        pilot_results = {
            "mean": 100.0,
            "std": 15.0,
            "ci_width": 0.15,  # Exactly at threshold
            "n_paths": 500,
        }

        path_count = engine.determine_path_count(
            pilot_results,
            ci_width_threshold=0.15,
            min_paths=1000,
            max_paths=50000,
        )

        # At threshold should return minimum
        assert path_count == 1000


class TestSimulationEngineInit:
    """Tests for SimulationEngine initialization."""

    def test_init_without_seed(self):
        """Test initialization without seed produces non-deterministic results."""
        engine1 = SimulationEngine()
        engine2 = SimulationEngine()

        # Generate some random numbers
        prices1 = engine1.simulate_gbm(100.0, 0.08, 0.2, 30, 100)
        prices2 = engine2.simulate_gbm(100.0, 0.08, 0.2, 30, 100)

        # Very unlikely to be equal without same seed
        # Note: There's a tiny probability they could be equal by chance
        assert not np.array_equal(prices1, prices2)

    def test_init_with_seed(self):
        """Test initialization with seed stores the RNG properly."""
        engine = SimulationEngine(seed=42)
        assert engine._rng is not None
