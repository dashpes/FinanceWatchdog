"""Tests for SensitivityAnalyzer."""

import pytest

from investment_monitor.simulation.engine import SimulationEngine
from investment_monitor.simulation.models import SensitivityResult
from investment_monitor.simulation.sensitivity import SensitivityAnalyzer


class TestAnalyzeVolatilitySensitivity:
    """Tests for volatility sensitivity analysis."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed for reproducibility."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def analyzer(self, engine):
        """Create a SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(engine)

    def test_analyze_volatility_sensitivity(self, analyzer):
        """Verify dict with multipliers as keys."""
        S0 = 100.0
        mu = 0.08  # 8% annual drift
        base_sigma = 0.2  # 20% annual volatility
        days = 30
        n_paths = 1000

        result = analyzer.analyze_volatility_sensitivity(
            S0=S0,
            mu=mu,
            base_sigma=base_sigma,
            days=days,
            n_paths=n_paths,
        )

        # Should return a dict
        assert isinstance(result, dict)

        # Default multipliers should be [0.5, 0.8, 1.0, 1.2, 1.5]
        expected_multipliers = {0.5, 0.8, 1.0, 1.2, 1.5}
        assert set(result.keys()) == expected_multipliers

        # All values should be positive floats (mean terminal prices)
        for multiplier, mean_price in result.items():
            assert isinstance(mean_price, float)
            assert mean_price > 0

    def test_analyze_volatility_sensitivity_custom_multipliers(self, analyzer):
        """Test with custom multipliers."""
        S0 = 100.0
        mu = 0.08
        base_sigma = 0.2
        days = 30
        n_paths = 500
        custom_multipliers = [0.25, 0.5, 1.0, 2.0]

        result = analyzer.analyze_volatility_sensitivity(
            S0=S0,
            mu=mu,
            base_sigma=base_sigma,
            days=days,
            n_paths=n_paths,
            multipliers=custom_multipliers,
        )

        assert set(result.keys()) == set(custom_multipliers)


class TestAnalyzeDriftSensitivity:
    """Tests for drift sensitivity analysis."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed for reproducibility."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def analyzer(self, engine):
        """Create a SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(engine)

    def test_analyze_drift_sensitivity(self, analyzer):
        """Verify optimistic > neutral > pessimistic."""
        S0 = 100.0
        base_mu = 0.08  # 8% annual drift
        sigma = 0.2  # 20% annual volatility
        days = 252  # Full year for clearer drift effect
        n_paths = 5000  # More paths for stable results

        result = analyzer.analyze_drift_sensitivity(
            S0=S0,
            base_mu=base_mu,
            sigma=sigma,
            days=days,
            n_paths=n_paths,
        )

        # Should return a dict with scenario names
        assert isinstance(result, dict)
        assert "pessimistic" in result
        assert "neutral" in result
        assert "optimistic" in result

        # All values should be positive floats
        for scenario, mean_price in result.items():
            assert isinstance(mean_price, float)
            assert mean_price > 0

        # Optimistic should have higher mean price than neutral,
        # and neutral should have higher than pessimistic
        assert result["optimistic"] > result["neutral"], (
            f"Optimistic ({result['optimistic']:.2f}) should be > "
            f"Neutral ({result['neutral']:.2f})"
        )
        assert result["neutral"] > result["pessimistic"], (
            f"Neutral ({result['neutral']:.2f}) should be > "
            f"Pessimistic ({result['pessimistic']:.2f})"
        )


class TestAnalyzeLookbackSensitivity:
    """Tests for lookback period sensitivity analysis."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed for reproducibility."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def analyzer(self, engine):
        """Create a SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(engine)

    def test_analyze_lookback_sensitivity(self, analyzer):
        """Verify dict with lookback days as keys."""
        S0 = 100.0
        mu = 0.08
        days = 30
        n_paths = 1000

        # Different lookback periods with their estimated volatilities
        lookback_volatilities = {
            30: 0.25,   # 30-day lookback with higher vol
            60: 0.22,   # 60-day lookback
            90: 0.20,   # 90-day lookback
            252: 0.18,  # 1-year lookback with lower vol
        }

        result = analyzer.analyze_lookback_sensitivity(
            S0=S0,
            mu=mu,
            days=days,
            n_paths=n_paths,
            lookback_volatilities=lookback_volatilities,
        )

        # Should return a dict with lookback days as keys
        assert isinstance(result, dict)
        assert set(result.keys()) == set(lookback_volatilities.keys())

        # All values should be positive floats
        for lookback_days, mean_price in result.items():
            assert isinstance(mean_price, float)
            assert mean_price > 0


class TestCalculateImpactScores:
    """Tests for impact score calculation."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def analyzer(self, engine):
        """Create a SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(engine)

    def test_calculate_impact_scores(self, analyzer):
        """Verify relative impact scores in 0-100 range."""
        # Create example result ranges (price ranges from different sensitivities)
        volatility_range = {
            0.5: 108.0,
            0.8: 106.0,
            1.0: 105.0,
            1.2: 104.0,
            1.5: 102.0,
        }  # Range: 6.0

        drift_range = {
            "pessimistic": 100.0,
            "neutral": 105.0,
            "optimistic": 112.0,
        }  # Range: 12.0

        lookback_range = {
            30: 103.0,
            60: 104.5,
            90: 105.5,
            252: 106.0,
        }  # Range: 3.0

        vol_impact, drift_impact, lookback_impact = analyzer.calculate_impact_scores(
            volatility_range=volatility_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )

        # All scores should be in 0-100 range
        assert 0 <= vol_impact <= 100, f"Volatility impact {vol_impact} not in 0-100"
        assert 0 <= drift_impact <= 100, f"Drift impact {drift_impact} not in 0-100"
        assert 0 <= lookback_impact <= 100, f"Lookback impact {lookback_impact} not in 0-100"

        # In this case, drift has highest range (12.0), then volatility (6.0), then lookback (3.0)
        # So drift should have highest impact score (100)
        assert drift_impact == 100.0, f"Drift should have max impact (100), got {drift_impact}"
        assert vol_impact > lookback_impact, (
            f"Volatility impact ({vol_impact}) should be > lookback impact ({lookback_impact})"
        )

    def test_calculate_impact_scores_all_equal(self, analyzer):
        """Test when all ranges are equal."""
        volatility_range = {0.5: 105.0, 1.0: 100.0}  # Range: 5.0
        drift_range = {"pessimistic": 95.0, "optimistic": 100.0}  # Range: 5.0
        lookback_range = {30: 102.5, 60: 97.5}  # Range: 5.0

        vol_impact, drift_impact, lookback_impact = analyzer.calculate_impact_scores(
            volatility_range=volatility_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )

        # All should be equal when ranges are equal
        assert vol_impact == drift_impact == lookback_impact

    def test_calculate_impact_scores_zero_range(self, analyzer):
        """Test handling of zero ranges."""
        volatility_range = {0.5: 100.0, 1.0: 100.0}  # Range: 0.0
        drift_range = {"pessimistic": 100.0, "optimistic": 100.0}  # Range: 0.0
        lookback_range = {30: 100.0, 60: 100.0}  # Range: 0.0

        vol_impact, drift_impact, lookback_impact = analyzer.calculate_impact_scores(
            volatility_range=volatility_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )

        # All scores should be 0 or equal when there's no variation
        assert vol_impact >= 0
        assert drift_impact >= 0
        assert lookback_impact >= 0


class TestRunFullSensitivityAnalysis:
    """Tests for complete sensitivity analysis."""

    @pytest.fixture
    def engine(self):
        """Create a SimulationEngine instance with fixed seed for reproducibility."""
        return SimulationEngine(seed=42)

    @pytest.fixture
    def analyzer(self, engine):
        """Create a SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(engine)

    def test_run_full_sensitivity_analysis(self, analyzer):
        """Verify returns SensitivityResult with primary_driver."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 30
        n_paths = 500

        lookback_volatilities = {
            30: 0.25,
            60: 0.22,
            90: 0.20,
        }

        result = analyzer.run_analysis(
            S0=S0,
            mu=mu,
            sigma=sigma,
            days=days,
            n_paths=n_paths,
            lookback_volatilities=lookback_volatilities,
        )

        # Should return a SensitivityResult
        assert isinstance(result, SensitivityResult)

        # Check required fields
        assert 0 <= result.volatility_impact <= 100
        assert 0 <= result.drift_impact <= 100
        assert 0 <= result.lookback_impact <= 100

        # primary_driver should be one of the valid options
        assert result.primary_driver in ["volatility", "drift", "lookback"]

        # The primary_driver should be the one with the highest impact score
        impacts = {
            "volatility": result.volatility_impact,
            "drift": result.drift_impact,
            "lookback": result.lookback_impact,
        }
        max_impact = max(impacts.values())
        expected_drivers = [k for k, v in impacts.items() if v == max_impact]
        assert result.primary_driver in expected_drivers

        # Check that ranges are populated
        assert len(result.volatility_range) > 0
        assert len(result.drift_range) > 0
        assert len(result.lookback_range) > 0

    def test_run_full_sensitivity_analysis_custom_multipliers(self, analyzer):
        """Test with custom volatility multipliers."""
        S0 = 100.0
        mu = 0.08
        sigma = 0.2
        days = 30
        n_paths = 500

        lookback_volatilities = {30: 0.25, 60: 0.22}
        custom_multipliers = [0.25, 0.5, 1.0, 1.5, 2.0]

        result = analyzer.run_analysis(
            S0=S0,
            mu=mu,
            sigma=sigma,
            days=days,
            n_paths=n_paths,
            lookback_volatilities=lookback_volatilities,
            volatility_multipliers=custom_multipliers,
        )

        # Check that custom multipliers are used
        assert set(result.volatility_range.keys()) == set(custom_multipliers)
