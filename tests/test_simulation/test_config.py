"""Tests for Monte Carlo simulation configuration."""

from investment_monitor.config import MonteCarloSettings


class TestMonteCarloSettings:
    """Tests for MonteCarloSettings configuration class."""

    def test_monte_carlo_settings_exists(self):
        """MonteCarloSettings class should exist and be importable."""
        settings = MonteCarloSettings()
        assert settings is not None

    def test_default_score_threshold(self):
        """Default score_threshold should be 80.0."""
        settings = MonteCarloSettings()
        assert settings.score_threshold == 80.0

    def test_default_horizons(self):
        """Default horizons should be [30, 90, 252]."""
        settings = MonteCarloSettings()
        assert settings.horizons == [30, 90, 252]

    def test_default_min_paths(self):
        """Default min_paths should be 1000."""
        settings = MonteCarloSettings()
        assert settings.min_paths == 1000

    def test_default_max_paths(self):
        """Default max_paths should be 50000."""
        settings = MonteCarloSettings()
        assert settings.max_paths == 50000

    def test_default_ci_width_threshold(self):
        """Default ci_width_threshold should be 0.15."""
        settings = MonteCarloSettings()
        assert settings.ci_width_threshold == 0.15

    def test_default_min_lookback_days(self):
        """Default min_lookback_days should be 252."""
        settings = MonteCarloSettings()
        assert settings.min_lookback_days == 252

    def test_default_max_lookback_days(self):
        """Default max_lookback_days should be 1260."""
        settings = MonteCarloSettings()
        assert settings.max_lookback_days == 1260

    def test_default_volatility_multipliers(self):
        """Default volatility_multipliers should be [0.5, 0.8, 1.0, 1.2, 1.5]."""
        settings = MonteCarloSettings()
        assert settings.volatility_multipliers == [0.5, 0.8, 1.0, 1.2, 1.5]

    def test_default_drift_scenarios(self):
        """Default drift_scenarios should be pessimistic, neutral, optimistic."""
        settings = MonteCarloSettings()
        assert settings.drift_scenarios == ["pessimistic", "neutral", "optimistic"]

    def test_default_include_in_reports(self):
        """Default include_in_reports should be True."""
        settings = MonteCarloSettings()
        assert settings.include_in_reports is True

    def test_default_disclaimer(self):
        """Default disclaimer should be set."""
        settings = MonteCarloSettings()
        assert settings.disclaimer == "Simulation based on historical returns. Not a prediction."

    def test_default_scenarios(self):
        """Default scenarios should have all crisis scenarios enabled."""
        settings = MonteCarloSettings()
        expected_scenarios = {
            "base_gbm": True,
            "crisis_2008": True,
            "dotcom_crash": True,
            "covid_crash": True,
            "stagflation_1970s": True,
            "black_monday_1987": True,
            "rising_rates_2022": True,
            "regime_democrat": True,
            "regime_republican": True,
        }
        assert settings.scenarios == expected_scenarios

    def test_custom_score_threshold(self):
        """Custom score_threshold can be set."""
        settings = MonteCarloSettings(score_threshold=75.0)
        assert settings.score_threshold == 75.0

    def test_custom_horizons(self):
        """Custom horizons can be set."""
        settings = MonteCarloSettings(horizons=[30, 60, 180])
        assert settings.horizons == [30, 60, 180]

    def test_custom_paths(self):
        """Custom min_paths and max_paths can be set."""
        settings = MonteCarloSettings(min_paths=5000, max_paths=100000)
        assert settings.min_paths == 5000
        assert settings.max_paths == 100000

    def test_custom_scenarios(self):
        """Custom scenarios can be set."""
        custom_scenarios = {
            "base_gbm": True,
            "crisis_2008": False,
            "dotcom_crash": True,
            "covid_crash": False,
            "stagflation_1970s": True,
            "black_monday_1987": False,
            "rising_rates_2022": True,
            "regime_democrat": False,
            "regime_republican": False,
        }
        settings = MonteCarloSettings(scenarios=custom_scenarios)
        assert settings.scenarios == custom_scenarios
        assert settings.scenarios["crisis_2008"] is False

    def test_custom_disclaimer(self):
        """Custom disclaimer can be set."""
        custom_disclaimer = "Custom risk warning message."
        settings = MonteCarloSettings(disclaimer=custom_disclaimer)
        assert settings.disclaimer == custom_disclaimer

    def test_multiple_instances_independent(self):
        """Multiple instances should have independent mutable defaults."""
        settings1 = MonteCarloSettings()
        settings2 = MonteCarloSettings()

        # Modify settings1's mutable defaults
        settings1.horizons.append(365)
        settings1.scenarios["custom_scenario"] = True

        # settings2 should not be affected
        assert 365 not in settings2.horizons
        assert "custom_scenario" not in settings2.scenarios

    def test_all_settings_can_be_customized(self):
        """All settings can be customized at once."""
        settings = MonteCarloSettings(
            score_threshold=70.0,
            horizons=[7, 14, 30],
            min_paths=500,
            max_paths=10000,
            ci_width_threshold=0.20,
            min_lookback_days=126,
            max_lookback_days=504,
            volatility_multipliers=[0.8, 1.0, 1.2],
            drift_scenarios=["bear", "neutral", "bull"],
            include_in_reports=False,
            disclaimer="Test disclaimer.",
            scenarios={"base_gbm": True, "crisis_2008": False},
        )
        assert settings.score_threshold == 70.0
        assert settings.horizons == [7, 14, 30]
        assert settings.min_paths == 500
        assert settings.max_paths == 10000
        assert settings.ci_width_threshold == 0.20
        assert settings.min_lookback_days == 126
        assert settings.max_lookback_days == 504
        assert settings.volatility_multipliers == [0.8, 1.0, 1.2]
        assert settings.drift_scenarios == ["bear", "neutral", "bull"]
        assert settings.include_in_reports is False
        assert settings.disclaimer == "Test disclaimer."
        assert settings.scenarios == {"base_gbm": True, "crisis_2008": False}
