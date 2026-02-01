"""Tests for Monte Carlo simulation models."""

import pytest
from pydantic import ValidationError

from investment_monitor.simulation.models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationConfig,
    SimulationOutput,
)


class TestScenarioResult:
    """Tests for ScenarioResult model."""

    def test_valid_scenario_result(self):
        result = ScenarioResult(
            name="2008 Financial Crisis",
            mean=124.50,
            median=120.00,
            std=25.30,
            ci_80=(98.0, 142.0),
            ci_95=(85.0, 158.0),
            var_95=-0.189,
            cvar_95=-0.242,
            prob_loss_20pct=0.68,
        )
        assert result.name == "2008 Financial Crisis"
        assert result.mean == 124.50
        assert result.prob_loss_20pct == 0.68

    def test_scenario_result_requires_name(self):
        with pytest.raises(ValidationError):
            ScenarioResult(
                mean=124.50,
                median=120.00,
                std=25.30,
                ci_80=(98.0, 142.0),
                ci_95=(85.0, 158.0),
                var_95=-0.189,
                cvar_95=-0.242,
                prob_loss_20pct=0.68,
            )


class TestHorizonResult:
    """Tests for HorizonResult model."""

    def test_valid_horizon_result(self):
        scenario = ScenarioResult(
            name="Base GBM",
            mean=182.0,
            median=180.0,
            std=15.0,
            ci_80=(171.0, 188.0),
            ci_95=(165.0, 195.0),
            var_95=-0.08,
            cvar_95=-0.12,
            prob_loss_20pct=0.05,
        )
        result = HorizonResult(
            days=30,
            base_mean=182.0,
            base_median=180.0,
            base_std=15.0,
            base_skewness=-0.15,
            base_percentiles={5: 165.0, 25: 175.0, 50: 180.0, 75: 188.0, 95: 195.0},
            base_ci_80=(171.0, 188.0),
            base_ci_95=(165.0, 195.0),
            base_var_95=-0.08,
            base_cvar_95=-0.12,
            scenarios={"base_gbm": scenario},
        )
        assert result.days == 30
        assert result.base_mean == 182.0
        assert "base_gbm" in result.scenarios

    def test_horizon_result_validates_days(self):
        with pytest.raises(ValidationError):
            HorizonResult(
                days=-1,
                base_mean=182.0,
                base_median=180.0,
                base_std=15.0,
                base_skewness=-0.15,
                base_percentiles={},
                base_ci_80=(171.0, 188.0),
                base_ci_95=(165.0, 195.0),
                base_var_95=-0.08,
                base_cvar_95=-0.12,
                scenarios={},
            )


class TestSensitivityResult:
    """Tests for SensitivityResult model."""

    def test_valid_sensitivity_result(self):
        result = SensitivityResult(
            volatility_impact=85.0,
            drift_impact=32.0,
            lookback_impact=18.0,
            primary_driver="volatility",
            volatility_range={0.5: 195.0, 1.0: 182.0, 1.5: 165.0},
            drift_range={"pessimistic": 170.0, "neutral": 182.0, "optimistic": 195.0},
            lookback_range={252: 180.0, 756: 182.0, 1260: 184.0},
        )
        assert result.primary_driver == "volatility"
        assert result.volatility_impact == 85.0


class TestSimulationConfig:
    """Tests for SimulationConfig model."""

    def test_default_config(self):
        config = SimulationConfig()
        assert config.score_threshold == 80.0
        assert config.horizons == [30, 90, 252]
        assert config.min_paths == 1000
        assert config.max_paths == 50000

    def test_custom_config(self):
        config = SimulationConfig(
            score_threshold=75.0,
            horizons=[30, 60],
            min_paths=5000,
        )
        assert config.score_threshold == 75.0
        assert config.horizons == [30, 60]


class TestSimulationOutput:
    """Tests for SimulationOutput model."""

    def test_valid_simulation_output(self):
        scenario = ScenarioResult(
            name="Base GBM",
            mean=182.0,
            median=180.0,
            std=15.0,
            ci_80=(171.0, 188.0),
            ci_95=(165.0, 195.0),
            var_95=-0.08,
            cvar_95=-0.12,
            prob_loss_20pct=0.05,
        )
        horizon = HorizonResult(
            days=30,
            base_mean=182.0,
            base_median=180.0,
            base_std=15.0,
            base_skewness=-0.15,
            base_percentiles={5: 165.0, 25: 175.0, 50: 180.0, 75: 188.0, 95: 195.0},
            base_ci_80=(171.0, 188.0),
            base_ci_95=(165.0, 195.0),
            base_var_95=-0.08,
            base_cvar_95=-0.12,
            scenarios={"base_gbm": scenario},
        )
        sensitivity = SensitivityResult(
            volatility_impact=85.0,
            drift_impact=32.0,
            lookback_impact=18.0,
            primary_driver="volatility",
            volatility_range={0.5: 195.0, 1.0: 182.0, 1.5: 165.0},
            drift_range={"pessimistic": 170.0, "neutral": 182.0, "optimistic": 195.0},
            lookback_range={252: 180.0, 756: 182.0, 1260: 184.0},
        )
        output = SimulationOutput(
            ticker="AAPL",
            entry_price=178.50,
            composite_score=85.0,
            num_simulations=10000,
            lookback_days=756,
            volatility=0.25,
            drift=0.08,
            results={30: horizon},
            sensitivity=sensitivity,
        )
        assert output.ticker == "AAPL"
        assert output.entry_price == 178.50
        assert 30 in output.results
