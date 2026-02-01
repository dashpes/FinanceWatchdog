"""Pydantic models for Monte Carlo simulation."""

from pydantic import BaseModel, Field


class ScenarioResult(BaseModel):
    """Results from a single stress scenario simulation."""

    name: str
    mean: float
    median: float
    std: float
    ci_80: tuple[float, float]
    ci_95: tuple[float, float]
    var_95: float  # Value at Risk (as return, e.g., -0.189 = -18.9%)
    cvar_95: float  # Conditional VaR
    prob_loss_20pct: float  # P(loss > 20%)


class HorizonResult(BaseModel):
    """Results for a single time horizon (30, 90, or 252 days)."""

    days: int = Field(gt=0)

    # Base case (GBM) statistics
    base_mean: float
    base_median: float
    base_std: float
    base_skewness: float
    base_percentiles: dict[int, float]  # {5: 142.5, 25: 158.2, ...}
    base_ci_80: tuple[float, float]
    base_ci_95: tuple[float, float]
    base_var_95: float
    base_cvar_95: float

    # Stress scenarios
    scenarios: dict[str, ScenarioResult]


class SensitivityResult(BaseModel):
    """Results from sensitivity analysis."""

    volatility_impact: float  # 0-100 impact score
    drift_impact: float
    lookback_impact: float
    primary_driver: str  # Most sensitive input
    volatility_range: dict[float, float]  # multiplier -> mean price
    drift_range: dict[str, float]  # scenario name -> mean price
    lookback_range: dict[int, float]  # days -> mean price


class SimulationConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    # Gating
    score_threshold: float = 80.0

    # Simulation parameters
    horizons: list[int] = [30, 90, 252]
    min_paths: int = 1000
    max_paths: int = 50000
    ci_width_threshold: float = 0.15

    # Historical data
    min_lookback_days: int = 252
    max_lookback_days: int = 1260

    # Sensitivity analysis
    volatility_multipliers: list[float] = [0.5, 0.8, 1.0, 1.2, 1.5]
    drift_scenarios: list[str] = ["pessimistic", "neutral", "optimistic"]

    # Scenarios enabled
    scenarios_enabled: dict[str, bool] = {
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

    # Report settings
    include_in_reports: bool = True
    disclaimer: str = "Simulation based on historical returns. Not a prediction."


class SimulationOutput(BaseModel):
    """Complete simulation output for a ticker."""

    ticker: str
    entry_price: float
    composite_score: float

    # Parameters used
    num_simulations: int
    lookback_days: int
    volatility: float  # Annualized
    drift: float  # Annualized

    # Results per horizon
    results: dict[int, HorizonResult]  # {30: HorizonResult, 90: ..., 252: ...}

    # Sensitivity analysis
    sensitivity: SensitivityResult
