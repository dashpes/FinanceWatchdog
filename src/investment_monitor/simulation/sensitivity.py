"""Sensitivity analysis for Monte Carlo simulations.

This module provides the SensitivityAnalyzer class which tests how sensitive
simulation results are to input assumptions including volatility, drift, and
lookback period.
"""

import numpy as np

from .engine import SimulationEngine
from .models import SensitivityResult


class SensitivityAnalyzer:
    """Analyzes sensitivity of simulation results to input parameters.

    This class uses a SimulationEngine to test how terminal price distributions
    change when varying input assumptions. It helps identify which parameters
    have the greatest impact on simulation outcomes.

    Attributes:
        _engine: The underlying SimulationEngine for running simulations.

    Example:
        >>> engine = SimulationEngine(seed=42)
        >>> analyzer = SensitivityAnalyzer(engine)
        >>> result = analyzer.run_analysis(
        ...     S0=100.0, mu=0.08, sigma=0.2, days=252, n_paths=10000,
        ...     lookback_volatilities={30: 0.25, 60: 0.22, 90: 0.20}
        ... )
        >>> print(f"Primary driver: {result.primary_driver}")
    """

    # Default volatility multipliers for sensitivity testing
    DEFAULT_VOLATILITY_MULTIPLIERS = [0.5, 0.8, 1.0, 1.2, 1.5]

    def __init__(self, engine: SimulationEngine) -> None:
        """Initialize with a simulation engine.

        Args:
            engine: SimulationEngine instance to use for simulations.
        """
        self._engine = engine

    def analyze_volatility_sensitivity(
        self,
        S0: float,
        mu: float,
        base_sigma: float,
        days: int,
        n_paths: int,
        multipliers: list[float] | None = None,
    ) -> dict[float, float]:
        """Test different volatility multipliers and measure impact.

        Runs GBM simulations with volatility scaled by different multipliers
        to understand how sensitive results are to volatility assumptions.

        Args:
            S0: Initial stock price.
            mu: Annual drift (expected return).
            base_sigma: Base annual volatility.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths per scenario.
            multipliers: Optional list of volatility multipliers.
                        Defaults to [0.5, 0.8, 1.0, 1.2, 1.5].

        Returns:
            Dictionary mapping volatility multiplier to mean terminal price.

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> analyzer = SensitivityAnalyzer(engine)
            >>> results = analyzer.analyze_volatility_sensitivity(
            ...     S0=100.0, mu=0.08, base_sigma=0.2, days=252, n_paths=1000
            ... )
            >>> results[1.0]  # Mean price at base volatility
            108.45
        """
        if multipliers is None:
            multipliers = self.DEFAULT_VOLATILITY_MULTIPLIERS

        results: dict[float, float] = {}

        for mult in multipliers:
            adjusted_sigma = base_sigma * mult
            terminal_prices = self._engine.simulate_gbm(
                S0=S0,
                mu=mu,
                sigma=adjusted_sigma,
                days=days,
                n_paths=n_paths,
            )
            results[mult] = float(np.mean(terminal_prices))

        return results

    def analyze_drift_sensitivity(
        self,
        S0: float,
        base_mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> dict[str, float]:
        """Test pessimistic, neutral, and optimistic drift scenarios.

        Runs GBM simulations with different drift assumptions:
        - Pessimistic: 0% annual drift
        - Neutral: base_mu (provided drift)
        - Optimistic: base_mu + 2%

        Args:
            S0: Initial stock price.
            base_mu: Neutral drift assumption (annual return).
            sigma: Annual volatility.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths per scenario.

        Returns:
            Dictionary mapping scenario name to mean terminal price.

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> analyzer = SensitivityAnalyzer(engine)
            >>> results = analyzer.analyze_drift_sensitivity(
            ...     S0=100.0, base_mu=0.08, sigma=0.2, days=252, n_paths=1000
            ... )
            >>> results["optimistic"] > results["neutral"]
            True
        """
        scenarios = {
            "pessimistic": 0.0,
            "neutral": base_mu,
            "optimistic": base_mu + 0.02,  # +2%
        }

        results: dict[str, float] = {}

        for scenario_name, drift in scenarios.items():
            terminal_prices = self._engine.simulate_gbm(
                S0=S0,
                mu=drift,
                sigma=sigma,
                days=days,
                n_paths=n_paths,
            )
            results[scenario_name] = float(np.mean(terminal_prices))

        return results

    def analyze_lookback_sensitivity(
        self,
        S0: float,
        mu: float,
        days: int,
        n_paths: int,
        lookback_volatilities: dict[int, float],
    ) -> dict[int, float]:
        """Test different lookback periods with their estimated volatilities.

        Different lookback periods capture different market regimes and
        produce different volatility estimates. This method tests how
        using volatilities from different lookback windows affects results.

        Args:
            S0: Initial stock price.
            mu: Annual drift (expected return).
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths per scenario.
            lookback_volatilities: Dictionary mapping lookback days to
                                   estimated volatility for that period.

        Returns:
            Dictionary mapping lookback days to mean terminal price.

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> analyzer = SensitivityAnalyzer(engine)
            >>> lookbacks = {30: 0.25, 60: 0.22, 90: 0.20, 252: 0.18}
            >>> results = analyzer.analyze_lookback_sensitivity(
            ...     S0=100.0, mu=0.08, days=30, n_paths=1000,
            ...     lookback_volatilities=lookbacks
            ... )
        """
        results: dict[int, float] = {}

        for lookback_days, volatility in lookback_volatilities.items():
            terminal_prices = self._engine.simulate_gbm(
                S0=S0,
                mu=mu,
                sigma=volatility,
                days=days,
                n_paths=n_paths,
            )
            results[lookback_days] = float(np.mean(terminal_prices))

        return results

    def calculate_impact_scores(
        self,
        volatility_range: dict[float, float],
        drift_range: dict[str, float],
        lookback_range: dict[int, float],
    ) -> tuple[float, float, float]:
        """Calculate normalized impact scores (0-100) for each input.

        Impact scores are calculated based on the range of mean terminal prices
        produced by varying each input. The input with the highest range gets
        a score of 100, and others are scaled proportionally.

        Args:
            volatility_range: Results from analyze_volatility_sensitivity.
            drift_range: Results from analyze_drift_sensitivity.
            lookback_range: Results from analyze_lookback_sensitivity.

        Returns:
            Tuple of (vol_impact, drift_impact, lookback_impact), each 0-100.

        Example:
            >>> vol_range = {0.5: 108.0, 1.0: 105.0, 1.5: 102.0}
            >>> drift_range = {"pessimistic": 100.0, "neutral": 105.0, "optimistic": 112.0}
            >>> lookback_range = {30: 103.0, 90: 105.5, 252: 106.0}
            >>> vol_impact, drift_impact, lookback_impact = analyzer.calculate_impact_scores(
            ...     vol_range, drift_range, lookback_range
            ... )
        """
        # Calculate the price range for each input type
        vol_values = list(volatility_range.values())
        drift_values = list(drift_range.values())
        lookback_values = list(lookback_range.values())

        vol_spread = max(vol_values) - min(vol_values) if vol_values else 0.0
        drift_spread = max(drift_values) - min(drift_values) if drift_values else 0.0
        lookback_spread = (
            max(lookback_values) - min(lookback_values) if lookback_values else 0.0
        )

        # Find the maximum spread for normalization
        max_spread = max(vol_spread, drift_spread, lookback_spread)

        # Handle edge case where all spreads are zero
        if max_spread == 0:
            # All equal impact (could be 0 or 100, using 100 for equal contribution)
            return (100.0, 100.0, 100.0)

        # Normalize to 0-100 scale
        vol_impact = (vol_spread / max_spread) * 100.0
        drift_impact = (drift_spread / max_spread) * 100.0
        lookback_impact = (lookback_spread / max_spread) * 100.0

        return (vol_impact, drift_impact, lookback_impact)

    def run_analysis(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int,
        lookback_volatilities: dict[int, float],
        volatility_multipliers: list[float] | None = None,
    ) -> SensitivityResult:
        """Run complete sensitivity analysis and return SensitivityResult.

        Performs volatility, drift, and lookback sensitivity analyses,
        calculates impact scores, and determines the primary driver.

        Args:
            S0: Initial stock price.
            mu: Annual drift (expected return).
            sigma: Base annual volatility.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths per scenario.
            lookback_volatilities: Dictionary mapping lookback days to
                                   estimated volatility for that period.
            volatility_multipliers: Optional list of volatility multipliers.
                                   Defaults to [0.5, 0.8, 1.0, 1.2, 1.5].

        Returns:
            SensitivityResult containing impact scores, primary driver,
            and detailed results for each sensitivity test.

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> analyzer = SensitivityAnalyzer(engine)
            >>> result = analyzer.run_analysis(
            ...     S0=100.0, mu=0.08, sigma=0.2, days=252, n_paths=10000,
            ...     lookback_volatilities={30: 0.25, 60: 0.22, 90: 0.20}
            ... )
            >>> print(f"Primary driver: {result.primary_driver}")
        """
        # Run individual sensitivity analyses
        volatility_range = self.analyze_volatility_sensitivity(
            S0=S0,
            mu=mu,
            base_sigma=sigma,
            days=days,
            n_paths=n_paths,
            multipliers=volatility_multipliers,
        )

        drift_range = self.analyze_drift_sensitivity(
            S0=S0,
            base_mu=mu,
            sigma=sigma,
            days=days,
            n_paths=n_paths,
        )

        lookback_range = self.analyze_lookback_sensitivity(
            S0=S0,
            mu=mu,
            days=days,
            n_paths=n_paths,
            lookback_volatilities=lookback_volatilities,
        )

        # Calculate impact scores
        vol_impact, drift_impact, lookback_impact = self.calculate_impact_scores(
            volatility_range=volatility_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )

        # Determine primary driver (highest impact)
        impacts = {
            "volatility": vol_impact,
            "drift": drift_impact,
            "lookback": lookback_impact,
        }
        primary_driver = max(impacts, key=lambda k: impacts[k])

        return SensitivityResult(
            volatility_impact=vol_impact,
            drift_impact=drift_impact,
            lookback_impact=lookback_impact,
            primary_driver=primary_driver,
            volatility_range=volatility_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )
