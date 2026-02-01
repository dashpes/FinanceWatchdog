"""Monte Carlo Analyzer - main orchestrator for risk simulation.

This module provides the MonteCarloAnalyzer class which ties together:
- SimulationEngine for GBM and bootstrap simulations
- CrisisDataLoader for historical stress scenarios
- SensitivityAnalyzer for input parameter sensitivity

It serves as the main entry point for running Monte Carlo risk analysis
on stocks that meet the quality threshold from the research system.
"""

from datetime import date, timedelta

import numpy as np
import yfinance as yf
from loguru import logger

from .crisis_loader import CrisisDataLoader, CrisisScenario
from .engine import SimulationEngine
from .models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationConfig,
    SimulationOutput,
)
from .sensitivity import SensitivityAnalyzer


# Mapping from config scenario names to CrisisScenario enum
_SCENARIO_MAP = {
    "crisis_2008": CrisisScenario.CRISIS_2008,
    "dotcom_crash": CrisisScenario.DOTCOM_CRASH,
    "covid_crash": CrisisScenario.COVID_CRASH,
    "stagflation_1970s": CrisisScenario.STAGFLATION_1970S,
    "black_monday_1987": CrisisScenario.BLACK_MONDAY_1987,
    "rising_rates_2022": CrisisScenario.RISING_RATES_2022,
    "regime_democrat": CrisisScenario.REGIME_DEMOCRAT,
    "regime_republican": CrisisScenario.REGIME_REPUBLICAN,
}


class MonteCarloAnalyzer:
    """Main orchestrator for Monte Carlo simulation analysis.

    This class coordinates all simulation components to produce comprehensive
    risk analysis for a given stock. It handles:
    - Gating based on composite score threshold
    - Historical parameter estimation from price data
    - Base case GBM simulations across multiple horizons
    - Stress testing with historical crisis scenarios
    - Sensitivity analysis for input parameters

    Attributes:
        config: SimulationConfig with thresholds and parameters.
        _engine: SimulationEngine for running GBM and bootstrap simulations.
        _crisis_loader: CrisisDataLoader for loading historical crisis data.
        _sensitivity_analyzer: SensitivityAnalyzer for input sensitivity testing.

    Example:
        >>> analyzer = MonteCarloAnalyzer(seed=42)
        >>> output = analyzer.analyze("AAPL", entry_price=150.0, composite_score=85.0)
        >>> print(f"30-day VaR: {output.results[30].base_var_95:.2%}")
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Monte Carlo analyzer.

        Args:
            config: Optional SimulationConfig. If None, uses defaults.
            seed: Optional random seed for reproducibility.
        """
        self.config = config or SimulationConfig()
        self._engine = SimulationEngine(seed=seed)
        self._crisis_loader = CrisisDataLoader()
        self._sensitivity_analyzer = SensitivityAnalyzer(self._engine)

    def should_run_simulation(
        self,
        composite_score: float,
        force: bool = False,
    ) -> bool:
        """Check if simulation should be run based on score threshold.

        Simulations are gated by the composite score to focus compute
        resources on high-quality investment candidates.

        Args:
            composite_score: The research system's composite score (0-100).
            force: If True, bypass the threshold check.

        Returns:
            True if simulation should run, False otherwise.
        """
        if force:
            return True
        return composite_score >= self.config.score_threshold

    def calculate_historical_parameters(
        self,
        prices: np.ndarray,
    ) -> tuple[float, float]:
        """Calculate annualized drift and volatility from price history.

        Uses log returns to estimate historical drift (mu) and volatility (sigma).
        These parameters are then used in GBM simulations.

        Args:
            prices: Array of historical prices (oldest first).

        Returns:
            Tuple of (annualized_drift, annualized_volatility).
        """
        # Calculate daily log returns
        log_returns = np.diff(np.log(prices))

        # Daily statistics
        daily_mean = np.mean(log_returns)
        daily_std = np.std(log_returns, ddof=1)

        # Annualize (252 trading days per year)
        annualized_drift = daily_mean * 252
        annualized_volatility = daily_std * np.sqrt(252)

        return float(annualized_drift), float(annualized_volatility)

    def run_base_case_simulation(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> dict:
        """Run GBM simulation and compute statistics.

        Args:
            S0: Initial stock price.
            mu: Annualized drift (expected return).
            sigma: Annualized volatility.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths.

        Returns:
            Dictionary with simulation statistics:
            - mean, median, std: Basic statistics of terminal prices
            - percentiles: Dict mapping percentile to price
            - ci_80, ci_95: Confidence interval tuples
            - var_95, cvar_95: Value at Risk metrics (as returns)
            - skewness: Skewness of terminal price distribution
        """
        # Run GBM simulation
        terminal_prices = self._engine.simulate_gbm(S0, mu, sigma, days, n_paths)

        # Basic statistics
        mean = float(np.mean(terminal_prices))
        median = float(np.median(terminal_prices))
        std = float(np.std(terminal_prices))

        # Percentiles
        percentiles = {
            p: float(np.percentile(terminal_prices, p))
            for p in [5, 25, 50, 75, 95]
        }

        # Confidence intervals
        ci_80 = (
            float(np.percentile(terminal_prices, 10)),
            float(np.percentile(terminal_prices, 90)),
        )
        ci_95 = (
            float(np.percentile(terminal_prices, 2.5)),
            float(np.percentile(terminal_prices, 97.5)),
        )

        # Risk metrics
        var_95 = self._engine.calculate_var(terminal_prices, S0, confidence=0.95)
        cvar_95 = self._engine.calculate_cvar(terminal_prices, S0, confidence=0.95)

        # Skewness (using numpy formula: E[(X-mu)^3] / sigma^3)
        n = len(terminal_prices)
        m3 = np.mean((terminal_prices - mean) ** 3)
        skewness = float(m3 / (std ** 3)) if std > 0 else 0.0

        return {
            "mean": mean,
            "median": median,
            "std": std,
            "percentiles": percentiles,
            "ci_80": ci_80,
            "ci_95": ci_95,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": skewness,
        }

    def run_stress_scenario(
        self,
        S0: float,
        scenario: CrisisScenario,
        days: int,
        n_paths: int,
        beta: float = 1.0,
    ) -> dict:
        """Run bootstrap simulation for a stress scenario.

        Uses historical crisis returns with beta adjustment to simulate
        how the stock might perform under similar conditions.

        Args:
            S0: Initial stock price.
            scenario: The CrisisScenario to simulate.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths.
            beta: Stock's beta relative to S&P 500 for return scaling.

        Returns:
            Dictionary with scenario results:
            - name: Human-readable scenario name
            - mean, median, std: Basic statistics
            - ci_80, ci_95: Confidence intervals
            - var_95, cvar_95: Risk metrics
            - prob_loss_20pct: Probability of 20%+ loss
        """
        # Load crisis returns and apply beta adjustment
        crisis_returns = self._crisis_loader.load_crisis_returns(scenario)
        adjusted_returns = self._crisis_loader.apply_beta_adjustment(
            crisis_returns, beta
        )

        # Run bootstrap simulation
        terminal_prices = self._engine.simulate_bootstrap(
            S0, adjusted_returns, days, n_paths
        )

        # Get scenario metadata for the name
        metadata = self._crisis_loader.get_scenario_metadata(scenario)
        name = metadata["name"]

        # Basic statistics
        mean = float(np.mean(terminal_prices))
        median = float(np.median(terminal_prices))
        std = float(np.std(terminal_prices))

        # Confidence intervals
        ci_80 = (
            float(np.percentile(terminal_prices, 10)),
            float(np.percentile(terminal_prices, 90)),
        )
        ci_95 = (
            float(np.percentile(terminal_prices, 2.5)),
            float(np.percentile(terminal_prices, 97.5)),
        )

        # Risk metrics
        var_95 = self._engine.calculate_var(terminal_prices, S0, confidence=0.95)
        cvar_95 = self._engine.calculate_cvar(terminal_prices, S0, confidence=0.95)

        # Probability of 20%+ loss
        returns = terminal_prices / S0 - 1
        prob_loss_20pct = float(np.mean(returns < -0.20))

        return {
            "name": name,
            "mean": mean,
            "median": median,
            "std": std,
            "ci_80": ci_80,
            "ci_95": ci_95,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "prob_loss_20pct": prob_loss_20pct,
        }

    def build_horizon_result(
        self,
        days: int,
        base_stats: dict,
        scenario_results: list[dict],
    ) -> HorizonResult:
        """Build HorizonResult from statistics dictionaries.

        Args:
            days: Number of days for this horizon.
            base_stats: Dictionary of base case statistics.
            scenario_results: List of scenario result dictionaries.

        Returns:
            HorizonResult with all statistics and scenarios.
        """
        # Build scenarios dict
        scenarios = {}
        for sr in scenario_results:
            scenario_result = ScenarioResult(
                name=sr["name"],
                mean=sr["mean"],
                median=sr["median"],
                std=sr["std"],
                ci_80=sr["ci_80"],
                ci_95=sr["ci_95"],
                var_95=sr["var_95"],
                cvar_95=sr["cvar_95"],
                prob_loss_20pct=sr["prob_loss_20pct"],
            )
            scenarios[sr["name"]] = scenario_result

        return HorizonResult(
            days=days,
            base_mean=base_stats["mean"],
            base_median=base_stats["median"],
            base_std=base_stats["std"],
            base_skewness=base_stats["skewness"],
            base_percentiles=base_stats["percentiles"],
            base_ci_80=base_stats["ci_80"],
            base_ci_95=base_stats["ci_95"],
            base_var_95=base_stats["var_95"],
            base_cvar_95=base_stats["cvar_95"],
            scenarios=scenarios,
        )

    def _fetch_price_history(
        self,
        ticker: str,
        days: int,
    ) -> np.ndarray:
        """Fetch historical prices from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol.
            days: Number of calendar days of history to fetch.

        Returns:
            NumPy array of closing prices (oldest first).

        Raises:
            ValueError: If no price data is available.
        """
        end_date = date.today()
        # Add buffer for weekends/holidays
        start_date = end_date - timedelta(days=int(days * 1.5))

        logger.debug(f"Fetching price history for {ticker} from {start_date}")

        data = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
        )

        if data.empty:
            raise ValueError(f"No price data available for {ticker}")

        # Extract close prices
        if "Close" in data.columns:
            prices = data["Close"].dropna().values
        else:
            raise ValueError(f"No Close prices in data for {ticker}")

        if len(prices) == 0:
            raise ValueError(f"No valid close prices for {ticker}")

        return prices.astype(np.float64)

    def _calculate_beta(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> float:
        """Calculate stock beta relative to S&P 500.

        Uses linear regression of stock returns against market returns.

        Args:
            ticker: Stock ticker symbol.
            lookback_days: Number of trading days for calculation.

        Returns:
            Beta coefficient (1.0 means moves with market).
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))

        logger.debug(f"Calculating beta for {ticker} over {lookback_days} days")

        # Download stock and SPY data together
        data = yf.download(
            tickers=[ticker, "SPY"],
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            group_by="ticker",
        )

        if data.empty:
            logger.warning(f"No data for beta calculation, using default beta=1.0")
            return 1.0

        try:
            # Extract close prices
            stock_prices = data[ticker]["Close"].dropna()
            market_prices = data["SPY"]["Close"].dropna()

            # Align indices
            common_idx = stock_prices.index.intersection(market_prices.index)
            if len(common_idx) < 30:
                logger.warning(f"Insufficient data for beta, using default beta=1.0")
                return 1.0

            stock_prices = stock_prices.loc[common_idx]
            market_prices = market_prices.loc[common_idx]

            # Calculate returns
            stock_returns = np.diff(np.log(stock_prices.values))
            market_returns = np.diff(np.log(market_prices.values))

            # Calculate beta using linear regression
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance
            return float(np.clip(beta, 0.1, 5.0))  # Reasonable bounds

        except Exception as e:
            logger.warning(f"Beta calculation failed: {e}, using default beta=1.0")
            return 1.0

    def _determine_lookback_days(
        self,
        available_days: int,
    ) -> int:
        """Determine optimal lookback period based on data availability.

        Args:
            available_days: Number of trading days of data available.

        Returns:
            Optimal lookback days to use.
        """
        # Use the minimum of available data and max configured lookback
        optimal = min(available_days, self.config.max_lookback_days)

        # But at least use min_lookback if we have it
        if available_days >= self.config.min_lookback_days:
            optimal = min(available_days, self.config.max_lookback_days)
        else:
            # Use whatever we have
            optimal = available_days

        return optimal

    def analyze(
        self,
        ticker: str,
        entry_price: float,
        composite_score: float,
        force: bool = False,
    ) -> SimulationOutput:
        """Run complete Monte Carlo analysis for a ticker.

        This is the main entry point that orchestrates:
        1. Gating check based on composite score
        2. Price history fetching
        3. Historical parameter estimation
        4. Base case simulations for all horizons
        5. Stress scenario simulations
        6. Sensitivity analysis

        Args:
            ticker: Stock ticker symbol.
            entry_price: Current/entry price for the stock.
            composite_score: Research system's composite score (0-100).
            force: If True, bypass the score threshold check.

        Returns:
            SimulationOutput with complete analysis results.

        Raises:
            ValueError: If score is below threshold and force=False.
        """
        # Check if we should run
        if not self.should_run_simulation(composite_score, force):
            raise ValueError(
                f"Score {composite_score} below threshold {self.config.score_threshold}. "
                "Use force=True to override."
            )

        logger.info(f"Starting Monte Carlo analysis for {ticker}")

        # Fetch price history (get extra for lookback determination)
        max_needed_days = self.config.max_lookback_days + max(self.config.horizons)
        prices = self._fetch_price_history(ticker, max_needed_days)

        # Determine actual lookback
        available_days = len(prices)
        lookback_days = self._determine_lookback_days(available_days)

        # Use the most recent lookback_days for parameter estimation
        lookback_prices = prices[-lookback_days:] if len(prices) > lookback_days else prices

        # Calculate historical parameters
        mu, sigma = self.calculate_historical_parameters(lookback_prices)
        logger.debug(f"{ticker}: drift={mu:.4f}, volatility={sigma:.4f}")

        # Calculate beta for stress scenarios
        beta = self._calculate_beta(ticker)
        logger.debug(f"{ticker}: beta={beta:.2f}")

        # Determine number of paths (could use adaptive in future)
        n_paths = self.config.min_paths

        # Run simulations for each horizon
        results: dict[int, HorizonResult] = {}

        for horizon_days in self.config.horizons:
            logger.debug(f"Running {horizon_days}-day simulations for {ticker}")

            # Base case simulation
            base_stats = self.run_base_case_simulation(
                entry_price, mu, sigma, horizon_days, n_paths
            )

            # Stress scenarios
            scenario_results = []
            for scenario_name, enabled in self.config.scenarios_enabled.items():
                if not enabled or scenario_name == "base_gbm":
                    continue

                if scenario_name in _SCENARIO_MAP:
                    scenario = _SCENARIO_MAP[scenario_name]
                    try:
                        sr = self.run_stress_scenario(
                            entry_price, scenario, horizon_days, n_paths, beta
                        )
                        scenario_results.append(sr)
                    except Exception as e:
                        logger.warning(f"Scenario {scenario_name} failed: {e}")

            # Build horizon result
            horizon_result = self.build_horizon_result(
                horizon_days, base_stats, scenario_results
            )
            results[horizon_days] = horizon_result

        # Run sensitivity analysis
        # Create lookback volatilities for different windows
        lookback_volatilities = {}
        for lb_days in [30, 60, 90, 252]:
            if lb_days <= len(prices):
                lb_prices = prices[-lb_days:]
                _, lb_vol = self.calculate_historical_parameters(lb_prices)
                lookback_volatilities[lb_days] = lb_vol

        # Use median horizon for sensitivity analysis
        median_horizon = sorted(self.config.horizons)[len(self.config.horizons) // 2]

        sensitivity = self._sensitivity_analyzer.run_analysis(
            S0=entry_price,
            mu=mu,
            sigma=sigma,
            days=median_horizon,
            n_paths=n_paths,
            lookback_volatilities=lookback_volatilities,
            volatility_multipliers=self.config.volatility_multipliers,
        )

        logger.info(f"Completed Monte Carlo analysis for {ticker}")

        return SimulationOutput(
            ticker=ticker,
            entry_price=entry_price,
            composite_score=composite_score,
            num_simulations=n_paths,
            lookback_days=lookback_days,
            volatility=sigma,
            drift=mu,
            results=results,
            sensitivity=sensitivity,
        )
