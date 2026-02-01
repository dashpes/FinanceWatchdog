"""Core Monte Carlo simulation engine for GBM and bootstrap simulations.

This module provides the SimulationEngine class which implements:
- Geometric Brownian Motion (GBM) simulation for price paths
- Block bootstrap simulation using historical crisis returns
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Adaptive path count determination based on confidence interval width
"""

import numpy as np
from numpy.random import Generator


class SimulationEngine:
    """Monte Carlo simulation engine for risk analysis.

    This class provides methods for simulating stock price paths using
    Geometric Brownian Motion (GBM) and block bootstrap methods. It also
    provides risk metric calculations including VaR and CVaR.

    The engine uses numpy's default_rng for random number generation,
    supporting optional seeds for reproducibility.

    Attributes:
        _rng: NumPy random number generator.

    Example:
        >>> engine = SimulationEngine(seed=42)
        >>> terminal_prices = engine.simulate_gbm(
        ...     S0=100.0, mu=0.08, sigma=0.2, days=252, n_paths=10000
        ... )
        >>> var_95 = engine.calculate_var(terminal_prices, entry_price=100.0)
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the simulation engine.

        Args:
            seed: Optional random seed for reproducibility. If None,
                  the random number generator is initialized without a seed,
                  producing non-deterministic results.
        """
        self._rng: Generator = np.random.default_rng(seed)

    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> np.ndarray:
        """Simulate stock price paths using Geometric Brownian Motion.

        Implements the GBM model: dS = mu*S*dt + sigma*S*dW

        The simulation uses the exact solution for GBM:
            S(T) = S0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)

        where Z ~ N(0, 1) and T is the time horizon in years.

        Args:
            S0: Initial stock price.
            mu: Expected annual return (drift).
            sigma: Annual volatility (standard deviation of returns).
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths.

        Returns:
            NumPy array of terminal prices with shape (n_paths,).

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> prices = engine.simulate_gbm(
            ...     S0=100.0, mu=0.08, sigma=0.2, days=252, n_paths=1000
            ... )
            >>> prices.shape
            (1000,)
        """
        # Time step in years (252 trading days per year)
        dt = 1 / 252
        T = days * dt

        # Generate standard normal random variables
        Z = self._rng.standard_normal(n_paths)

        # Exact GBM solution for terminal price:
        # S(T) = S0 * exp((mu - sigma^2/2)*T + sigma*sqrt(T)*Z)
        drift_term = (mu - 0.5 * sigma**2) * T
        diffusion_term = sigma * np.sqrt(T) * Z

        terminal_prices = S0 * np.exp(drift_term + diffusion_term)

        return terminal_prices.astype(np.float64)

    def simulate_bootstrap(
        self,
        S0: float,
        crisis_returns: np.ndarray,
        days: int,
        n_paths: int,
        block_size: int = 5,
    ) -> np.ndarray:
        """Simulate stock price paths using block bootstrap on historical returns.

        Block bootstrap preserves autocorrelation in the return series by
        sampling contiguous blocks of returns rather than individual observations.

        For short crisis periods (fewer returns than block_size), falls back
        to sampling individual returns with replacement.

        Args:
            S0: Initial stock price.
            crisis_returns: Array of historical daily log returns.
            days: Number of trading days to simulate.
            n_paths: Number of simulation paths.
            block_size: Size of blocks to sample. Defaults to 5 days
                       to preserve weekly return patterns.

        Returns:
            NumPy array of terminal prices with shape (n_paths,).

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> crisis_returns = np.array([-0.02, -0.01, 0.005, -0.03, -0.015])
            >>> prices = engine.simulate_bootstrap(
            ...     S0=100.0, crisis_returns=crisis_returns, days=30, n_paths=1000
            ... )
        """
        n_returns = len(crisis_returns)

        # Handle short crisis periods gracefully
        if n_returns < block_size:
            # Fall back to individual sampling when crisis period is too short
            effective_block_size = 1
        else:
            effective_block_size = block_size

        # Calculate number of blocks needed per path
        # We need 'days' returns for each path
        n_blocks = int(np.ceil(days / effective_block_size))

        # Initialize cumulative returns for each path
        cumulative_returns = np.zeros(n_paths, dtype=np.float64)

        # Generate block starting indices for all paths at once
        max_start_idx = max(1, n_returns - effective_block_size + 1)
        block_starts = self._rng.integers(
            0, max_start_idx, size=(n_paths, n_blocks)
        )

        # Accumulate returns from sampled blocks
        for path_idx in range(n_paths):
            returns_sampled = 0
            for block_idx in range(n_blocks):
                start = block_starts[path_idx, block_idx]
                end = min(start + effective_block_size, n_returns)

                # Extract block returns
                block_returns = crisis_returns[start:end]

                # Only use as many returns as needed
                remaining = days - returns_sampled
                n_to_use = min(len(block_returns), remaining)

                cumulative_returns[path_idx] += np.sum(block_returns[:n_to_use])
                returns_sampled += n_to_use

                if returns_sampled >= days:
                    break

        # Convert cumulative log returns to terminal prices
        terminal_prices = S0 * np.exp(cumulative_returns)

        return terminal_prices.astype(np.float64)

    def calculate_var(
        self,
        terminal_prices: np.ndarray,
        entry_price: float,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Value at Risk (VaR) as a return.

        VaR represents the maximum expected loss at a given confidence level.
        For example, 95% VaR is the loss level that is exceeded only 5% of the time.

        Args:
            terminal_prices: Array of simulated terminal prices.
            entry_price: Initial entry price for calculating returns.
            confidence: Confidence level (e.g., 0.95 for 95% VaR).
                       Defaults to 0.95.

        Returns:
            VaR as a return (e.g., -0.20 means -20% loss).
            Negative values indicate losses.

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> prices = np.array([90, 95, 100, 105, 110, 85, 80, 75, 70, 65])
            >>> var = engine.calculate_var(prices, entry_price=100.0)
            >>> var < 0  # VaR is typically negative (indicating a loss)
            True
        """
        # Calculate simple returns
        returns = terminal_prices / entry_price - 1

        # VaR is the percentile corresponding to (1 - confidence)
        # For 95% confidence, we want the 5th percentile
        percentile = (1 - confidence) * 100
        var = float(np.percentile(returns, percentile))

        return var

    def calculate_cvar(
        self,
        terminal_prices: np.ndarray,
        entry_price: float,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.
        It represents the average of all returns below the VaR threshold,
        providing a more complete picture of tail risk.

        Args:
            terminal_prices: Array of simulated terminal prices.
            entry_price: Initial entry price for calculating returns.
            confidence: Confidence level (e.g., 0.95 for 95% CVaR).
                       Defaults to 0.95.

        Returns:
            CVaR as a return (e.g., -0.25 means -25% expected loss
            when losses exceed VaR).

        Example:
            >>> engine = SimulationEngine(seed=42)
            >>> prices = np.array([90, 95, 100, 105, 110, 85, 80, 75, 70, 65])
            >>> cvar = engine.calculate_cvar(prices, entry_price=100.0)
            >>> var = engine.calculate_var(prices, entry_price=100.0)
            >>> cvar <= var  # CVaR is more extreme than VaR
            True
        """
        # Calculate simple returns
        returns = terminal_prices / entry_price - 1

        # Get the VaR threshold
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)

        # CVaR is the mean of returns at or below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            # Edge case: no returns below threshold
            return var_threshold

        cvar = float(np.mean(tail_returns))

        return cvar

    def determine_path_count(
        self,
        pilot_results: dict,
        ci_width_threshold: float = 0.15,
        min_paths: int = 1000,
        max_paths: int = 50000,
    ) -> int:
        """Determine adaptive number of simulation paths based on pilot results.

        Uses the confidence interval width from a pilot simulation to determine
        if more paths are needed for accurate estimates.

        The algorithm:
        - If CI width <= threshold: return min_paths (estimates are precise enough)
        - If CI width > threshold: scale up paths proportionally, capped at max_paths

        Args:
            pilot_results: Dictionary containing pilot simulation results with keys:
                - 'mean': Mean of pilot simulation
                - 'std': Standard deviation of pilot simulation
                - 'ci_width': Relative CI width (as fraction of mean)
                - 'n_paths': Number of paths in pilot simulation
            ci_width_threshold: Maximum acceptable CI width as fraction.
                               Defaults to 0.15 (15%).
            min_paths: Minimum number of paths to return. Defaults to 1000.
            max_paths: Maximum number of paths to return. Defaults to 50000.

        Returns:
            Recommended number of simulation paths.

        Example:
            >>> engine = SimulationEngine()
            >>> pilot = {'mean': 100, 'std': 15, 'ci_width': 0.10, 'n_paths': 500}
            >>> engine.determine_path_count(pilot)  # Narrow CI
            1000
            >>> pilot['ci_width'] = 0.25  # Wide CI
            >>> engine.determine_path_count(pilot) > 1000
            True
        """
        ci_width = pilot_results.get("ci_width", 0.0)

        # If CI is tight enough, use minimum paths
        if ci_width <= ci_width_threshold:
            return min_paths

        # Scale up paths based on how much wider the CI is than threshold
        # CI width scales as 1/sqrt(n), so to reduce CI by factor k,
        # we need k^2 times more paths
        scale_factor = (ci_width / ci_width_threshold) ** 2
        pilot_n = pilot_results.get("n_paths", min_paths)

        # Calculate recommended paths, scaled from pilot
        recommended = int(pilot_n * scale_factor)

        # Clamp to valid range
        return max(min_paths, min(recommended, max_paths))
