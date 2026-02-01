"""Crisis data loader for Monte Carlo stress testing simulations.

This module provides access to historical crisis data (daily log returns) for use
in Monte Carlo simulations. The data is bundled as CSV files in the crisis_data
subdirectory.
"""

from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class CrisisScenario(Enum):
    """Enumeration of available crisis scenarios for stress testing."""

    CRISIS_2008 = "sp500_2008_crisis"
    DOTCOM_CRASH = "sp500_dotcom_crash"
    COVID_CRASH = "sp500_covid_crash"
    STAGFLATION_1970S = "sp500_stagflation_1970s"
    BLACK_MONDAY_1987 = "sp500_black_monday_1987"
    RISING_RATES_2022 = "sp500_rising_rates_2022"
    REGIME_DEMOCRAT = "regime_democrat_returns"
    REGIME_REPUBLICAN = "regime_republican_returns"


# Metadata for each crisis scenario
_SCENARIO_METADATA: dict[CrisisScenario, dict] = {
    CrisisScenario.CRISIS_2008: {
        "name": "2008 Financial Crisis",
        "start_date": "2008-09-01",
        "end_date": "2009-03-31",
        "description": (
            "Global financial crisis triggered by the collapse of Lehman Brothers "
            "and the subprime mortgage crisis. The S&P 500 fell approximately 57% "
            "from its October 2007 peak to March 2009 trough."
        ),
    },
    CrisisScenario.DOTCOM_CRASH: {
        "name": "Dot-com Crash",
        "start_date": "2000-03-01",
        "end_date": "2002-10-31",
        "description": (
            "The bursting of the technology bubble after the dot-com era. "
            "The NASDAQ fell nearly 80% from its peak, and the S&P 500 "
            "declined approximately 49%."
        ),
    },
    CrisisScenario.COVID_CRASH: {
        "name": "COVID-19 Crash",
        "start_date": "2020-02-01",
        "end_date": "2020-03-31",
        "description": (
            "Rapid market decline triggered by the global COVID-19 pandemic. "
            "The S&P 500 fell 34% in just 23 trading days, the fastest decline "
            "from an all-time high in history."
        ),
    },
    CrisisScenario.STAGFLATION_1970S: {
        "name": "1970s Stagflation",
        "start_date": "1973-01-01",
        "end_date": "1974-12-31",
        "description": (
            "Period of high inflation and economic stagnation following the "
            "1973 oil embargo. The S&P 500 fell approximately 48% during "
            "the 1973-1974 bear market."
        ),
    },
    CrisisScenario.BLACK_MONDAY_1987: {
        "name": "Black Monday 1987",
        "start_date": "1987-10-01",
        "end_date": "1987-10-31",
        "description": (
            "October 19, 1987 saw the largest single-day percentage decline "
            "in stock market history, with the Dow falling 22.6%. The crisis "
            "was exacerbated by program trading and portfolio insurance."
        ),
    },
    CrisisScenario.RISING_RATES_2022: {
        "name": "Rising Rates 2022",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "description": (
            "Federal Reserve's aggressive rate hiking cycle to combat inflation. "
            "The S&P 500 fell approximately 19% for the year as interest rates "
            "rose from near-zero to over 4%."
        ),
    },
    CrisisScenario.REGIME_DEMOCRAT: {
        "name": "Democrat Administration Returns",
        "start_date": "1993-01-20",
        "end_date": "2024-12-31",
        "description": (
            "Aggregate S&P 500 returns during Democratic presidential administrations: "
            "Clinton (1993-2001), Obama (2009-2017), and Biden (2021-present). "
            "Used for political regime analysis."
        ),
    },
    CrisisScenario.REGIME_REPUBLICAN: {
        "name": "Republican Administration Returns",
        "start_date": "1989-01-20",
        "end_date": "2021-01-19",
        "description": (
            "Aggregate S&P 500 returns during Republican presidential administrations: "
            "Bush Sr (1989-1993), Bush Jr (2001-2009), and Trump (2017-2021). "
            "Used for political regime analysis."
        ),
    },
}


class CrisisDataLoader:
    """Loader for crisis scenario data used in Monte Carlo stress testing.

    This class provides access to historical daily log returns for various
    crisis scenarios. Data is loaded from bundled CSV files and cached
    for performance.

    Attributes:
        data_dir: Path to the directory containing crisis data CSV files.

    Example:
        >>> loader = CrisisDataLoader()
        >>> returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)
        >>> returns.shape
        (144,)
        >>> returns.dtype
        dtype('float64')
    """

    # Default data directory relative to this module
    _DEFAULT_DATA_DIR: ClassVar[Path] = Path(__file__).parent / "crisis_data"

    def __init__(self, data_dir: Path | None = None):
        """Initialize the crisis data loader.

        Args:
            data_dir: Optional path to directory containing crisis CSV files.
                     Defaults to the bundled crisis_data directory.
        """
        self.data_dir = data_dir or self._DEFAULT_DATA_DIR

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Crisis data directory not found: {self.data_dir}. "
                "Run 'python scripts/fetch_crisis_data.py' to generate data files."
            )

        # Cache for loaded returns
        self._cache: dict[CrisisScenario, np.ndarray] = {}

    def load_crisis_returns(self, scenario: CrisisScenario) -> np.ndarray:
        """Load daily log returns for a crisis scenario.

        Args:
            scenario: The crisis scenario to load.

        Returns:
            NumPy array of daily log returns (float64).

        Raises:
            FileNotFoundError: If the CSV file for the scenario doesn't exist.
            ValueError: If the CSV file is empty or malformed.
        """
        # Check cache first
        if scenario in self._cache:
            return self._cache[scenario]

        # Load from CSV
        csv_path = self.data_dir / f"{scenario.value}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Crisis data file not found: {csv_path}. "
                f"Run 'python scripts/fetch_crisis_data.py' to generate data files."
            )

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {csv_path}: {e}") from e

        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")

        if "daily_return" not in df.columns:
            raise ValueError(
                f"CSV file missing 'daily_return' column: {csv_path}. "
                f"Found columns: {list(df.columns)}"
            )

        # Convert to numpy array with float64 dtype
        returns = df["daily_return"].to_numpy(dtype=np.float64)

        # Validate data
        if len(returns) == 0:
            raise ValueError(f"No return data found in {csv_path}")

        # Cache the result
        self._cache[scenario] = returns

        return returns

    def apply_beta_adjustment(
        self, base_returns: np.ndarray, beta: float
    ) -> np.ndarray:
        """Apply beta adjustment to market returns for a specific stock.

        For a stock with beta > 1, returns are amplified (more volatile).
        For a stock with beta < 1, returns are dampened (less volatile).

        The adjustment assumes the stock moves proportionally to the market:
            stock_return = beta * market_return

        Args:
            base_returns: Array of market (S&P 500) log returns.
            beta: The stock's beta coefficient relative to the market.
                 Beta of 1.0 means the stock moves with the market.
                 Beta of 1.5 means 50% more volatile than market.
                 Beta of 0.5 means 50% less volatile than market.

        Returns:
            Beta-adjusted returns as a new array.

        Example:
            >>> loader = CrisisDataLoader()
            >>> market_returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)
            >>> # For a stock with beta of 1.3 (30% more volatile than market)
            >>> stock_returns = loader.apply_beta_adjustment(market_returns, beta=1.3)
        """
        return beta * base_returns

    def get_scenario_metadata(self, scenario: CrisisScenario) -> dict:
        """Get metadata for a crisis scenario.

        Args:
            scenario: The crisis scenario to get metadata for.

        Returns:
            Dictionary containing:
                - name: Human-readable name of the scenario
                - start_date: Start date of the crisis period (YYYY-MM-DD)
                - end_date: End date of the crisis period (YYYY-MM-DD)
                - description: Detailed description of the crisis
        """
        return _SCENARIO_METADATA[scenario].copy()

    @staticmethod
    def get_all_scenarios() -> list[CrisisScenario]:
        """Get a list of all available crisis scenarios.

        Returns:
            List of all CrisisScenario enum values.
        """
        return list(CrisisScenario)

    def clear_cache(self) -> None:
        """Clear the internal cache of loaded returns."""
        self._cache.clear()

    def preload_all(self) -> None:
        """Preload all crisis scenarios into cache.

        This can be useful for performance when you know you'll need
        all scenarios, as it avoids repeated file I/O.
        """
        for scenario in CrisisScenario:
            self.load_crisis_returns(scenario)

    def get_combined_returns(
        self, scenarios: list[CrisisScenario]
    ) -> np.ndarray:
        """Combine returns from multiple scenarios into a single array.

        This is useful for creating a combined crisis distribution
        from multiple historical periods.

        Args:
            scenarios: List of scenarios to combine.

        Returns:
            Concatenated array of returns from all specified scenarios.
        """
        all_returns = []
        for scenario in scenarios:
            returns = self.load_crisis_returns(scenario)
            all_returns.append(returns)
        return np.concatenate(all_returns)
