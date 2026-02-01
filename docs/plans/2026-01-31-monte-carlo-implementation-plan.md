# Monte Carlo Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Monte Carlo simulation for risk analysis of high-scoring stock candidates (≥80 composite score), integrated with research reports.

**Architecture:** Hybrid approach using GBM for base case projections and block bootstrap for 8 historical stress scenarios. Results stored in SQLite and included in research reports sent via notification channels.

**Tech Stack:** NumPy/SciPy for simulation, yfinance for historical data fetching, SQLAlchemy for persistence, Pydantic for runtime models, Typer for CLI.

---

## Task 1: Create Pydantic Models for Simulation Results

**Files:**
- Create: `src/investment_monitor/simulation/__init__.py`
- Create: `src/investment_monitor/simulation/models.py`
- Test: `tests/test_simulation/test_models.py`

**Step 1: Create the simulation package**

Create `src/investment_monitor/simulation/__init__.py`:
```python
"""Monte Carlo simulation module for risk analysis."""

from .models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationConfig,
    SimulationOutput,
)

__all__ = [
    "HorizonResult",
    "ScenarioResult",
    "SensitivityResult",
    "SimulationConfig",
    "SimulationOutput",
]
```

**Step 2: Write the failing test**

Create `tests/test_simulation/__init__.py` (empty file).

Create `tests/test_simulation/test_models.py`:
```python
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
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'investment_monitor.simulation'"

**Step 4: Write minimal implementation**

Create `src/investment_monitor/simulation/models.py`:
```python
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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_models.py -v`
Expected: PASS (all tests green)

**Step 6: Commit**

```bash
git add src/investment_monitor/simulation/__init__.py src/investment_monitor/simulation/models.py tests/test_simulation/__init__.py tests/test_simulation/test_models.py
git commit -m "feat(simulation): add Pydantic models for Monte Carlo simulation"
```

---

## Task 2: Create SQLAlchemy Model for Simulation Persistence

**Files:**
- Modify: `src/investment_monitor/storage/research_models.py`
- Test: `tests/test_simulation/test_storage.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_storage.py`:
```python
"""Tests for simulation result storage."""

import json
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from investment_monitor.storage.database import Base
from investment_monitor.storage.research_models import SimulationResult


@pytest.fixture
def session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestSimulationResultModel:
    """Tests for SimulationResult ORM model."""

    def test_create_simulation_result(self, session: Session):
        result = SimulationResult(
            ticker="AAPL",
            run_date=date(2026, 1, 31),
            entry_price=178.50,
            composite_score=85.0,
            num_simulations=10000,
            lookback_days=756,
            volatility=0.25,
            drift=0.08,
            results_30d={"base_mean": 182.0, "scenarios": {}},
            results_90d={"base_mean": 189.0, "scenarios": {}},
            results_252d={"base_mean": 198.0, "scenarios": {}},
            sensitivity_analysis={"primary_driver": "volatility"},
        )
        session.add(result)
        session.commit()

        retrieved = session.query(SimulationResult).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.ticker == "AAPL"
        assert retrieved.entry_price == 178.50
        assert retrieved.composite_score == 85.0
        assert retrieved.results_30d["base_mean"] == 182.0

    def test_simulation_result_auto_timestamps(self, session: Session):
        result = SimulationResult(
            ticker="MSFT",
            run_date=date(2026, 1, 31),
            entry_price=400.00,
            composite_score=82.0,
            num_simulations=5000,
            lookback_days=504,
            volatility=0.22,
            drift=0.10,
            results_30d={},
            results_90d={},
            results_252d={},
            sensitivity_analysis={},
        )
        session.add(result)
        session.commit()

        assert result.created_at is not None
        assert isinstance(result.created_at, datetime)

    def test_multiple_simulations_for_ticker(self, session: Session):
        """A ticker can have multiple simulation runs over time."""
        for i in range(3):
            result = SimulationResult(
                ticker="GOOGL",
                run_date=date(2026, 1, 31 - i),
                entry_price=150.0 + i,
                composite_score=80.0 + i,
                num_simulations=1000,
                lookback_days=252,
                volatility=0.20,
                drift=0.05,
                results_30d={},
                results_90d={},
                results_252d={},
                sensitivity_analysis={},
            )
            session.add(result)
        session.commit()

        results = session.query(SimulationResult).filter_by(ticker="GOOGL").all()
        assert len(results) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_storage.py -v`
Expected: FAIL with "ImportError: cannot import name 'SimulationResult'"

**Step 3: Write minimal implementation**

Modify `src/investment_monitor/storage/research_models.py`, add after existing models:
```python
class SimulationResult(Base):
    """Stores Monte Carlo simulation results for a ticker."""

    __tablename__ = "simulation_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    run_date: Mapped[date] = mapped_column(Date, index=True)
    entry_price: Mapped[float] = mapped_column(Float)
    composite_score: Mapped[float] = mapped_column(Float)

    # Parameters used
    num_simulations: Mapped[int] = mapped_column(Integer)
    lookback_days: Mapped[int] = mapped_column(Integer)
    volatility: Mapped[float] = mapped_column(Float)
    drift: Mapped[float] = mapped_column(Float)

    # Results stored as JSON blobs per horizon
    results_30d: Mapped[dict] = mapped_column(JSON, default=dict)
    results_90d: Mapped[dict] = mapped_column(JSON, default=dict)
    results_252d: Mapped[dict] = mapped_column(JSON, default=dict)

    # Sensitivity analysis summary
    sensitivity_analysis: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
```

Also add the necessary imports at the top of the file if not present:
```python
from sqlalchemy import JSON
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_storage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/investment_monitor/storage/research_models.py tests/test_simulation/test_storage.py
git commit -m "feat(simulation): add SimulationResult SQLAlchemy model"
```

---

## Task 3: Create Crisis Data Loader with Bundled CSV Files

**Files:**
- Create: `src/investment_monitor/simulation/crisis_loader.py`
- Create: `src/investment_monitor/simulation/crisis_data/` directory and CSV files
- Test: `tests/test_simulation/test_crisis_loader.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_crisis_loader.py`:
```python
"""Tests for crisis data loader."""

import numpy as np
import pytest

from investment_monitor.simulation.crisis_loader import CrisisDataLoader, CrisisScenario


class TestCrisisScenario:
    """Tests for CrisisScenario enum."""

    def test_all_scenarios_defined(self):
        scenarios = list(CrisisScenario)
        assert len(scenarios) == 8
        assert CrisisScenario.CRISIS_2008 in scenarios
        assert CrisisScenario.DOTCOM_CRASH in scenarios
        assert CrisisScenario.COVID_CRASH in scenarios
        assert CrisisScenario.STAGFLATION_1970S in scenarios
        assert CrisisScenario.BLACK_MONDAY_1987 in scenarios
        assert CrisisScenario.RISING_RATES_2022 in scenarios
        assert CrisisScenario.REGIME_DEMOCRAT in scenarios
        assert CrisisScenario.REGIME_REPUBLICAN in scenarios


class TestCrisisDataLoader:
    """Tests for CrisisDataLoader."""

    def test_load_crisis_returns(self):
        loader = CrisisDataLoader()
        returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)

        assert isinstance(returns, np.ndarray)
        assert len(returns) > 0
        assert returns.dtype == np.float64

    def test_all_crisis_data_loads(self):
        loader = CrisisDataLoader()
        for scenario in CrisisScenario:
            returns = loader.load_crisis_returns(scenario)
            assert len(returns) > 5, f"{scenario.name} has insufficient data"

    def test_apply_beta_adjustment(self):
        loader = CrisisDataLoader()
        base_returns = np.array([0.01, -0.02, 0.015, -0.01])
        beta = 1.5

        adjusted = loader.apply_beta_adjustment(base_returns, beta)

        assert len(adjusted) == len(base_returns)
        np.testing.assert_array_almost_equal(adjusted, base_returns * beta)

    def test_get_scenario_metadata(self):
        loader = CrisisDataLoader()
        metadata = loader.get_scenario_metadata(CrisisScenario.CRISIS_2008)

        assert "name" in metadata
        assert "start_date" in metadata
        assert "end_date" in metadata
        assert "description" in metadata
        assert metadata["name"] == "2008 Financial Crisis"

    def test_crisis_returns_are_log_returns(self):
        """Verify data is stored as log returns for simulation compatibility."""
        loader = CrisisDataLoader()
        returns = loader.load_crisis_returns(CrisisScenario.COVID_CRASH)

        # Log returns should typically be in range [-0.15, 0.15] for daily
        assert np.all(returns > -0.5)
        assert np.all(returns < 0.5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_crisis_loader.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create crisis data directory and CSV files**

First, we need to fetch real historical data from Yahoo Finance. Create a script to generate the CSVs.

Create `src/investment_monitor/simulation/crisis_data/README.md`:
```markdown
# Crisis Data for Monte Carlo Simulations

This directory contains historical S&P 500 daily returns during major market stress periods.

## Data Sources
- Source: Yahoo Finance (^GSPC)
- Format: Daily log returns
- Generated: 2026-01-31

## Scenarios

| File | Period | Description |
|------|--------|-------------|
| sp500_2008_crisis.csv | Sep 2008 - Mar 2009 | Global Financial Crisis |
| sp500_dotcom_crash.csv | Mar 2000 - Oct 2002 | Tech Bubble Burst |
| sp500_covid_crash.csv | Feb - Mar 2020 | COVID-19 Pandemic |
| sp500_stagflation_1970s.csv | Jan 1973 - Dec 1974 | Oil Crisis & Stagflation |
| sp500_black_monday_1987.csv | Oct 1987 | Black Monday Crash |
| sp500_rising_rates_2022.csv | Jan - Dec 2022 | Fed Rate Hikes |
| regime_democrat_returns.csv | Various | Aggregated Democrat admin returns |
| regime_republican_returns.csv | Various | Aggregated Republican admin returns |

## CSV Format
```csv
date,daily_return
2008-09-15,-0.0479
```

## Usage
Returns are log returns: `ln(P_t / P_{t-1})`

To convert to simple returns: `exp(log_return) - 1`

## Beta Adjustment
Individual stock stress returns are derived by:
`stock_return = index_return * stock_beta`
```

**Step 4: Write the crisis loader implementation**

Create `src/investment_monitor/simulation/crisis_loader.py`:
```python
"""Load pre-bundled crisis period return data for stress testing."""

import csv
from enum import Enum
from pathlib import Path

import numpy as np


class CrisisScenario(Enum):
    """Available crisis scenarios for stress testing."""

    CRISIS_2008 = "sp500_2008_crisis"
    DOTCOM_CRASH = "sp500_dotcom_crash"
    COVID_CRASH = "sp500_covid_crash"
    STAGFLATION_1970S = "sp500_stagflation_1970s"
    BLACK_MONDAY_1987 = "sp500_black_monday_1987"
    RISING_RATES_2022 = "sp500_rising_rates_2022"
    REGIME_DEMOCRAT = "regime_democrat_returns"
    REGIME_REPUBLICAN = "regime_republican_returns"


SCENARIO_METADATA = {
    CrisisScenario.CRISIS_2008: {
        "name": "2008 Financial Crisis",
        "start_date": "2008-09-01",
        "end_date": "2009-03-31",
        "description": "Global financial crisis triggered by subprime mortgage collapse",
    },
    CrisisScenario.DOTCOM_CRASH: {
        "name": "Dot-com Crash",
        "start_date": "2000-03-01",
        "end_date": "2002-10-31",
        "description": "Technology bubble burst",
    },
    CrisisScenario.COVID_CRASH: {
        "name": "COVID-19 Crash",
        "start_date": "2020-02-01",
        "end_date": "2020-03-31",
        "description": "Pandemic-induced market crash",
    },
    CrisisScenario.STAGFLATION_1970S: {
        "name": "1970s Stagflation",
        "start_date": "1973-01-01",
        "end_date": "1974-12-31",
        "description": "Oil crisis and stagflation period",
    },
    CrisisScenario.BLACK_MONDAY_1987: {
        "name": "Black Monday 1987",
        "start_date": "1987-10-01",
        "end_date": "1987-10-31",
        "description": "Single-day market crash of 22%",
    },
    CrisisScenario.RISING_RATES_2022: {
        "name": "Rising Rates 2022",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "description": "Federal Reserve aggressive rate hikes",
    },
    CrisisScenario.REGIME_DEMOCRAT: {
        "name": "Democrat Administration",
        "start_date": "Various",
        "end_date": "Various",
        "description": "Aggregated returns during Democratic presidencies",
    },
    CrisisScenario.REGIME_REPUBLICAN: {
        "name": "Republican Administration",
        "start_date": "Various",
        "end_date": "Various",
        "description": "Aggregated returns during Republican presidencies",
    },
}


class CrisisDataLoader:
    """Load and manage crisis period return data."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize with optional custom data directory."""
        if data_dir is None:
            self._data_dir = Path(__file__).parent / "crisis_data"
        else:
            self._data_dir = data_dir
        self._cache: dict[CrisisScenario, np.ndarray] = {}

    def load_crisis_returns(self, scenario: CrisisScenario) -> np.ndarray:
        """
        Load daily log returns for a crisis scenario.

        Args:
            scenario: The crisis scenario to load

        Returns:
            NumPy array of daily log returns
        """
        if scenario in self._cache:
            return self._cache[scenario]

        csv_path = self._data_dir / f"{scenario.value}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Crisis data not found: {csv_path}")

        returns = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                returns.append(float(row["daily_return"]))

        result = np.array(returns, dtype=np.float64)
        self._cache[scenario] = result
        return result

    def apply_beta_adjustment(
        self, base_returns: np.ndarray, beta: float
    ) -> np.ndarray:
        """
        Adjust index returns for individual stock beta.

        Args:
            base_returns: S&P 500 log returns
            beta: Stock's beta relative to S&P 500

        Returns:
            Beta-adjusted returns
        """
        return base_returns * beta

    def get_scenario_metadata(self, scenario: CrisisScenario) -> dict:
        """Get metadata about a crisis scenario."""
        return SCENARIO_METADATA[scenario]

    def get_all_scenarios(self) -> list[CrisisScenario]:
        """Get list of all available scenarios."""
        return list(CrisisScenario)
```

**Step 5: Create a script to fetch and generate the CSV files**

Create `scripts/fetch_crisis_data.py` (one-time script to generate CSVs):
```python
#!/usr/bin/env python3
"""Fetch historical crisis data from Yahoo Finance and save as CSV."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf

OUTPUT_DIR = Path("src/investment_monitor/simulation/crisis_data")

CRISIS_PERIODS = {
    "sp500_2008_crisis": ("2008-09-01", "2009-03-31"),
    "sp500_dotcom_crash": ("2000-03-01", "2002-10-31"),
    "sp500_covid_crash": ("2020-02-01", "2020-03-31"),
    "sp500_stagflation_1970s": ("1973-01-01", "1974-12-31"),
    "sp500_black_monday_1987": ("1987-10-01", "1987-10-31"),
    "sp500_rising_rates_2022": ("2022-01-01", "2022-12-31"),
}

# Political regime periods (simplified - major periods)
DEMOCRAT_PERIODS = [
    ("1993-01-20", "2001-01-19"),  # Clinton
    ("2009-01-20", "2017-01-19"),  # Obama
    ("2021-01-20", "2025-01-19"),  # Biden
]

REPUBLICAN_PERIODS = [
    ("2001-01-20", "2009-01-19"),  # Bush
    ("2017-01-20", "2021-01-19"),  # Trump
]


def fetch_sp500_returns(start: str, end: str) -> list[tuple[str, float]]:
    """Fetch S&P 500 daily log returns for a period."""
    ticker = yf.Ticker("^GSPC")
    data = ticker.history(start=start, end=end)

    if data.empty:
        print(f"  Warning: No data for {start} to {end}")
        return []

    # Calculate log returns
    prices = data["Close"].values
    log_returns = np.diff(np.log(prices))
    dates = data.index[1:].strftime("%Y-%m-%d").tolist()

    return list(zip(dates, log_returns))


def save_csv(filename: str, data: list[tuple[str, float]]):
    """Save returns data to CSV."""
    output_path = OUTPUT_DIR / f"{filename}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "daily_return"])
        for date, ret in data:
            writer.writerow([date, f"{ret:.6f}"])
    print(f"  Saved {len(data)} rows to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching crisis period data...")
    for name, (start, end) in CRISIS_PERIODS.items():
        print(f"  {name}: {start} to {end}")
        data = fetch_sp500_returns(start, end)
        if data:
            save_csv(name, data)

    print("\nFetching political regime data...")

    # Democrat periods
    print("  Democrat periods...")
    dem_data = []
    for start, end in DEMOCRAT_PERIODS:
        dem_data.extend(fetch_sp500_returns(start, end))
    if dem_data:
        save_csv("regime_democrat_returns", dem_data)

    # Republican periods
    print("  Republican periods...")
    rep_data = []
    for start, end in REPUBLICAN_PERIODS:
        rep_data.extend(fetch_sp500_returns(start, end))
    if rep_data:
        save_csv("regime_republican_returns", rep_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
```

**Step 6: Run the script to generate CSV files**

Run: `python scripts/fetch_crisis_data.py`

**Step 7: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_crisis_loader.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/investment_monitor/simulation/crisis_loader.py src/investment_monitor/simulation/crisis_data/ scripts/fetch_crisis_data.py tests/test_simulation/test_crisis_loader.py
git commit -m "feat(simulation): add crisis data loader with historical S&P 500 returns"
```

---

## Task 4: Implement Core GBM Simulation Engine

**Files:**
- Create: `src/investment_monitor/simulation/engine.py`
- Test: `tests/test_simulation/test_engine.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_engine.py`:
```python
"""Tests for Monte Carlo simulation engine."""

import numpy as np
import pytest

from investment_monitor.simulation.engine import SimulationEngine


class TestSimulationEngine:
    """Tests for the core simulation engine."""

    @pytest.fixture
    def engine(self):
        return SimulationEngine(seed=42)

    def test_simulate_gbm_returns_correct_shape(self, engine):
        result = engine.simulate_gbm(
            S0=100.0,
            mu=0.08,
            sigma=0.20,
            days=30,
            n_paths=1000,
        )
        assert result.shape == (1000,)

    def test_simulate_gbm_positive_prices(self, engine):
        result = engine.simulate_gbm(
            S0=100.0,
            mu=0.08,
            sigma=0.20,
            days=252,
            n_paths=5000,
        )
        assert np.all(result > 0)

    def test_simulate_gbm_mean_near_expected(self, engine):
        """Over many paths, mean should approach theoretical expectation."""
        S0 = 100.0
        mu = 0.08
        days = 252

        result = engine.simulate_gbm(
            S0=S0,
            mu=mu,
            sigma=0.20,
            days=days,
            n_paths=50000,
        )

        # Expected terminal value: S0 * exp(mu * T)
        expected_mean = S0 * np.exp(mu * days / 252)
        actual_mean = np.mean(result)

        # Allow 5% tolerance due to stochastic nature
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05

    def test_simulate_gbm_reproducible_with_seed(self):
        engine1 = SimulationEngine(seed=123)
        engine2 = SimulationEngine(seed=123)

        result1 = engine1.simulate_gbm(S0=100.0, mu=0.08, sigma=0.20, days=30, n_paths=100)
        result2 = engine2.simulate_gbm(S0=100.0, mu=0.08, sigma=0.20, days=30, n_paths=100)

        np.testing.assert_array_equal(result1, result2)

    def test_simulate_bootstrap_returns_correct_shape(self, engine):
        crisis_returns = np.random.normal(-0.001, 0.02, 100)

        result = engine.simulate_bootstrap(
            S0=100.0,
            crisis_returns=crisis_returns,
            days=30,
            n_paths=1000,
            block_size=5,
        )

        assert result.shape == (1000,)

    def test_simulate_bootstrap_preserves_return_distribution(self, engine):
        """Bootstrap should sample from actual crisis returns."""
        crisis_returns = np.array([-0.05, -0.03, -0.02, -0.01, 0.01, 0.02])

        result = engine.simulate_bootstrap(
            S0=100.0,
            crisis_returns=crisis_returns,
            days=30,
            n_paths=10000,
            block_size=3,
        )

        # During a crisis with mostly negative returns, mean should be below starting price
        assert np.mean(result) < 100.0

    def test_calculate_var(self, engine):
        prices = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
        entry_price = 100.0

        var_95 = engine.calculate_var(prices, entry_price, confidence=0.95)

        # 5th percentile price is 80, so VaR = (80 - 100) / 100 = -0.20
        assert var_95 == pytest.approx(-0.20, rel=0.01)

    def test_calculate_cvar(self, engine):
        prices = np.array([70, 75, 80, 85, 90, 95, 100, 105, 110, 115])
        entry_price = 100.0

        cvar_95 = engine.calculate_cvar(prices, entry_price, confidence=0.95)

        # CVaR is mean of returns below VaR threshold
        # 5th percentile is 70, CVaR = mean of returns for prices <= 70
        assert cvar_95 < engine.calculate_var(prices, entry_price, confidence=0.95)

    def test_determine_path_count_narrow_ci(self, engine):
        # Tight distribution -> fewer paths needed
        pilot_results = np.random.normal(100, 5, 1000)

        n_paths = engine.determine_path_count(pilot_results, ci_width_threshold=0.15)

        assert n_paths == 1000  # Minimum paths

    def test_determine_path_count_wide_ci(self, engine):
        # Wide distribution -> more paths needed
        pilot_results = np.random.normal(100, 30, 1000)

        n_paths = engine.determine_path_count(pilot_results, ci_width_threshold=0.15)

        assert n_paths > 1000  # Should request more paths
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `src/investment_monitor/simulation/engine.py`:
```python
"""Core Monte Carlo simulation engine."""

import numpy as np
from numpy.random import Generator, default_rng


class SimulationEngine:
    """Core engine for running Monte Carlo simulations."""

    def __init__(self, seed: int | None = None):
        """Initialize with optional random seed for reproducibility."""
        self._rng: Generator = default_rng(seed)

    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> np.ndarray:
        """
        Simulate terminal prices using Geometric Brownian Motion.

        dS = μSdt + σSdW

        Args:
            S0: Starting price
            mu: Annualized drift (expected return)
            sigma: Annualized volatility
            days: Time horizon in trading days
            n_paths: Number of simulation paths

        Returns:
            Array of terminal prices with shape (n_paths,)
        """
        dt = 1 / 252  # Daily timestep (trading days per year)

        # Generate random standard normal values
        Z = self._rng.standard_normal((n_paths, days))

        # Calculate drift and diffusion components
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        # Calculate log returns and cumulative sum
        log_returns = drift + diffusion
        cumulative_log_returns = np.sum(log_returns, axis=1)

        # Convert to terminal prices
        terminal_prices = S0 * np.exp(cumulative_log_returns)

        return terminal_prices

    def simulate_bootstrap(
        self,
        S0: float,
        crisis_returns: np.ndarray,
        days: int,
        n_paths: int,
        block_size: int = 5,
    ) -> np.ndarray:
        """
        Simulate terminal prices using block bootstrap from crisis period returns.

        Args:
            S0: Starting price
            crisis_returns: Historical daily log returns from crisis period
            days: Time horizon in trading days
            n_paths: Number of simulation paths
            block_size: Size of blocks to preserve autocorrelation

        Returns:
            Array of terminal prices with shape (n_paths,)
        """
        n_returns = len(crisis_returns)
        if n_returns < block_size:
            block_size = max(1, n_returns)

        n_blocks = (days // block_size) + 1
        max_start = n_returns - block_size

        if max_start <= 0:
            # If crisis period is very short, sample with replacement
            sampled_returns = self._rng.choice(crisis_returns, size=(n_paths, days))
        else:
            # Block bootstrap
            paths = np.zeros((n_paths, days))
            for i in range(n_paths):
                starts = self._rng.integers(0, max_start + 1, size=n_blocks)
                sampled = np.concatenate(
                    [crisis_returns[s : s + block_size] for s in starts]
                )
                paths[i] = sampled[:days]
            sampled_returns = paths

        # Calculate terminal prices from cumulative log returns
        cumulative_log_returns = np.sum(sampled_returns, axis=1)
        terminal_prices = S0 * np.exp(cumulative_log_returns)

        return terminal_prices

    def calculate_var(
        self,
        terminal_prices: np.ndarray,
        entry_price: float,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk as a return.

        Args:
            terminal_prices: Array of simulated terminal prices
            entry_price: Starting price
            confidence: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            VaR as a return (negative for losses)
        """
        returns = (terminal_prices - entry_price) / entry_price
        percentile = (1 - confidence) * 100  # e.g., 5th percentile for 95% VaR
        var = np.percentile(returns, percentile)
        return float(var)

    def calculate_cvar(
        self,
        terminal_prices: np.ndarray,
        entry_price: float,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        CVaR is the expected return given that returns are below VaR.

        Args:
            terminal_prices: Array of simulated terminal prices
            entry_price: Starting price
            confidence: Confidence level

        Returns:
            CVaR as a return (negative for losses)
        """
        returns = (terminal_prices - entry_price) / entry_price
        var = self.calculate_var(terminal_prices, entry_price, confidence)

        # Mean of returns below or equal to VaR
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var

        return float(np.mean(tail_returns))

    def determine_path_count(
        self,
        pilot_results: np.ndarray,
        ci_width_threshold: float = 0.15,
        min_paths: int = 1000,
        max_paths: int = 50000,
    ) -> int:
        """
        Determine optimal number of paths based on CI width.

        Args:
            pilot_results: Results from pilot simulation (e.g., 1000 paths)
            ci_width_threshold: Target relative CI width
            min_paths: Minimum paths to run
            max_paths: Maximum paths to run

        Returns:
            Recommended number of paths
        """
        ci_high = np.percentile(pilot_results, 97.5)
        ci_low = np.percentile(pilot_results, 2.5)
        ci_width = ci_high - ci_low
        mean = np.mean(pilot_results)

        if mean == 0:
            return min_paths

        relative_width = ci_width / mean

        if relative_width < ci_width_threshold:
            return min_paths
        elif relative_width < ci_width_threshold * 1.67:  # ~0.25
            return min(10000, max_paths)
        else:
            return max_paths
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_engine.py -v`
Expected: PASS

**Step 5: Update the package __init__.py**

Update `src/investment_monitor/simulation/__init__.py`:
```python
"""Monte Carlo simulation module for risk analysis."""

from .crisis_loader import CrisisDataLoader, CrisisScenario
from .engine import SimulationEngine
from .models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationConfig,
    SimulationOutput,
)

__all__ = [
    "CrisisDataLoader",
    "CrisisScenario",
    "HorizonResult",
    "ScenarioResult",
    "SensitivityResult",
    "SimulationConfig",
    "SimulationEngine",
    "SimulationOutput",
]
```

**Step 6: Commit**

```bash
git add src/investment_monitor/simulation/engine.py src/investment_monitor/simulation/__init__.py tests/test_simulation/test_engine.py
git commit -m "feat(simulation): implement core GBM and bootstrap simulation engine"
```

---

## Task 5: Implement Sensitivity Analyzer

**Files:**
- Create: `src/investment_monitor/simulation/sensitivity.py`
- Test: `tests/test_simulation/test_sensitivity.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_sensitivity.py`:
```python
"""Tests for sensitivity analysis."""

import numpy as np
import pytest

from investment_monitor.simulation.engine import SimulationEngine
from investment_monitor.simulation.models import SensitivityResult
from investment_monitor.simulation.sensitivity import SensitivityAnalyzer


class TestSensitivityAnalyzer:
    """Tests for the sensitivity analyzer."""

    @pytest.fixture
    def analyzer(self):
        engine = SimulationEngine(seed=42)
        return SensitivityAnalyzer(engine)

    def test_analyze_volatility_sensitivity(self, analyzer):
        result = analyzer.analyze_volatility_sensitivity(
            S0=100.0,
            mu=0.08,
            base_sigma=0.25,
            days=252,
            n_paths=1000,
            multipliers=[0.5, 1.0, 1.5],
        )

        assert isinstance(result, dict)
        assert 0.5 in result
        assert 1.0 in result
        assert 1.5 in result
        # Higher volatility should lead to wider range (but mean stays similar)
        # Lower vol multiplier should give tighter distribution

    def test_analyze_drift_sensitivity(self, analyzer):
        result = analyzer.analyze_drift_sensitivity(
            S0=100.0,
            base_mu=0.08,
            sigma=0.25,
            days=252,
            n_paths=1000,
        )

        assert isinstance(result, dict)
        assert "pessimistic" in result
        assert "neutral" in result
        assert "optimistic" in result
        # Optimistic drift should give higher mean
        assert result["optimistic"] > result["neutral"]
        assert result["neutral"] > result["pessimistic"]

    def test_analyze_lookback_sensitivity(self, analyzer):
        # Create mock historical data with different characteristics
        short_vol = 0.20  # Lower volatility in short period
        long_vol = 0.30   # Higher volatility in longer period

        result = analyzer.analyze_lookback_sensitivity(
            S0=100.0,
            mu=0.08,
            days=252,
            n_paths=1000,
            lookback_volatilities={252: short_vol, 756: long_vol, 1260: long_vol},
        )

        assert isinstance(result, dict)
        assert 252 in result
        assert 756 in result
        assert 1260 in result

    def test_calculate_impact_scores(self, analyzer):
        volatility_range = {0.5: 120.0, 1.0: 100.0, 1.5: 80.0}
        drift_range = {"pessimistic": 90.0, "neutral": 100.0, "optimistic": 110.0}
        lookback_range = {252: 98.0, 756: 100.0, 1260: 102.0}

        vol_impact, drift_impact, lookback_impact = analyzer.calculate_impact_scores(
            volatility_range, drift_range, lookback_range
        )

        # Volatility has 40pt range (120-80), drift has 20pt (110-90), lookback has 4pt
        assert vol_impact > drift_impact
        assert drift_impact > lookback_impact
        assert 0 <= vol_impact <= 100
        assert 0 <= drift_impact <= 100
        assert 0 <= lookback_impact <= 100

    def test_run_full_sensitivity_analysis(self, analyzer):
        result = analyzer.run_analysis(
            S0=100.0,
            mu=0.08,
            sigma=0.25,
            days=252,
            n_paths=1000,
            lookback_volatilities={252: 0.25, 756: 0.25, 1260: 0.25},
        )

        assert isinstance(result, SensitivityResult)
        assert result.primary_driver in ["volatility", "drift", "lookback"]
        assert 0 <= result.volatility_impact <= 100
        assert 0 <= result.drift_impact <= 100
        assert 0 <= result.lookback_impact <= 100
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_sensitivity.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `src/investment_monitor/simulation/sensitivity.py`:
```python
"""Sensitivity analysis for Monte Carlo simulations."""

import numpy as np

from .engine import SimulationEngine
from .models import SensitivityResult


class SensitivityAnalyzer:
    """Analyze sensitivity of simulation results to input assumptions."""

    def __init__(self, engine: SimulationEngine):
        """Initialize with a simulation engine."""
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
        """
        Analyze how volatility assumptions affect results.

        Args:
            S0: Starting price
            mu: Drift rate
            base_sigma: Base volatility
            days: Time horizon
            n_paths: Number of paths per scenario
            multipliers: Volatility multipliers to test

        Returns:
            Dict mapping multiplier -> mean terminal price
        """
        if multipliers is None:
            multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]

        results = {}
        for mult in multipliers:
            sigma = base_sigma * mult
            prices = self._engine.simulate_gbm(S0, mu, sigma, days, n_paths)
            results[mult] = float(np.mean(prices))

        return results

    def analyze_drift_sensitivity(
        self,
        S0: float,
        base_mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> dict[str, float]:
        """
        Analyze how drift assumptions affect results.

        Args:
            S0: Starting price
            base_mu: Historical drift rate
            sigma: Volatility
            days: Time horizon
            n_paths: Number of paths per scenario

        Returns:
            Dict mapping scenario name -> mean terminal price
        """
        drift_scenarios = {
            "pessimistic": 0.0,  # Zero expected return
            "neutral": base_mu,  # Historical rate
            "optimistic": base_mu + 0.02,  # Historical + 2%
        }

        results = {}
        for name, mu in drift_scenarios.items():
            prices = self._engine.simulate_gbm(S0, mu, sigma, days, n_paths)
            results[name] = float(np.mean(prices))

        return results

    def analyze_lookback_sensitivity(
        self,
        S0: float,
        mu: float,
        days: int,
        n_paths: int,
        lookback_volatilities: dict[int, float],
    ) -> dict[int, float]:
        """
        Analyze how lookback period affects results.

        Args:
            S0: Starting price
            mu: Drift rate
            days: Time horizon
            n_paths: Number of paths per scenario
            lookback_volatilities: Dict mapping lookback days -> calculated volatility

        Returns:
            Dict mapping lookback days -> mean terminal price
        """
        results = {}
        for lookback_days, sigma in lookback_volatilities.items():
            prices = self._engine.simulate_gbm(S0, mu, sigma, days, n_paths)
            results[lookback_days] = float(np.mean(prices))

        return results

    def calculate_impact_scores(
        self,
        volatility_range: dict[float, float],
        drift_range: dict[str, float],
        lookback_range: dict[int, float],
    ) -> tuple[float, float, float]:
        """
        Calculate normalized impact scores (0-100) for each input.

        Args:
            volatility_range: Results from volatility sensitivity
            drift_range: Results from drift sensitivity
            lookback_range: Results from lookback sensitivity

        Returns:
            Tuple of (volatility_impact, drift_impact, lookback_impact)
        """
        # Calculate ranges
        vol_values = list(volatility_range.values())
        drift_values = list(drift_range.values())
        lookback_values = list(lookback_range.values())

        vol_range = max(vol_values) - min(vol_values) if vol_values else 0
        drift_range_val = max(drift_values) - min(drift_values) if drift_values else 0
        lookback_range_val = (
            max(lookback_values) - min(lookback_values) if lookback_values else 0
        )

        # Normalize to 0-100 scale
        total_range = vol_range + drift_range_val + lookback_range_val
        if total_range == 0:
            return 33.3, 33.3, 33.3

        vol_impact = (vol_range / total_range) * 100
        drift_impact = (drift_range_val / total_range) * 100
        lookback_impact = (lookback_range_val / total_range) * 100

        return vol_impact, drift_impact, lookback_impact

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
        """
        Run complete sensitivity analysis.

        Args:
            S0: Starting price
            mu: Drift rate
            sigma: Base volatility
            days: Time horizon (use longest horizon, e.g., 252)
            n_paths: Number of paths per scenario
            lookback_volatilities: Dict mapping lookback period -> volatility
            volatility_multipliers: Multipliers for volatility sensitivity

        Returns:
            Complete sensitivity analysis results
        """
        # Run each sensitivity analysis
        vol_range = self.analyze_volatility_sensitivity(
            S0, mu, sigma, days, n_paths, volatility_multipliers
        )
        drift_range = self.analyze_drift_sensitivity(S0, mu, sigma, days, n_paths)
        lookback_range = self.analyze_lookback_sensitivity(
            S0, mu, days, n_paths, lookback_volatilities
        )

        # Calculate impact scores
        vol_impact, drift_impact, lookback_impact = self.calculate_impact_scores(
            vol_range, drift_range, lookback_range
        )

        # Determine primary driver
        impacts = {
            "volatility": vol_impact,
            "drift": drift_impact,
            "lookback": lookback_impact,
        }
        primary_driver = max(impacts, key=impacts.get)

        return SensitivityResult(
            volatility_impact=vol_impact,
            drift_impact=drift_impact,
            lookback_impact=lookback_impact,
            primary_driver=primary_driver,
            volatility_range=vol_range,
            drift_range=drift_range,
            lookback_range=lookback_range,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_sensitivity.py -v`
Expected: PASS

**Step 5: Update package __init__.py**

Update `src/investment_monitor/simulation/__init__.py` to add:
```python
from .sensitivity import SensitivityAnalyzer
```

And add `"SensitivityAnalyzer"` to `__all__`.

**Step 6: Commit**

```bash
git add src/investment_monitor/simulation/sensitivity.py src/investment_monitor/simulation/__init__.py tests/test_simulation/test_sensitivity.py
git commit -m "feat(simulation): implement sensitivity analyzer"
```

---

## Task 6: Implement Main MonteCarloAnalyzer Orchestrator

**Files:**
- Create: `src/investment_monitor/simulation/analyzer.py`
- Test: `tests/test_simulation/test_analyzer.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_analyzer.py`:
```python
"""Tests for Monte Carlo analyzer orchestrator."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from investment_monitor.simulation.analyzer import MonteCarloAnalyzer
from investment_monitor.simulation.crisis_loader import CrisisScenario
from investment_monitor.simulation.models import (
    HorizonResult,
    SimulationConfig,
    SimulationOutput,
)


class TestMonteCarloAnalyzer:
    """Tests for the main analyzer orchestrator."""

    @pytest.fixture
    def config(self):
        return SimulationConfig(
            horizons=[30, 90],
            min_paths=100,
            max_paths=500,
        )

    @pytest.fixture
    def analyzer(self, config):
        return MonteCarloAnalyzer(config=config, seed=42)

    @pytest.fixture
    def mock_price_history(self):
        """Generate mock price history."""
        np.random.seed(42)
        days = 500
        returns = np.random.normal(0.0003, 0.015, days)
        prices = 100 * np.exp(np.cumsum(returns))
        return prices

    def test_calculate_historical_parameters(self, analyzer, mock_price_history):
        mu, sigma = analyzer.calculate_historical_parameters(mock_price_history)

        # Should return annualized values
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0  # Volatility is always positive
        # Typical annualized vol is 15-30% for equities
        assert 0.05 < sigma < 0.60

    def test_run_base_case_simulation(self, analyzer):
        result = analyzer.run_base_case_simulation(
            S0=100.0,
            mu=0.08,
            sigma=0.25,
            days=30,
            n_paths=1000,
        )

        assert isinstance(result, dict)
        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "percentiles" in result
        assert "ci_80" in result
        assert "ci_95" in result
        assert "var_95" in result
        assert "cvar_95" in result
        assert "skewness" in result

    def test_run_stress_scenario(self, analyzer):
        result = analyzer.run_stress_scenario(
            S0=100.0,
            scenario=CrisisScenario.COVID_CRASH,
            days=30,
            n_paths=1000,
            beta=1.2,
        )

        assert "name" in result
        assert "mean" in result
        assert "prob_loss_20pct" in result

    def test_build_horizon_result(self, analyzer):
        base_stats = {
            "mean": 102.0,
            "median": 101.0,
            "std": 8.0,
            "skewness": -0.1,
            "percentiles": {5: 90, 25: 96, 50: 101, 75: 107, 95: 115},
            "ci_80": (94.0, 110.0),
            "ci_95": (90.0, 115.0),
            "var_95": -0.10,
            "cvar_95": -0.15,
        }
        scenario_results = {
            "covid_crash": {
                "name": "COVID-19 Crash",
                "mean": 85.0,
                "median": 84.0,
                "std": 12.0,
                "ci_80": (75.0, 95.0),
                "ci_95": (70.0, 100.0),
                "var_95": -0.30,
                "cvar_95": -0.38,
                "prob_loss_20pct": 0.45,
            }
        }

        result = analyzer.build_horizon_result(30, base_stats, scenario_results)

        assert isinstance(result, HorizonResult)
        assert result.days == 30
        assert result.base_mean == 102.0
        assert "covid_crash" in result.scenarios

    @patch("investment_monitor.simulation.analyzer.MonteCarloAnalyzer._fetch_price_history")
    @patch("investment_monitor.simulation.analyzer.MonteCarloAnalyzer._calculate_beta")
    def test_analyze_returns_simulation_output(
        self, mock_beta, mock_fetch, analyzer, mock_price_history
    ):
        mock_fetch.return_value = mock_price_history
        mock_beta.return_value = 1.1

        result = analyzer.analyze(
            ticker="AAPL",
            entry_price=178.50,
            composite_score=85.0,
        )

        assert isinstance(result, SimulationOutput)
        assert result.ticker == "AAPL"
        assert result.entry_price == 178.50
        assert result.composite_score == 85.0
        assert 30 in result.results
        assert 90 in result.results

    def test_should_run_simulation_above_threshold(self, analyzer):
        assert analyzer.should_run_simulation(composite_score=85.0) is True

    def test_should_run_simulation_below_threshold(self, analyzer):
        assert analyzer.should_run_simulation(composite_score=75.0) is False

    def test_should_run_simulation_with_override(self, analyzer):
        assert analyzer.should_run_simulation(composite_score=50.0, force=True) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

Create `src/investment_monitor/simulation/analyzer.py`:
```python
"""Main Monte Carlo analyzer orchestrator."""

from datetime import date, datetime, timedelta

import numpy as np
from scipy import stats

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


class MonteCarloAnalyzer:
    """Orchestrates Monte Carlo simulations for stock risk analysis."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the analyzer.

        Args:
            config: Simulation configuration (uses defaults if not provided)
            seed: Random seed for reproducibility
        """
        self._config = config or SimulationConfig()
        self._engine = SimulationEngine(seed=seed)
        self._crisis_loader = CrisisDataLoader()
        self._sensitivity = SensitivityAnalyzer(self._engine)

    def should_run_simulation(
        self, composite_score: float, force: bool = False
    ) -> bool:
        """Check if simulation should run based on score threshold."""
        if force:
            return True
        return composite_score >= self._config.score_threshold

    def calculate_historical_parameters(
        self, prices: np.ndarray
    ) -> tuple[float, float]:
        """
        Calculate annualized drift and volatility from price history.

        Args:
            prices: Array of historical prices (oldest to newest)

        Returns:
            Tuple of (annualized_drift, annualized_volatility)
        """
        # Calculate log returns
        log_returns = np.diff(np.log(prices))

        # Annualize (assuming 252 trading days)
        daily_mean = np.mean(log_returns)
        daily_std = np.std(log_returns, ddof=1)

        annualized_drift = daily_mean * 252
        annualized_vol = daily_std * np.sqrt(252)

        return float(annualized_drift), float(annualized_vol)

    def run_base_case_simulation(
        self,
        S0: float,
        mu: float,
        sigma: float,
        days: int,
        n_paths: int,
    ) -> dict:
        """
        Run GBM base case simulation and compute statistics.

        Returns:
            Dict with mean, median, std, percentiles, CI, VaR, CVaR, skewness
        """
        terminal_prices = self._engine.simulate_gbm(S0, mu, sigma, days, n_paths)

        # Compute statistics
        percentiles = {
            p: float(np.percentile(terminal_prices, p)) for p in [5, 25, 50, 75, 95]
        }

        return {
            "mean": float(np.mean(terminal_prices)),
            "median": float(np.median(terminal_prices)),
            "std": float(np.std(terminal_prices)),
            "skewness": float(stats.skew(terminal_prices)),
            "percentiles": percentiles,
            "ci_80": (
                float(np.percentile(terminal_prices, 10)),
                float(np.percentile(terminal_prices, 90)),
            ),
            "ci_95": (
                float(np.percentile(terminal_prices, 2.5)),
                float(np.percentile(terminal_prices, 97.5)),
            ),
            "var_95": self._engine.calculate_var(terminal_prices, S0, 0.95),
            "cvar_95": self._engine.calculate_cvar(terminal_prices, S0, 0.95),
        }

    def run_stress_scenario(
        self,
        S0: float,
        scenario: CrisisScenario,
        days: int,
        n_paths: int,
        beta: float = 1.0,
    ) -> dict:
        """
        Run bootstrap simulation for a stress scenario.

        Args:
            S0: Starting price
            scenario: Crisis scenario to simulate
            days: Time horizon
            n_paths: Number of paths
            beta: Stock beta for adjusting index returns

        Returns:
            Dict with scenario statistics
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

        # Calculate probability of >20% loss
        returns = (terminal_prices - S0) / S0
        prob_loss_20pct = float(np.mean(returns < -0.20))

        metadata = self._crisis_loader.get_scenario_metadata(scenario)

        return {
            "name": metadata["name"],
            "mean": float(np.mean(terminal_prices)),
            "median": float(np.median(terminal_prices)),
            "std": float(np.std(terminal_prices)),
            "ci_80": (
                float(np.percentile(terminal_prices, 10)),
                float(np.percentile(terminal_prices, 90)),
            ),
            "ci_95": (
                float(np.percentile(terminal_prices, 2.5)),
                float(np.percentile(terminal_prices, 97.5)),
            ),
            "var_95": self._engine.calculate_var(terminal_prices, S0, 0.95),
            "cvar_95": self._engine.calculate_cvar(terminal_prices, S0, 0.95),
            "prob_loss_20pct": prob_loss_20pct,
        }

    def build_horizon_result(
        self,
        days: int,
        base_stats: dict,
        scenario_results: dict[str, dict],
    ) -> HorizonResult:
        """Build a HorizonResult from simulation outputs."""
        scenarios = {}
        for key, data in scenario_results.items():
            scenarios[key] = ScenarioResult(
                name=data["name"],
                mean=data["mean"],
                median=data["median"],
                std=data["std"],
                ci_80=data["ci_80"],
                ci_95=data["ci_95"],
                var_95=data["var_95"],
                cvar_95=data["cvar_95"],
                prob_loss_20pct=data["prob_loss_20pct"],
            )

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

    def _fetch_price_history(self, ticker: str, days: int) -> np.ndarray:
        """Fetch historical prices from Yahoo Finance."""
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Extra buffer

        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No price history available for {ticker}")

        return data["Close"].values

    def _calculate_beta(self, ticker: str, lookback_days: int = 252) -> float:
        """Calculate stock beta relative to S&P 500."""
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))

        stock = yf.Ticker(ticker)
        market = yf.Ticker("^GSPC")

        stock_data = stock.history(start=start_date, end=end_date)
        market_data = market.history(start=start_date, end=end_date)

        if stock_data.empty or market_data.empty:
            return 1.0  # Default beta if data unavailable

        # Align dates
        common_dates = stock_data.index.intersection(market_data.index)
        if len(common_dates) < 30:
            return 1.0

        stock_returns = np.diff(np.log(stock_data.loc[common_dates, "Close"].values))
        market_returns = np.diff(np.log(market_data.loc[common_dates, "Close"].values))

        # Beta = Cov(stock, market) / Var(market)
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns, ddof=1)

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance
        # Clamp to reasonable range
        return float(np.clip(beta, 0.3, 3.0))

    def _determine_lookback_days(self, available_days: int) -> int:
        """Determine optimal lookback period based on data availability."""
        if available_days >= self._config.max_lookback_days:
            return self._config.max_lookback_days
        elif available_days >= 756:  # 3 years
            return 756
        elif available_days >= self._config.min_lookback_days:
            return self._config.min_lookback_days
        else:
            return available_days

    def analyze(
        self,
        ticker: str,
        entry_price: float,
        composite_score: float,
        force: bool = False,
    ) -> SimulationOutput:
        """
        Run complete Monte Carlo analysis for a ticker.

        Args:
            ticker: Stock symbol
            entry_price: Current price for simulation
            composite_score: Research score that triggered analysis
            force: Run even if below score threshold

        Returns:
            Complete simulation results

        Raises:
            ValueError: If score below threshold and not forced
        """
        if not self.should_run_simulation(composite_score, force):
            raise ValueError(
                f"Score {composite_score} below threshold {self._config.score_threshold}"
            )

        # Fetch historical data
        max_lookback = self._config.max_lookback_days
        prices = self._fetch_price_history(ticker, max_lookback)
        lookback_days = self._determine_lookback_days(len(prices))

        # Use most recent data up to lookback
        recent_prices = prices[-lookback_days:]
        mu, sigma = self.calculate_historical_parameters(recent_prices)

        # Calculate beta for stress scenarios
        beta = self._calculate_beta(ticker)

        # Determine path count with pilot simulation
        pilot = self._engine.simulate_gbm(entry_price, mu, sigma, 252, 1000)
        n_paths = self._engine.determine_path_count(
            pilot,
            self._config.ci_width_threshold,
            self._config.min_paths,
            self._config.max_paths,
        )

        # Run simulations for each horizon
        results = {}
        for days in self._config.horizons:
            # Base case
            base_stats = self.run_base_case_simulation(
                entry_price, mu, sigma, days, n_paths
            )

            # Stress scenarios
            scenario_results = {}
            scenario_map = {
                "crisis_2008": CrisisScenario.CRISIS_2008,
                "dotcom_crash": CrisisScenario.DOTCOM_CRASH,
                "covid_crash": CrisisScenario.COVID_CRASH,
                "stagflation_1970s": CrisisScenario.STAGFLATION_1970S,
                "black_monday_1987": CrisisScenario.BLACK_MONDAY_1987,
                "rising_rates_2022": CrisisScenario.RISING_RATES_2022,
                "regime_democrat": CrisisScenario.REGIME_DEMOCRAT,
                "regime_republican": CrisisScenario.REGIME_REPUBLICAN,
            }

            for key, scenario in scenario_map.items():
                if self._config.scenarios_enabled.get(key, True):
                    scenario_results[key] = self.run_stress_scenario(
                        entry_price, scenario, days, n_paths, beta
                    )

            results[days] = self.build_horizon_result(days, base_stats, scenario_results)

        # Run sensitivity analysis on longest horizon
        longest_horizon = max(self._config.horizons)
        lookback_vols = {}
        for lb in [252, 756, 1260]:
            if lb <= len(prices):
                _, vol = self.calculate_historical_parameters(prices[-lb:])
                lookback_vols[lb] = vol
            else:
                lookback_vols[lb] = sigma  # Use base if not enough data

        sensitivity = self._sensitivity.run_analysis(
            S0=entry_price,
            mu=mu,
            sigma=sigma,
            days=longest_horizon,
            n_paths=min(n_paths, 5000),  # Limit for sensitivity runs
            lookback_volatilities=lookback_vols,
            volatility_multipliers=self._config.volatility_multipliers,
        )

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_analyzer.py -v`
Expected: PASS

**Step 5: Update package __init__.py**

Add to `src/investment_monitor/simulation/__init__.py`:
```python
from .analyzer import MonteCarloAnalyzer
```

Add `"MonteCarloAnalyzer"` to `__all__`.

**Step 6: Commit**

```bash
git add src/investment_monitor/simulation/analyzer.py src/investment_monitor/simulation/__init__.py tests/test_simulation/test_analyzer.py
git commit -m "feat(simulation): implement MonteCarloAnalyzer orchestrator"
```

---

## Task 7: Add Report Formatting for Research Reports

**Files:**
- Create: `src/investment_monitor/simulation/report_formatter.py`
- Test: `tests/test_simulation/test_report_formatter.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_report_formatter.py`:
```python
"""Tests for simulation report formatter."""

import pytest

from investment_monitor.simulation.models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationOutput,
)
from investment_monitor.simulation.report_formatter import SimulationReportFormatter


@pytest.fixture
def sample_output():
    """Create a sample SimulationOutput for testing."""
    scenario = ScenarioResult(
        name="2008 Financial Crisis",
        mean=124.0,
        median=120.0,
        std=25.0,
        ci_80=(98.0, 142.0),
        ci_95=(85.0, 158.0),
        var_95=-0.30,
        cvar_95=-0.38,
        prob_loss_20pct=0.68,
    )

    horizon_30 = HorizonResult(
        days=30,
        base_mean=182.0,
        base_median=180.0,
        base_std=8.0,
        base_skewness=-0.1,
        base_percentiles={5: 165, 25: 175, 50: 180, 75: 188, 95: 195},
        base_ci_80=(171.0, 188.0),
        base_ci_95=(165.0, 195.0),
        base_var_95=-0.08,
        base_cvar_95=-0.12,
        scenarios={"crisis_2008": scenario},
    )

    horizon_252 = HorizonResult(
        days=252,
        base_mean=198.0,
        base_median=195.0,
        base_std=25.0,
        base_skewness=-0.15,
        base_percentiles={5: 149, 25: 175, 50: 195, 75: 215, 95: 240},
        base_ci_80=(160.0, 230.0),
        base_ci_95=(149.0, 245.0),
        base_var_95=-0.17,
        base_cvar_95=-0.24,
        scenarios={"crisis_2008": scenario},
    )

    sensitivity = SensitivityResult(
        volatility_impact=65.0,
        drift_impact=25.0,
        lookback_impact=10.0,
        primary_driver="volatility",
        volatility_range={0.5: 195.0, 1.0: 182.0, 1.5: 165.0},
        drift_range={"pessimistic": 170.0, "neutral": 182.0, "optimistic": 195.0},
        lookback_range={252: 180.0, 756: 182.0, 1260: 184.0},
    )

    return SimulationOutput(
        ticker="AAPL",
        entry_price=178.50,
        composite_score=85.0,
        num_simulations=10000,
        lookback_days=756,
        volatility=0.25,
        drift=0.08,
        results={30: horizon_30, 252: horizon_252},
        sensitivity=sensitivity,
    )


class TestSimulationReportFormatter:
    """Tests for report formatting."""

    def test_format_markdown_contains_header(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "## Risk Analysis (Monte Carlo Simulation)" in markdown

    def test_format_markdown_contains_entry_info(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "$178.50" in markdown
        assert "10,000" in markdown or "10000" in markdown

    def test_format_markdown_contains_horizons(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "30 days" in markdown
        assert "1 year" in markdown or "252" in markdown

    def test_format_markdown_contains_stress_scenarios(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "2008 Financial Crisis" in markdown

    def test_format_markdown_contains_risk_metrics(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "VaR" in markdown or "Value at Risk" in markdown

    def test_format_markdown_contains_sensitivity(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "Sensitivity" in markdown
        assert "volatility" in markdown.lower()

    def test_format_markdown_contains_disclaimer(self, sample_output):
        formatter = SimulationReportFormatter()
        markdown = formatter.format_markdown(sample_output)

        assert "Not a prediction" in markdown or "not a prediction" in markdown

    def test_format_compact(self, sample_output):
        """Test compact format for Slack."""
        formatter = SimulationReportFormatter()
        compact = formatter.format_compact(sample_output)

        assert "AAPL" in compact
        assert len(compact) < 1000  # Should be concise
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_report_formatter.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `src/investment_monitor/simulation/report_formatter.py`:
```python
"""Format simulation results for research reports."""

from .models import SimulationOutput


class SimulationReportFormatter:
    """Format Monte Carlo simulation results for reports."""

    def __init__(self, disclaimer: str | None = None):
        """Initialize with optional custom disclaimer."""
        self._disclaimer = (
            disclaimer
            or "Simulation based on historical returns. Not a prediction. Past performance ≠ future results."
        )

    def format_markdown(self, output: SimulationOutput) -> str:
        """
        Format simulation results as markdown for email reports.

        Args:
            output: Complete simulation output

        Returns:
            Markdown-formatted string
        """
        lines = []

        # Header
        lines.append("## Risk Analysis (Monte Carlo Simulation)")
        lines.append("")
        lines.append(
            f"**Entry Point:** ${output.entry_price:.2f} | "
            f"**Simulations:** {output.num_simulations:,} paths | "
            f"**Data:** {output.lookback_days / 252:.1f} years"
        )
        lines.append("")

        # Projected Price Ranges
        lines.append("### Projected Price Ranges")
        lines.append("")
        lines.append("| Horizon | Expected | 80% Confidence | Worst 5% |")
        lines.append("|---------|----------|----------------|----------|")

        for days in sorted(output.results.keys()):
            horizon = output.results[days]
            horizon_label = self._format_horizon_label(days)
            worst_5 = horizon.base_percentiles.get(5, horizon.base_ci_95[0])
            lines.append(
                f"| {horizon_label} | ${horizon.base_mean:.0f} | "
                f"${horizon.base_ci_80[0]:.0f} - ${horizon.base_ci_80[1]:.0f} | "
                f"Below ${worst_5:.0f} |"
            )
        lines.append("")

        # Stress Test Results (longest horizon only)
        longest_days = max(output.results.keys())
        longest_horizon = output.results[longest_days]

        if longest_horizon.scenarios:
            lines.append(
                f"### Stress Test Results ({self._format_horizon_label(longest_days)} Horizon)"
            )
            lines.append("")
            lines.append(
                "| Scenario | Expected | 80% Range | Chance of >20% Loss |"
            )
            lines.append("|----------|----------|-----------|---------------------|")

            for key, scenario in longest_horizon.scenarios.items():
                prob_pct = f"{scenario.prob_loss_20pct * 100:.0f}%"
                lines.append(
                    f"| {scenario.name} | ${scenario.mean:.0f} | "
                    f"${scenario.ci_80[0]:.0f} - ${scenario.ci_80[1]:.0f} | "
                    f"{prob_pct} |"
                )
            lines.append("")

        # Risk Metrics
        lines.append("### Risk Metrics")
        lines.append("")
        lines.append(
            f"- **Value at Risk (95%):** {longest_horizon.base_var_95 * 100:.1f}%"
        )
        lines.append(
            f"- **Conditional VaR (95%):** {longest_horizon.base_cvar_95 * 100:.1f}%"
        )

        # Calculate probability of gain
        prob_gain = 1.0 - (
            len([p for p in [longest_horizon.base_var_95] if p < 0]) / 1
        )
        # Approximate from percentiles
        if longest_horizon.base_median > output.entry_price:
            prob_gain = 0.5 + (
                longest_horizon.base_mean - output.entry_price
            ) / (2 * longest_horizon.base_std)
            prob_gain = min(max(prob_gain, 0.3), 0.8)

        lines.append(f"- **Base Case Probability of Gain:** {prob_gain * 100:.0f}%")
        lines.append("")

        # Sensitivity Check
        sens = output.sensitivity
        lines.append("### Sensitivity Check")
        lines.append("")
        lines.append("| Input Assumption | Impact on Results |")
        lines.append("|------------------|-------------------|")

        vol_impact = self._impact_label(sens.volatility_impact)
        drift_impact = self._impact_label(sens.drift_impact)
        lookback_impact = self._impact_label(sens.lookback_impact)

        vol_range = max(sens.volatility_range.values()) - min(
            sens.volatility_range.values()
        )
        drift_range = max(sens.drift_range.values()) - min(sens.drift_range.values())
        lookback_range = max(sens.lookback_range.values()) - min(
            sens.lookback_range.values()
        )

        lines.append(
            f"| Volatility | {vol_impact} — ±${vol_range / 2:.0f} swing |"
        )
        lines.append(
            f"| Return Assumption | {drift_impact} — ±${drift_range / 2:.0f} swing |"
        )
        lines.append(
            f"| Lookback Period | {lookback_impact} — ±${lookback_range / 2:.0f} swing |"
        )
        lines.append("")

        lines.append(
            f"**Bottom Line:** Projections are most sensitive to {sens.primary_driver} assumptions."
        )
        lines.append("")
        lines.append("---")
        lines.append(f"*{self._disclaimer}*")

        return "\n".join(lines)

    def format_compact(self, output: SimulationOutput) -> str:
        """
        Format simulation results as compact text for Slack.

        Args:
            output: Complete simulation output

        Returns:
            Compact text string
        """
        lines = []

        lines.append(f"📊 *{output.ticker} Monte Carlo Analysis*")
        lines.append(f"Entry: ${output.entry_price:.2f} | Score: {output.composite_score:.0f}")
        lines.append("")

        # Key metrics from longest horizon
        longest_days = max(output.results.keys())
        horizon = output.results[longest_days]

        lines.append(f"*{self._format_horizon_label(longest_days)} Outlook:*")
        lines.append(f"• Expected: ${horizon.base_mean:.0f}")
        lines.append(
            f"• 80% Range: ${horizon.base_ci_80[0]:.0f} - ${horizon.base_ci_80[1]:.0f}"
        )
        lines.append(f"• VaR (95%): {horizon.base_var_95 * 100:.1f}%")

        # Worst stress scenario
        if horizon.scenarios:
            worst = max(horizon.scenarios.values(), key=lambda s: s.prob_loss_20pct)
            lines.append(
                f"• Worst Stress ({worst.name}): {worst.prob_loss_20pct * 100:.0f}% chance of >20% loss"
            )

        lines.append("")
        lines.append(f"Primary risk driver: {output.sensitivity.primary_driver}")

        return "\n".join(lines)

    def _format_horizon_label(self, days: int) -> str:
        """Format days into human-readable label."""
        if days <= 30:
            return "30 days"
        elif days <= 90:
            return "90 days"
        elif days <= 252:
            return "1 year"
        else:
            years = days / 252
            return f"{years:.1f} years"

    def _impact_label(self, impact: float) -> str:
        """Convert impact score to label."""
        if impact >= 50:
            return "HIGH"
        elif impact >= 25:
            return "MEDIUM"
        else:
            return "LOW"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_report_formatter.py -v`
Expected: PASS

**Step 5: Update package __init__.py**

Add to exports.

**Step 6: Commit**

```bash
git add src/investment_monitor/simulation/report_formatter.py tests/test_simulation/test_report_formatter.py src/investment_monitor/simulation/__init__.py
git commit -m "feat(simulation): add report formatter for markdown and Slack output"
```

---

## Task 8: Add CLI Commands for Simulation

**Files:**
- Modify: `src/investment_monitor/research_cli.py`
- Test: `tests/test_simulation/test_cli.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_cli.py`:
```python
"""Tests for simulation CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from investment_monitor.research_cli import app


runner = CliRunner()


class TestSimulateCLI:
    """Tests for simulate CLI commands."""

    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf.Ticker")
    def test_simulate_single_ticker(self, mock_ticker, mock_analyzer):
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.info = {"regularMarketPrice": 178.50}
        mock_ticker.return_value = mock_ticker_obj

        mock_output = MagicMock()
        mock_output.ticker = "AAPL"
        mock_output.entry_price = 178.50
        mock_output.num_simulations = 10000
        mock_analyzer.return_value.analyze.return_value = mock_output

        result = runner.invoke(app, ["simulate", "--ticker", "AAPL"])

        assert result.exit_code == 0
        mock_analyzer.return_value.analyze.assert_called_once()

    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf.Ticker")
    def test_simulate_multiple_tickers(self, mock_ticker, mock_analyzer):
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.info = {"regularMarketPrice": 100.00}
        mock_ticker.return_value = mock_ticker_obj

        mock_output = MagicMock()
        mock_output.ticker = "AAPL"
        mock_output.entry_price = 100.0
        mock_output.num_simulations = 10000
        mock_analyzer.return_value.analyze.return_value = mock_output

        result = runner.invoke(
            app, ["simulate", "--tickers", "AAPL,MSFT,GOOGL"]
        )

        assert result.exit_code == 0
        assert mock_analyzer.return_value.analyze.call_count == 3

    def test_simulate_requires_ticker_or_auto(self):
        result = runner.invoke(app, ["simulate"])

        assert result.exit_code != 0 or "Error" in result.output or "ticker" in result.output.lower()

    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_by_ticker(self, mock_get_results):
        mock_get_results.return_value = []

        result = runner.invoke(
            app, ["simulation-results", "--ticker", "AAPL"]
        )

        assert result.exit_code == 0
        mock_get_results.assert_called()

    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_latest(self, mock_get_results):
        mock_get_results.return_value = []

        result = runner.invoke(app, ["simulation-results", "--latest", "5"])

        assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_cli.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Modify `src/investment_monitor/research_cli.py` to add simulation commands.

Add imports at top:
```python
import yfinance as yf

from investment_monitor.simulation import MonteCarloAnalyzer, SimulationConfig
from investment_monitor.simulation.report_formatter import SimulationReportFormatter
from investment_monitor.storage.research_operations import (
    get_simulation_results,
    save_simulation_result,
)
```

Add new commands:
```python
@app.command()
def simulate(
    ticker: str = typer.Option(None, help="Single ticker to simulate"),
    tickers: str = typer.Option(None, help="Comma-separated list of tickers"),
    auto: bool = typer.Option(False, help="Auto-simulate all candidates >= 80 score"),
    horizons: str = typer.Option("30,90,252", help="Comma-separated time horizons in days"),
    min_paths: int = typer.Option(1000, help="Minimum simulation paths"),
    force: bool = typer.Option(False, help="Force simulation regardless of score"),
):
    """
    Run Monte Carlo simulation for risk analysis.

    Examples:
        investment-research simulate --ticker AAPL
        investment-research simulate --tickers AAPL,MSFT,GOOGL
        investment-research simulate --auto
    """
    if not ticker and not tickers and not auto:
        typer.echo("Error: Provide --ticker, --tickers, or --auto")
        raise typer.Exit(1)

    # Parse horizons
    horizon_list = [int(h.strip()) for h in horizons.split(",")]

    config = SimulationConfig(
        horizons=horizon_list,
        min_paths=min_paths,
    )
    analyzer = MonteCarloAnalyzer(config=config)
    formatter = SimulationReportFormatter()

    # Build ticker list
    ticker_list = []
    if ticker:
        ticker_list.append(ticker.upper())
    if tickers:
        ticker_list.extend([t.strip().upper() for t in tickers.split(",")])
    if auto:
        # Get candidates from DB with score >= 80
        from investment_monitor.storage.research_operations import get_high_scoring_candidates
        candidates = get_high_scoring_candidates(min_score=80.0)
        ticker_list.extend([c.ticker for c in candidates])

    if not ticker_list:
        typer.echo("No tickers to simulate")
        raise typer.Exit(0)

    typer.echo(f"Running Monte Carlo simulation for {len(ticker_list)} ticker(s)...")

    for tkr in ticker_list:
        try:
            # Get current price
            stock = yf.Ticker(tkr)
            price = stock.info.get("regularMarketPrice") or stock.info.get("currentPrice")
            if not price:
                typer.echo(f"  {tkr}: Could not fetch current price, skipping")
                continue

            # Get score (if available)
            from investment_monitor.storage.research_operations import get_latest_score
            score_record = get_latest_score(tkr)
            composite_score = score_record.composite_score if score_record else 0.0

            typer.echo(f"  {tkr}: Running simulation (price=${price:.2f}, score={composite_score:.1f})...")

            output = analyzer.analyze(
                ticker=tkr,
                entry_price=price,
                composite_score=composite_score,
                force=force or not score_record,
            )

            # Save to DB
            save_simulation_result(output)

            # Display summary
            typer.echo(f"    ✓ Completed {output.num_simulations:,} simulations")
            typer.echo(formatter.format_compact(output))
            typer.echo("")

        except ValueError as e:
            typer.echo(f"  {tkr}: {e}")
        except Exception as e:
            typer.echo(f"  {tkr}: Error - {e}")


@app.command("simulation-results")
def simulation_results(
    ticker: str = typer.Option(None, help="Show results for specific ticker"),
    latest: int = typer.Option(None, help="Show N most recent simulation results"),
):
    """
    View simulation results.

    Examples:
        investment-research simulation-results --ticker AAPL
        investment-research simulation-results --latest 10
    """
    results = get_simulation_results(ticker=ticker, limit=latest or 10)

    if not results:
        typer.echo("No simulation results found")
        raise typer.Exit(0)

    formatter = SimulationReportFormatter()

    for result in results:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Ticker: {result.ticker}")
        typer.echo(f"Run Date: {result.run_date}")
        typer.echo(f"Entry Price: ${result.entry_price:.2f}")
        typer.echo(f"Score: {result.composite_score:.1f}")
        typer.echo(f"Simulations: {result.num_simulations:,}")
        typer.echo(f"Volatility: {result.volatility * 100:.1f}%")
        typer.echo(f"Drift: {result.drift * 100:.1f}%")

        # Display horizon summaries
        for horizon_key in ["results_30d", "results_90d", "results_252d"]:
            data = getattr(result, horizon_key, {})
            if data and "base_mean" in data:
                days = int(horizon_key.replace("results_", "").replace("d", ""))
                typer.echo(f"\n{days}-day projection:")
                typer.echo(f"  Expected: ${data['base_mean']:.0f}")
                if "base_ci_80" in data:
                    ci = data["base_ci_80"]
                    typer.echo(f"  80% CI: ${ci[0]:.0f} - ${ci[1]:.0f}")
```

**Step 4: Add storage operations**

Add to `src/investment_monitor/storage/research_operations.py`:
```python
from datetime import date

from sqlalchemy.orm import Session

from .research_models import SimulationResult
from ..simulation.models import SimulationOutput


def save_simulation_result(output: SimulationOutput, session: Session = None) -> SimulationResult:
    """Save simulation output to database."""
    if session is None:
        from .database import get_session
        session = get_session()

    result = SimulationResult(
        ticker=output.ticker,
        run_date=date.today(),
        entry_price=output.entry_price,
        composite_score=output.composite_score,
        num_simulations=output.num_simulations,
        lookback_days=output.lookback_days,
        volatility=output.volatility,
        drift=output.drift,
        results_30d=output.results[30].model_dump() if 30 in output.results else {},
        results_90d=output.results[90].model_dump() if 90 in output.results else {},
        results_252d=output.results[252].model_dump() if 252 in output.results else {},
        sensitivity_analysis=output.sensitivity.model_dump(),
    )

    session.add(result)
    session.commit()
    return result


def get_simulation_results(
    ticker: str = None,
    limit: int = 10,
    session: Session = None,
) -> list[SimulationResult]:
    """Get simulation results from database."""
    if session is None:
        from .database import get_session
        session = get_session()

    query = session.query(SimulationResult)

    if ticker:
        query = query.filter(SimulationResult.ticker == ticker.upper())

    query = query.order_by(SimulationResult.created_at.desc())

    if limit:
        query = query.limit(limit)

    return query.all()


def get_high_scoring_candidates(min_score: float = 80.0, session: Session = None) -> list:
    """Get candidates with composite score >= threshold."""
    if session is None:
        from .database import get_session
        session = get_session()

    from .research_models import CandidateScore

    return (
        session.query(CandidateScore)
        .filter(CandidateScore.composite_score >= min_score)
        .order_by(CandidateScore.composite_score.desc())
        .all()
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_cli.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/investment_monitor/research_cli.py src/investment_monitor/storage/research_operations.py tests/test_simulation/test_cli.py
git commit -m "feat(simulation): add CLI commands for simulate and simulation-results"
```

---

## Task 9: Integrate with ResearchOrchestrator for Auto-Simulation

**Files:**
- Modify: `src/investment_monitor/research/orchestrator.py`
- Test: `tests/test_simulation/test_orchestrator_integration.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_orchestrator_integration.py`:
```python
"""Tests for simulation integration with research orchestrator."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from investment_monitor.research.orchestrator import ResearchOrchestrator


class TestOrchestratorSimulationIntegration:
    """Tests for auto-simulation in research orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        return ResearchOrchestrator()

    @pytest.mark.asyncio
    @patch("investment_monitor.research.orchestrator.MonteCarloAnalyzer")
    @patch("investment_monitor.research.orchestrator.FundamentalsCollector")
    @patch("investment_monitor.research.orchestrator.ResearchScorer")
    async def test_auto_simulation_for_high_score(
        self, mock_scorer, mock_collector, mock_analyzer, orchestrator
    ):
        """High-scoring candidates trigger auto-simulation."""
        # Setup mocks
        mock_fundamentals = MagicMock()
        mock_fundamentals.ticker = "AAPL"
        mock_collector.return_value.collect = AsyncMock(return_value=MagicMock(
            data={"AAPL": mock_fundamentals}
        ))

        mock_score = MagicMock()
        mock_score.composite_score = 85.0  # Above threshold
        mock_scorer.return_value.score_candidate = AsyncMock(return_value=mock_score)

        mock_sim_output = MagicMock()
        mock_analyzer.return_value.analyze.return_value = mock_sim_output

        # Run research
        result = await orchestrator.research_ticker("AAPL", run_simulation=True)

        # Verify simulation was triggered
        mock_analyzer.return_value.analyze.assert_called_once()

    @pytest.mark.asyncio
    @patch("investment_monitor.research.orchestrator.MonteCarloAnalyzer")
    @patch("investment_monitor.research.orchestrator.FundamentalsCollector")
    @patch("investment_monitor.research.orchestrator.ResearchScorer")
    async def test_no_simulation_for_low_score(
        self, mock_scorer, mock_collector, mock_analyzer, orchestrator
    ):
        """Low-scoring candidates skip simulation."""
        mock_fundamentals = MagicMock()
        mock_fundamentals.ticker = "XYZ"
        mock_collector.return_value.collect = AsyncMock(return_value=MagicMock(
            data={"XYZ": mock_fundamentals}
        ))

        mock_score = MagicMock()
        mock_score.composite_score = 65.0  # Below threshold
        mock_scorer.return_value.score_candidate = AsyncMock(return_value=mock_score)

        result = await orchestrator.research_ticker("XYZ", run_simulation=True)

        # Simulation should not be called
        mock_analyzer.return_value.analyze.assert_not_called()

    @pytest.mark.asyncio
    @patch("investment_monitor.research.orchestrator.MonteCarloAnalyzer")
    async def test_simulation_results_included_in_report(
        self, mock_analyzer, orchestrator
    ):
        """Simulation results are included in research report."""
        mock_output = MagicMock()
        mock_output.results = {30: MagicMock(), 252: MagicMock()}
        mock_analyzer.return_value.analyze.return_value = mock_output

        # The orchestrator should include simulation data
        # This tests the integration point exists
        assert hasattr(orchestrator, "_run_simulation") or hasattr(orchestrator, "run_simulation")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_orchestrator_integration.py -v`
Expected: FAIL

**Step 3: Modify ResearchOrchestrator**

Add imports and integration to `src/investment_monitor/research/orchestrator.py`:

Add imports:
```python
from investment_monitor.simulation import MonteCarloAnalyzer, SimulationConfig
from investment_monitor.simulation.report_formatter import SimulationReportFormatter
from investment_monitor.storage.research_operations import save_simulation_result
```

Modify the `research_ticker` method to include simulation:
```python
async def research_ticker(
    self,
    ticker: str,
    run_report: bool = True,
    run_simulation: bool = True,
) -> ResearchResult:
    """
    Run complete research analysis for a ticker.

    Args:
        ticker: Stock symbol
        run_report: Generate AI research report
        run_simulation: Run Monte Carlo simulation if score >= 80

    Returns:
        ResearchResult with all analysis
    """
    # ... existing code for fundamentals, scoring, etc ...

    # After scoring, check if simulation should run
    simulation_output = None
    if run_simulation and score.composite_score >= 80.0:
        try:
            simulation_output = await self._run_simulation(
                ticker=ticker,
                entry_price=fundamentals.current_price or self._get_current_price(ticker),
                composite_score=score.composite_score,
            )
        except Exception as e:
            logger.warning(f"Simulation failed for {ticker}: {e}")

    # Include simulation in report if available
    report_body = None
    if run_report:
        report_body = await self._generate_report(fundamentals, score)
        if simulation_output:
            formatter = SimulationReportFormatter()
            report_body += "\n\n" + formatter.format_markdown(simulation_output)

    return ResearchResult(
        ticker=ticker,
        fundamentals=fundamentals,
        score=score,
        report=report_body,
        simulation=simulation_output,
    )

async def _run_simulation(
    self,
    ticker: str,
    entry_price: float,
    composite_score: float,
) -> SimulationOutput:
    """Run Monte Carlo simulation for a ticker."""
    analyzer = MonteCarloAnalyzer()

    output = analyzer.analyze(
        ticker=ticker,
        entry_price=entry_price,
        composite_score=composite_score,
    )

    # Save to database
    save_simulation_result(output)

    return output
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_orchestrator_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/investment_monitor/research/orchestrator.py tests/test_simulation/test_orchestrator_integration.py
git commit -m "feat(simulation): integrate Monte Carlo with ResearchOrchestrator"
```

---

## Task 10: Add Configuration Support

**Files:**
- Modify: `src/investment_monitor/config.py`
- Modify: `config/research.yaml.example`
- Test: `tests/test_simulation/test_config.py`

**Step 1: Write the failing test**

Create `tests/test_simulation/test_config.py`:
```python
"""Tests for simulation configuration."""

import pytest

from investment_monitor.config import load_config, MonteCarloSettings


class TestMonteCarloConfig:
    """Tests for Monte Carlo configuration."""

    def test_default_monte_carlo_settings(self):
        settings = MonteCarloSettings()

        assert settings.score_threshold == 80.0
        assert settings.horizons == [30, 90, 252]
        assert settings.min_paths == 1000
        assert settings.max_paths == 50000

    def test_load_config_includes_monte_carlo(self):
        config = load_config()

        assert hasattr(config, "monte_carlo")
        assert isinstance(config.monte_carlo, MonteCarloSettings)

    def test_custom_monte_carlo_settings(self):
        settings = MonteCarloSettings(
            score_threshold=75.0,
            horizons=[30, 60, 90],
            min_paths=5000,
        )

        assert settings.score_threshold == 75.0
        assert settings.horizons == [30, 60, 90]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation/test_config.py -v`
Expected: FAIL

**Step 3: Update config.py**

Add MonteCarloSettings to `src/investment_monitor/config.py`:
```python
class MonteCarloSettings(BaseModel):
    """Monte Carlo simulation settings."""

    score_threshold: float = 80.0
    horizons: list[int] = [30, 90, 252]
    min_paths: int = 1000
    max_paths: int = 50000
    ci_width_threshold: float = 0.15
    min_lookback_days: int = 252
    max_lookback_days: int = 1260
    volatility_multipliers: list[float] = [0.5, 0.8, 1.0, 1.2, 1.5]
    drift_scenarios: list[str] = ["pessimistic", "neutral", "optimistic"]
    include_in_reports: bool = True
    disclaimer: str = "Simulation based on historical returns. Not a prediction."

    # Scenario toggles
    scenarios: dict[str, bool] = {
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
```

Add to main Config class:
```python
class Config(BaseModel):
    # ... existing fields ...
    monte_carlo: MonteCarloSettings = MonteCarloSettings()
```

**Step 4: Update config/research.yaml.example**

Add monte_carlo section:
```yaml
# Monte Carlo Simulation Settings
monte_carlo:
  score_threshold: 80
  horizons: [30, 90, 252]
  min_paths: 1000
  max_paths: 50000
  ci_width_threshold: 0.15
  min_lookback_days: 252
  max_lookback_days: 1260
  include_in_reports: true
  disclaimer: "Simulation based on historical returns. Not a prediction."

  scenarios:
    base_gbm: true
    crisis_2008: true
    dotcom_crash: true
    covid_crash: true
    stagflation_1970s: true
    black_monday_1987: true
    rising_rates_2022: true
    regime_democrat: true
    regime_republican: true
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_simulation/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/investment_monitor/config.py config/research.yaml.example tests/test_simulation/test_config.py
git commit -m "feat(simulation): add Monte Carlo configuration support"
```

---

## Task 11: Create Database Migration for SimulationResult Table

**Files:**
- Create: `alembic/versions/xxxx_add_simulation_results.py`

**Step 1: Generate migration**

Run: `alembic revision -m "add_simulation_results_table"`

**Step 2: Edit migration file**

```python
"""add_simulation_results_table

Revision ID: xxxx
Revises: previous_revision
Create Date: 2026-01-31 xx:xx:xx

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers
revision = 'xxxx'
down_revision = 'previous'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'simulation_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(10), nullable=False),
        sa.Column('run_date', sa.Date(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('composite_score', sa.Float(), nullable=False),
        sa.Column('num_simulations', sa.Integer(), nullable=False),
        sa.Column('lookback_days', sa.Integer(), nullable=False),
        sa.Column('volatility', sa.Float(), nullable=False),
        sa.Column('drift', sa.Float(), nullable=False),
        sa.Column('results_30d', sa.JSON(), nullable=True),
        sa.Column('results_90d', sa.JSON(), nullable=True),
        sa.Column('results_252d', sa.JSON(), nullable=True),
        sa.Column('sensitivity_analysis', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_simulation_results_ticker', 'simulation_results', ['ticker'])
    op.create_index('ix_simulation_results_run_date', 'simulation_results', ['run_date'])


def downgrade() -> None:
    op.drop_index('ix_simulation_results_run_date')
    op.drop_index('ix_simulation_results_ticker')
    op.drop_table('simulation_results')
```

**Step 3: Run migration**

Run: `alembic upgrade head`
Expected: Migration completes successfully

**Step 4: Commit**

```bash
git add alembic/versions/
git commit -m "feat(simulation): add database migration for simulation_results table"
```

---

## Task 12: Run Full Test Suite and Integration Test

**Files:**
- All test files in `tests/test_simulation/`

**Step 1: Run all simulation tests**

Run: `pytest tests/test_simulation/ -v`
Expected: All tests pass

**Step 2: Run full test suite to check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: No regressions, all existing tests still pass

**Step 3: Manual integration test**

Run:
```bash
# Test CLI
investment-research simulate --ticker AAPL --force

# View results
investment-research simulation-results --ticker AAPL
```

Expected: Simulation runs and results are displayed

**Step 4: Final commit**

```bash
git add -A
git commit -m "test(simulation): verify full test suite passes with Monte Carlo integration"
```

---

## Summary

This plan implements Monte Carlo simulation in 12 tasks:

1. **Pydantic Models** - Runtime data models for simulation results
2. **SQLAlchemy Model** - Database persistence for simulation results
3. **Crisis Data Loader** - Load bundled historical crisis data CSVs
4. **Simulation Engine** - Core GBM and bootstrap algorithms
5. **Sensitivity Analyzer** - Analyze parameter sensitivity
6. **MonteCarloAnalyzer** - Main orchestrator combining all components
7. **Report Formatter** - Format results for email/Slack
8. **CLI Commands** - `simulate` and `simulation-results` commands
9. **Orchestrator Integration** - Auto-simulation for high-scoring candidates
10. **Configuration** - YAML config support
11. **Database Migration** - Alembic migration for new table
12. **Integration Testing** - Full test suite verification

Each task follows TDD with explicit test-first approach and frequent commits.
