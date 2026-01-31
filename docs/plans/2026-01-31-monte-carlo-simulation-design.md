# Monte Carlo Simulation for Research Screening

**Date:** 2026-01-31
**Status:** Approved
**Goal:** Add probabilistic risk analysis to stock candidate research via Monte Carlo simulation

## Overview

Implement Monte Carlo simulation to provide ballpark risk/return projections for research candidates. Simulations run automatically for high-scoring candidates (composite score ≥ 80) and results are included in research reports sent via email/Slack.

This is **not** for precise predictions—it's for understanding potential downside scenarios and stress-testing candidates before adding them to the watchlist.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Use case | Research screening (not portfolio risk) |
| Gating | Composite score ≥ 80, with manual CLI override |
| Base methodology | Hybrid: GBM for base case, bootstrap for stress scenarios |
| Stress scenarios | 8 scenarios (see below) |
| Time horizons | 30, 90, 252 days simultaneously |
| Path count | Adaptive: 1K → 10K → 50K based on CI width |
| Sensitivity analysis | Volatility, drift, lookback period |
| Output metrics | VaR, CVaR, distributions, confidence intervals, scenario comparison |
| Integration | Separate from scoring; included in research reports |
| Crisis data | Pre-bundled CSVs with S&P 500 returns + beta adjustment |
| Lookback | Adaptive: 1-5 years based on availability |
| Library | NumPy/SciPy (QuantLib optional future enhancement) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research Pipeline Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Candidates ──► Scoring (5-factor) ──► composite_score ≥ 80?    │
│                                              │                   │
│                              ┌───────────────┴───────────────┐   │
│                              ▼                               ▼   │
│                         Yes: Auto-simulate              No: Skip │
│                              │                               │   │
│                              ▼                               │   │
│                    MonteCarloAnalyzer                        │   │
│                    ├─ GBM Base Case                          │   │
│                    ├─ 8 Stress Scenarios (bootstrap)         │   │
│                    ├─ 3 Time Horizons (30/90/252 days)       │   │
│                    └─ Sensitivity Analysis                   │   │
│                              │                               │   │
│                              ▼                               │   │
│                    SimulationResult ──► SQLite               │   │
│                              │                               │   │
│                              ▼                               │   │
│                    ResearchReport (includes MC section)      │   │
│                              │                               │   │
│                              ▼                               │   │
│                    Email / Slack Notification                │   │
│                                                              │   │
│  Manual Override: cli.py simulate --ticker XYZ ─────────────►│   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Stress Scenarios

| Scenario | Period | Data Source |
|----------|--------|-------------|
| 2008 Financial Crisis | Sep 2008 - Mar 2009 | S&P 500 daily returns |
| Dot-com Crash | Mar 2000 - Oct 2002 | S&P 500 daily returns |
| COVID Crash | Feb - Mar 2020 | S&P 500 daily returns |
| 1970s Stagflation | 1973-1974 | S&P 500 daily returns |
| Black Monday | Oct 1987 | S&P 500 daily returns |
| Rising Rates 2022 | Jan - Dec 2022 | S&P 500 daily returns |
| Regime: Democrat | Historical average | Aggregated by administration |
| Regime: Republican | Historical average | Aggregated by administration |

Crisis data is pre-bundled as CSV files. Individual stock stress returns are derived by applying beta adjustment to index returns.

## Data Models

### SQLAlchemy Model (persistence)

```python
class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id: int                     # Primary key
    ticker: str                 # Stock symbol
    run_date: date              # When simulation was run
    entry_price: float          # Price at simulation time
    composite_score: float      # Score that triggered simulation

    # Parameters used
    num_simulations: int        # Actual paths run (adaptive)
    lookback_days: int          # Historical data used
    volatility: float           # Annualized vol used for GBM
    drift: float                # Annualized return used for GBM

    # Results stored as JSON blobs per horizon
    results_30d: JSON           # HorizonResult serialized
    results_90d: JSON
    results_252d: JSON

    # Sensitivity analysis summary
    sensitivity_analysis: JSON

    created_at: datetime
```

### Pydantic Models (runtime)

```python
class HorizonResult(BaseModel):
    days: int                           # 30, 90, or 252

    # Base case (GBM)
    base_mean: float
    base_median: float
    base_std: float
    base_skewness: float
    base_percentiles: dict[int, float]  # {5: 142.5, 25: 158.2, ...}
    base_ci_80: tuple[float, float]     # (low, high)
    base_ci_95: tuple[float, float]
    base_var_95: float                  # Value at Risk
    base_cvar_95: float                 # Conditional VaR

    # Stress scenarios
    scenarios: dict[str, ScenarioResult]


class ScenarioResult(BaseModel):
    name: str
    mean: float
    median: float
    std: float
    ci_80: tuple[float, float]
    ci_95: tuple[float, float]
    var_95: float
    cvar_95: float
    prob_loss_20pct: float              # P(loss > 20%)


class SensitivityResult(BaseModel):
    volatility_impact: float            # 0-100 impact score
    drift_impact: float
    lookback_impact: float
    primary_driver: str                 # Most sensitive input
    volatility_range: dict[float, float]
    drift_range: dict[str, float]
    lookback_range: dict[int, float]
```

## Simulation Engine

### GBM Base Case

```python
def _simulate_gbm(
    self,
    S0: float,              # Starting price
    mu: float,              # Annualized drift
    sigma: float,           # Annualized volatility
    days: int,              # Horizon
    n_paths: int,           # Number of simulations
) -> np.ndarray:
    """
    Geometric Brownian Motion: dS = μSdt + σSdW
    Returns array of terminal prices (n_paths,)
    """
    dt = 1/252  # Daily timestep
    Z = np.random.standard_normal((n_paths, days))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    price_paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return price_paths[:, -1]  # Terminal prices
```

### Bootstrap Stress Scenarios

```python
def _simulate_bootstrap(
    self,
    S0: float,
    crisis_returns: np.ndarray,  # Daily returns from crisis period
    days: int,
    n_paths: int,
    block_size: int = 5,         # Preserve autocorrelation
) -> np.ndarray:
    """
    Block bootstrap from actual crisis-period returns.
    """
    n_blocks = days // block_size + 1
    paths = np.zeros((n_paths, days))

    for i in range(n_paths):
        starts = np.random.randint(0, len(crisis_returns) - block_size, n_blocks)
        sampled = np.concatenate([crisis_returns[s:s+block_size] for s in starts])
        paths[i] = sampled[:days]

    return S0 * np.exp(np.cumsum(paths, axis=1))[:, -1]
```

### Adaptive Path Count

```python
def _determine_path_count(self, pilot_results: np.ndarray) -> int:
    """
    Start with 1000 paths. Increase if CI too wide.
    """
    ci_width = np.percentile(pilot_results, 97.5) - np.percentile(pilot_results, 2.5)
    relative_width = ci_width / np.mean(pilot_results)

    if relative_width < 0.15:
        return 1_000
    elif relative_width < 0.25:
        return 10_000
    else:
        return 50_000
```

### Beta Adjustment for Individual Stocks

```python
def adjust_for_stock(
    self,
    crisis_returns: np.ndarray,
    ticker: str,
    session: Session,
) -> np.ndarray:
    """Scale index returns by stock's beta."""
    beta = self._calculate_beta(ticker, session)
    return crisis_returns * beta
```

## Sensitivity Analysis

**Inputs varied:**
- Volatility: ×0.5, ×0.8, ×1.0, ×1.2, ×1.5
- Drift: pessimistic (0%), neutral (historical), optimistic (+2% annual)
- Lookback: 1 year, 3 years, 5 years

**Output:** Impact scores (0-100) showing which assumptions drive the most variance in results.

## Pre-bundled Crisis Data

```
src/investment_monitor/simulation/crisis_data/
├── README.md                    # Data sources, methodology
├── sp500_2008_crisis.csv        # Sep 2008 - Mar 2009
├── sp500_dotcom_crash.csv       # Mar 2000 - Oct 2002
├── sp500_covid_crash.csv        # Feb - Mar 2020
├── sp500_stagflation_1970s.csv  # 1973-1974
├── sp500_black_monday_1987.csv  # Oct 1987
├── sp500_rising_rates_2022.csv  # Jan - Dec 2022
├── regime_democrat_returns.csv  # Aggregated Dem admin returns
├── regime_republican_returns.csv # Aggregated Rep admin returns
└── sector_adjustments.json      # Beta multipliers by sector
```

**CSV format:**
```csv
date,daily_return
2008-09-15,-0.0479
2008-09-16,-0.0115
...
```

Data sourced from Yahoo Finance historical data for ^GSPC.

## CLI Commands

```bash
# Manual simulation (bypasses score threshold)
python -m investment_monitor.cli simulate --ticker AAPL

# Multiple tickers
python -m investment_monitor.cli simulate --tickers AAPL,MSFT,GOOGL

# Auto-simulate all candidates ≥ 80 score
python -m investment_monitor.cli simulate --auto

# View results
python -m investment_monitor.cli simulation-results --ticker AAPL
python -m investment_monitor.cli simulation-results --latest 10

# Override parameters
python -m investment_monitor.cli simulate --ticker AAPL \
    --horizons 30,90 \
    --min-paths 10000 \
    --scenarios base,crisis_2008,covid_crash
```

## Report Integration

New section added to research reports sent via email/Slack:

```markdown
## Risk Analysis (Monte Carlo Simulation)

**Entry Point:** $178.50 | **Simulations:** 10,000 paths | **Data:** 3.2 years

### Projected Price Ranges

| Horizon | Expected | 80% Confidence | Worst 5% |
|---------|----------|----------------|----------|
| 30 days | $182 | $171 - $188 | Below $165 |
| 90 days | $189 | $162 - $201 | Below $152 |
| 1 year | $198 | $149 - $224 | Below $138 |

### Stress Test Results (1-Year Horizon)

| Scenario | Expected | 80% Range | Chance of >20% Loss |
|----------|----------|-----------|---------------------|
| 2008 Financial Crisis | $124 | $98 - $142 | 68% |
| Dot-com Crash | $131 | $105 - $152 | 54% |
| COVID Crash | $156 | $134 - $178 | 31% |
| 1970s Stagflation | $142 | $118 - $162 | 45% |
| Rising Rates (2022) | $148 | $122 - $168 | 38% |
| Black Monday | $139 | $112 - $158 | 52% |
| Regime: Democrat | $185 | $158 - $210 | 12% |
| Regime: Republican | $178 | $152 - $202 | 15% |

### Risk Metrics

- **Value at Risk (95%):** -18.9%
- **Conditional VaR (95%):** -24.2%
- **Base Case Probability of Gain:** 62%

### Sensitivity Check

| Input Assumption | Impact on Results |
|------------------|-------------------|
| Volatility | HIGH — ±16% swing |
| Return Assumption | LOW — ±6% swing |
| Lookback Period | LOW — ±4% swing |

**Bottom Line:** Projections are most sensitive to volatility assumptions.

---
*Simulation based on historical returns. Not a prediction. Past performance ≠ future results.*
```

## Configuration

New section in `config/settings.yaml`:

```yaml
monte_carlo:
  # Gating
  score_threshold: 80

  # Simulation parameters
  horizons: [30, 90, 252]
  min_paths: 1000
  max_paths: 50000
  ci_width_threshold: 0.15

  # Historical data
  min_lookback_days: 252
  max_lookback_days: 1260

  # Sensitivity analysis
  volatility_multipliers: [0.5, 0.8, 1.0, 1.2, 1.5]
  drift_scenarios: ["pessimistic", "neutral", "optimistic"]

  # Scenarios
  scenarios:
    base_gbm: true
    crisis_2008: true
    dotcom_crash: true
    covid_crash: true
    stagflation_1970s: true
    regime_democrat: true
    regime_republican: true
    rising_rates_2022: true
    black_monday_1987: true

  # Report settings
  include_in_reports: true
  disclaimer: "Simulation based on historical returns. Not a prediction."
```

## Files to Create/Modify

### New Files
- `src/investment_monitor/simulation/__init__.py`
- `src/investment_monitor/simulation/analyzer.py` - Core MonteCarloAnalyzer
- `src/investment_monitor/simulation/models.py` - Pydantic models
- `src/investment_monitor/simulation/sensitivity.py` - SensitivityAnalyzer
- `src/investment_monitor/simulation/crisis_loader.py` - CSV loading
- `src/investment_monitor/simulation/crisis_data/` - All CSV files
- `src/investment_monitor/storage/simulation_models.py` - SQLAlchemy model
- `tests/test_simulation/` - Test suite

### Modified Files
- `src/investment_monitor/models/research.py` - Add MonteCarloReportSection
- `src/investment_monitor/analysis/report_generator.py` - Include MC in reports
- `src/investment_monitor/cli.py` - Add simulate commands
- `src/investment_monitor/config.py` - Add MonteCarloConfig
- `config/settings.yaml` - Add monte_carlo section
- `alembic/versions/` - Migration for simulation_results table

## Future Enhancements (Not in Scope)

- QuantLib integration for Sobol sequences (if performance issues arise)
- Portfolio-level simulation (correlated multi-asset)
- Custom scenario builder (user-defined crisis periods)
- Visualization/charts of simulation paths
- Real-time simulation updates as prices change
