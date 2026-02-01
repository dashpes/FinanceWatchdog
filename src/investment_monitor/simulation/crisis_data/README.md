# Crisis Data for Monte Carlo Simulation

This directory contains historical S&P 500 return data for various crisis and regime scenarios
used in Monte Carlo stress testing simulations.

## Data Source

All data is sourced from Yahoo Finance (^GSPC - S&P 500 Index) using the `yfinance` library.

## Data Format

Each CSV file contains two columns:
- `date`: Trading date in YYYY-MM-DD format
- `daily_return`: Daily log return calculated as ln(P_t / P_{t-1})

Log returns are used because they are additive across time and have better statistical properties
for simulation purposes.

## Included Scenarios

### Crisis Scenarios

| Scenario | File | Period | Description |
|----------|------|--------|-------------|
| 2008 Financial Crisis | `sp500_2008_crisis.csv` | Sep 2008 - Mar 2009 | Global financial crisis triggered by Lehman Brothers collapse |
| Dot-com Crash | `sp500_dotcom_crash.csv` | Mar 2000 - Oct 2002 | Technology bubble burst |
| COVID-19 Crash | `sp500_covid_crash.csv` | Feb 2020 - Mar 2020 | Pandemic-induced market crash |
| 1970s Stagflation | `sp500_stagflation_1970s.csv` | Jan 1973 - Dec 1974 | Oil crisis and stagflation period |
| Black Monday 1987 | `sp500_black_monday_1987.csv` | Oct 1987 | Single-day market crash (-22.6%) |
| Rising Rates 2022 | `sp500_rising_rates_2022.csv` | Jan 2022 - Dec 2022 | Federal Reserve rate hiking cycle |

### Political Regime Scenarios

| Scenario | File | Periods | Description |
|----------|------|---------|-------------|
| Democrat Returns | `regime_democrat_returns.csv` | Clinton, Obama, Biden | S&P 500 returns during Democratic administrations |
| Republican Returns | `regime_republican_returns.csv` | Bush Sr, Bush Jr, Trump | S&P 500 returns during Republican administrations |

## Regenerating Data

To regenerate the CSV files with fresh data from Yahoo Finance:

```bash
python scripts/fetch_crisis_data.py
```

## Usage

The `CrisisDataLoader` class in `crisis_loader.py` provides a clean interface for loading
this data in simulations:

```python
from investment_monitor.simulation import CrisisDataLoader, CrisisScenario

loader = CrisisDataLoader()
returns = loader.load_crisis_returns(CrisisScenario.CRISIS_2008)
```

## Notes

- The regime scenarios combine multiple presidential terms for each party
- Log returns can be converted to simple returns using: simple_return = exp(log_return) - 1
- For beta-adjusted scenarios, multiply returns by the stock's beta coefficient
