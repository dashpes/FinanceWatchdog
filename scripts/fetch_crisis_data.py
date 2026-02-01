#!/usr/bin/env python3
"""
One-time script to fetch historical crisis data from Yahoo Finance.

This script fetches S&P 500 (^GSPC) price data for specific historical periods
and calculates daily log returns, saving them to CSV files for use in
Monte Carlo simulations.

Usage:
    python scripts/fetch_crisis_data.py

The script generates CSV files in src/investment_monitor/simulation/crisis_data/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# Directory where CSV files will be saved
OUTPUT_DIR = Path(__file__).parent.parent / "src/investment_monitor/simulation/crisis_data"


# Crisis scenario definitions
# Each tuple: (filename, description, start_date, end_date)
CRISIS_SCENARIOS = {
    "sp500_2008_crisis": {
        "description": "2008 Financial Crisis",
        "start": "2008-09-01",
        "end": "2009-03-31",
        "ticker": "^GSPC",
    },
    "sp500_dotcom_crash": {
        "description": "Dot-com Crash",
        "start": "2000-03-01",
        "end": "2002-10-31",
        "ticker": "^GSPC",
    },
    "sp500_covid_crash": {
        "description": "COVID-19 Crash",
        "start": "2020-02-01",
        "end": "2020-03-31",
        "ticker": "^GSPC",
    },
    "sp500_stagflation_1970s": {
        "description": "1970s Stagflation",
        "start": "1973-01-01",
        "end": "1974-12-31",
        "ticker": "^GSPC",
    },
    "sp500_black_monday_1987": {
        "description": "Black Monday 1987",
        "start": "1987-10-01",
        "end": "1987-10-31",
        "ticker": "^GSPC",
    },
    "sp500_rising_rates_2022": {
        "description": "Rising Rates 2022",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "ticker": "^GSPC",
    },
}

# Presidential administration periods for regime analysis
# Data source: Historical S&P 500 returns during different administrations
REGIME_PERIODS = {
    "regime_democrat_returns": {
        "description": "Democrat Administration Returns (Clinton, Obama, Biden)",
        "periods": [
            # Clinton: Jan 20, 1993 - Jan 20, 2001
            ("1993-01-20", "2001-01-19"),
            # Obama: Jan 20, 2009 - Jan 20, 2017
            ("2009-01-20", "2017-01-19"),
            # Biden: Jan 20, 2021 - present (use data until end of 2024)
            ("2021-01-20", "2024-12-31"),
        ],
        "ticker": "^GSPC",
    },
    "regime_republican_returns": {
        "description": "Republican Administration Returns (Bush Sr, Bush Jr, Trump)",
        "periods": [
            # Bush Sr: Jan 20, 1989 - Jan 20, 1993
            ("1989-01-20", "1993-01-19"),
            # Bush Jr: Jan 20, 2001 - Jan 20, 2009
            ("2001-01-20", "2009-01-19"),
            # Trump: Jan 20, 2017 - Jan 20, 2021
            ("2017-01-20", "2021-01-19"),
        ],
        "ticker": "^GSPC",
    },
}


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance."""
    print(f"  Fetching {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")
    return data


def calculate_log_returns(prices: pd.DataFrame) -> pd.Series:
    """Calculate daily log returns from closing prices."""
    close = prices["Close"]
    # Handle MultiIndex columns from yfinance
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    log_returns = np.log(close / close.shift(1))
    return log_returns.dropna()


def save_returns_csv(returns: pd.Series, filename: str, description: str) -> int:
    """Save log returns to CSV file."""
    filepath = OUTPUT_DIR / f"{filename}.csv"

    df = pd.DataFrame({
        "date": returns.index.strftime("%Y-%m-%d"),
        "daily_return": returns.values
    })

    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} rows to {filepath.name}")
    return len(df)


def fetch_crisis_scenarios() -> dict[str, int]:
    """Fetch and save all crisis scenarios."""
    row_counts = {}

    print("\nFetching crisis scenarios...")
    print("-" * 50)

    for scenario_name, config in CRISIS_SCENARIOS.items():
        print(f"\n{config['description']} ({scenario_name}):")
        try:
            prices = fetch_price_data(config["ticker"], config["start"], config["end"])
            returns = calculate_log_returns(prices)
            rows = save_returns_csv(returns, scenario_name, config["description"])
            row_counts[scenario_name] = rows
        except Exception as e:
            print(f"  ERROR: {e}")
            row_counts[scenario_name] = 0

    return row_counts


def fetch_regime_scenarios() -> dict[str, int]:
    """Fetch and save regime (political) scenarios."""
    row_counts = {}

    print("\nFetching regime scenarios...")
    print("-" * 50)

    for scenario_name, config in REGIME_PERIODS.items():
        print(f"\n{config['description']} ({scenario_name}):")

        all_returns = []

        for start, end in config["periods"]:
            try:
                prices = fetch_price_data(config["ticker"], start, end)
                returns = calculate_log_returns(prices)
                all_returns.append(returns)
            except Exception as e:
                print(f"  WARNING: Failed to fetch {start} to {end}: {e}")

        if all_returns:
            combined_returns = pd.concat(all_returns)
            combined_returns = combined_returns.sort_index()
            rows = save_returns_csv(combined_returns, scenario_name, config["description"])
            row_counts[scenario_name] = rows
        else:
            print(f"  ERROR: No data collected for {scenario_name}")
            row_counts[scenario_name] = 0

    return row_counts


def main():
    """Main entry point."""
    print("=" * 60)
    print("Crisis Data Fetcher for Monte Carlo Simulation")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch all scenarios
    crisis_counts = fetch_crisis_scenarios()
    regime_counts = fetch_regime_scenarios()

    # Summary
    all_counts = {**crisis_counts, **regime_counts}

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_rows = 0
    for scenario, count in sorted(all_counts.items()):
        status = "OK" if count > 0 else "FAILED"
        print(f"  {scenario}: {count} rows [{status}]")
        total_rows += count

    print(f"\nTotal: {len(all_counts)} scenarios, {total_rows} data points")

    # Check for failures
    failures = [s for s, c in all_counts.items() if c == 0]
    if failures:
        print(f"\nWARNING: {len(failures)} scenario(s) failed to fetch data")
        return 1

    print("\nAll scenarios fetched successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
