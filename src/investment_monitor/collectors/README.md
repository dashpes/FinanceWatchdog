# Collectors Module

Data collection from external sources with rate limiting and retry logic.

## Overview

```
collectors/
├── base.py          # BaseCollector with rate limiting, retry
├── prices.py        # PriceCollector (yfinance)
├── insider.py       # InsiderCollector (SEC EDGAR)
├── news.py          # NewsCollector (RSS feeds)
├── earnings.py      # EarningsCollector (yfinance)
└── etf_holdings.py  # ETFHoldingsCollector
```

## Base Collector

All collectors inherit from `BaseCollector` which provides:

- **Rate limiting**: Sliding window to prevent API abuse
- **Retry with backoff**: Exponential backoff on failures
- **Timing**: Automatic duration tracking
- **Error handling**: Errors logged, don't crash collection

```python
from investment_monitor.collectors import BaseCollector, CollectorResult

class MyCollector(BaseCollector):
    name = "my_collector"
    rate_limit_calls = 30      # Max calls per period
    rate_limit_period = 60     # Period in seconds
    max_retries = 3            # Retry attempts
    retry_delay = 1.0          # Initial retry delay (doubles each retry)

    async def collect(self, tickers: list[str]) -> CollectorResult:
        # Implement batch collection
        ...

    async def collect_single(self, ticker: str) -> int:
        # Implement single ticker collection
        # Returns number of records saved
        ...
```

### CollectorResult

```python
result = CollectorResult(
    collector_name="prices",
    success=True,
    records_collected=15,
    errors=[],
    started_at=datetime(...),
    finished_at=datetime(...)
)

print(result.duration_seconds)  # 2.5
```

## Price Collector

Fetches daily OHLCV data using yfinance.

```python
from investment_monitor.collectors import PriceCollector

collector = PriceCollector(session, settings)

# Batch collection (efficient)
result = await collector.run(["AAPL", "MSFT", "GOOGL"])

# Get price with calculated changes
data = collector.get_price_with_change("AAPL")
# {
#     "ticker": "AAPL",
#     "price": 176.66,
#     "date": date(2026, 1, 28),
#     "daily_change_pct": -3.2,
#     "weekly_change_pct": -5.1,
#     "volume": 82000000,
#     "avg_volume_20d": 65000000,
#     "volume_ratio": 1.26
# }
```

**Rate limit:** 30 calls / 60 seconds

**Data source:** Yahoo Finance via `yfinance` library

**Troubleshooting:**
- "No data for ticker" - Check if market was open, ticker is valid
- Stale data - yfinance caches; data updates after market close
- Rate limited - Reduce batch size or increase delay

## Insider Collector

Fetches SEC Form 4 filings (insider transactions).

```python
from investment_monitor.collectors import InsiderCollector

collector = InsiderCollector(session, settings)
result = await collector.run(["AAPL", "MSFT"])
```

**Rate limit:** 10 calls / 1 second (SEC requirement)

**Data source:** SEC EDGAR RSS feeds and Form 4 XML

**Fields extracted:**
- Owner name and title (CEO, CFO, Director, 10% Owner)
- Transaction type (P=Purchase, S=Sale, M=Option Exercise)
- Shares and price per share
- Total transaction value
- SEC filing URL

**Troubleshooting:**
- "No CIK found" - Ticker may not have SEC filings (foreign stocks)
- Missing transactions - SEC data can be delayed 1-2 days
- Parse errors - Form 4 XML format variations

## News Collector

Aggregates news from RSS feeds.

```python
from investment_monitor.collectors import NewsCollector

# Default feeds
collector = NewsCollector(session, settings)

# Custom feeds
collector = NewsCollector(session, settings, feeds=[
    {"name": "Yahoo", "url": "https://finance.yahoo.com/rss/headline?s={ticker}", "per_ticker": True},
    {"name": "Reuters", "url": "https://www.reuters.com/rss/business", "per_ticker": False},
])

result = await collector.run(["AAPL", "MSFT"])
```

**Rate limit:** 30 calls / 60 seconds

**Feed types:**
- `per_ticker: true` - URL contains `{ticker}` placeholder
- `per_ticker: false` - General feed, matches headlines to tickers

**Ticker matching patterns:**
- `$AAPL` - Cashtag
- `AAPL` - Plain ticker (word boundary)
- `(AAPL)` - Parenthetical

**Troubleshooting:**
- "Feed parse error" - RSS feed may be down or format changed
- Missing news - Check if feed URL is still valid
- Duplicate detection - Uses URL as unique key

## Earnings Collector

Fetches upcoming earnings dates.

```python
from investment_monitor.collectors import EarningsCollector

collector = EarningsCollector(session, settings)
result = await collector.run(["AAPL", "MSFT", "VTI"])

# Get upcoming earnings
upcoming = collector.get_upcoming(["AAPL", "MSFT"], days_ahead=14)
# [
#     {"ticker": "AAPL", "date": date(2026, 2, 2), "days_until": 5},
#     {"ticker": "MSFT", "date": date(2026, 2, 9), "days_until": 12},
# ]
```

**Rate limit:** 30 calls / 60 seconds

**Data source:** yfinance calendar data

**Notes:**
- ETFs return 0 records (no earnings)
- Dates may be estimates until confirmed
- Updates existing records rather than creating duplicates

**Troubleshooting:**
- "No calendar data" - Some tickers don't have earnings info
- Wrong dates - yfinance data can be outdated

## ETF Holdings Collector

Tracks ETF composition changes.

```python
from investment_monitor.collectors import ETFHoldingsCollector

collector = ETFHoldingsCollector(session, settings)
result = await collector.run(["VTI", "AAPL"])  # Skips non-ETFs

# Detect changes
changes = collector.get_holdings_changes("VTI")
# {
#     "added": [{"ticker": "NVDA", "weight": 0.5}],
#     "removed": [{"ticker": "INTC", "weight": 0.3}],
#     "weight_changes": [{"ticker": "AAPL", "old": 5.2, "new": 5.8}]
# }
```

**Known ETFs:** VTI, VOO, SPY, QQQ, VGT, SCHD, VYM, IWM

**Rate limit:** 10 calls / 60 seconds

**Note:** Currently uses simulated data for MVP. Real implementation would fetch from provider APIs (Vanguard, iShares, etc.)

## Running All Collectors

```python
from investment_monitor.collectors import (
    PriceCollector, InsiderCollector, NewsCollector,
    EarningsCollector, ETFHoldingsCollector
)

async def run_all_collectors(session, settings, tickers):
    collectors = [
        PriceCollector(session, settings),
        InsiderCollector(session, settings),
        NewsCollector(session, settings),
        EarningsCollector(session, settings),
        ETFHoldingsCollector(session, settings),
    ]

    results = []
    for collector in collectors:
        try:
            result = await collector.run(tickers)
            print(f"{collector.name}: {result.records_collected} records")
            results.append(result)
        except Exception as e:
            print(f"{collector.name} failed: {e}")

    return results
```

## Extending

### Add New Collector

```python
# collectors/dividends.py
from .base import BaseCollector, CollectorResult

class DividendCollector(BaseCollector):
    name = "dividends"
    rate_limit_calls = 20
    rate_limit_period = 60

    async def collect(self, tickers: list[str]) -> CollectorResult:
        started_at = datetime.now()
        records = 0
        errors = []

        for ticker in tickers:
            await self._rate_limit()
            try:
                records += await self.collect_single(ticker)
            except Exception as e:
                errors.append(f"{ticker}: {e}")

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def collect_single(self, ticker: str) -> int:
        # Fetch dividend data
        # Save to database
        # Return count
        return 1
```

Then add to `__init__.py` exports and `main.py` orchestrator.
