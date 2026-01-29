"""ETF Holdings Collector for tracking ETF holdings changes."""

from datetime import date, datetime, timedelta

import httpx
from loguru import logger

from .base import BaseCollector, CollectorResult
from ..storage import ETFHolding, get_etf_holdings, save_etf_holdings


class ETFHoldingsCollector(BaseCollector):
    """
    Collector for ETF holdings data.

    Tracks ETF holdings to detect when funds add or drop positions.
    For MVP, uses simulated holdings data. In production, this would
    connect to providers like Vanguard, iShares, etc.
    """

    name = "etf_holdings"
    rate_limit_calls = 10
    rate_limit_period = 60

    # Known ETF tickers (common ones)
    KNOWN_ETFS = {"VTI", "VOO", "SPY", "QQQ", "VGT", "SCHD", "VYM", "IWM"}

    # Simulated holdings data for MVP
    # In production, this would be fetched from provider APIs
    SIMULATED_HOLDINGS: dict[str, list[dict]] = {
        "VTI": [
            {"ticker": "AAPL", "weight": 6.5, "shares": 1000000},
            {"ticker": "MSFT", "weight": 5.8, "shares": 800000},
            {"ticker": "GOOGL", "weight": 3.2, "shares": 400000},
            {"ticker": "AMZN", "weight": 2.9, "shares": 350000},
            {"ticker": "NVDA", "weight": 2.5, "shares": 300000},
            {"ticker": "META", "weight": 1.8, "shares": 250000},
            {"ticker": "TSLA", "weight": 1.5, "shares": 200000},
            {"ticker": "BRK.B", "weight": 1.4, "shares": 180000},
            {"ticker": "UNH", "weight": 1.2, "shares": 150000},
            {"ticker": "JNJ", "weight": 1.1, "shares": 140000},
        ],
        "VOO": [
            {"ticker": "AAPL", "weight": 7.0, "shares": 1200000},
            {"ticker": "MSFT", "weight": 6.2, "shares": 900000},
            {"ticker": "GOOGL", "weight": 3.5, "shares": 450000},
            {"ticker": "AMZN", "weight": 3.1, "shares": 380000},
            {"ticker": "NVDA", "weight": 2.7, "shares": 320000},
            {"ticker": "META", "weight": 2.0, "shares": 270000},
            {"ticker": "TSLA", "weight": 1.6, "shares": 220000},
            {"ticker": "BRK.B", "weight": 1.5, "shares": 190000},
            {"ticker": "UNH", "weight": 1.3, "shares": 160000},
            {"ticker": "JNJ", "weight": 1.2, "shares": 150000},
        ],
        "QQQ": [
            {"ticker": "AAPL", "weight": 12.0, "shares": 2000000},
            {"ticker": "MSFT", "weight": 10.5, "shares": 1500000},
            {"ticker": "GOOGL", "weight": 7.2, "shares": 900000},
            {"ticker": "AMZN", "weight": 6.5, "shares": 800000},
            {"ticker": "NVDA", "weight": 5.5, "shares": 650000},
            {"ticker": "META", "weight": 4.2, "shares": 550000},
            {"ticker": "TSLA", "weight": 3.5, "shares": 450000},
            {"ticker": "AVGO", "weight": 2.8, "shares": 350000},
            {"ticker": "COST", "weight": 2.5, "shares": 300000},
            {"ticker": "ADBE", "weight": 2.2, "shares": 280000},
        ],
        "VGT": [
            {"ticker": "AAPL", "weight": 18.0, "shares": 3000000},
            {"ticker": "MSFT", "weight": 16.5, "shares": 2500000},
            {"ticker": "NVDA", "weight": 8.5, "shares": 1000000},
            {"ticker": "AVGO", "weight": 4.2, "shares": 500000},
            {"ticker": "ADBE", "weight": 3.5, "shares": 400000},
            {"ticker": "CRM", "weight": 3.2, "shares": 380000},
            {"ticker": "CSCO", "weight": 2.8, "shares": 350000},
            {"ticker": "ACN", "weight": 2.5, "shares": 320000},
            {"ticker": "INTC", "weight": 2.2, "shares": 300000},
            {"ticker": "AMD", "weight": 2.0, "shares": 280000},
        ],
    }

    # Threshold for reporting weight changes (in percentage points)
    WEIGHT_CHANGE_THRESHOLD = 0.5

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Fetch holdings for ETF tickers only, skip non-ETFs.

        Args:
            tickers: List of ticker symbols (will filter for ETFs only)

        Returns:
            CollectorResult with collection metrics
        """
        started_at = datetime.now()
        etf_tickers = [t for t in tickers if self._is_etf(t)]

        if not etf_tickers:
            logger.info(f"{self.name}: No ETF tickers found in input list")
            return CollectorResult(
                collector_name=self.name,
                success=True,
                records_collected=0,
                errors=[],
                started_at=started_at,
                finished_at=datetime.now(),
            )

        logger.info(f"{self.name}: Processing {len(etf_tickers)} ETFs: {etf_tickers}")

        records_collected = 0
        errors: list[str] = []

        for etf_ticker in etf_tickers:
            try:
                count = await self._retry_with_backoff(
                    self.collect_single, etf_ticker
                )
                records_collected += count
            except Exception as e:
                error_msg = f"{etf_ticker}: {str(e)}"
                logger.error(f"{self.name}: {error_msg}")
                errors.append(error_msg)

        finished_at = datetime.now()
        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, etf_ticker: str) -> int:
        """
        Collect holdings data for a single ETF.

        1. Determine provider from ticker (simulated for MVP)
        2. Fetch holdings data
        3. Parse into standard format
        4. Save to database

        Args:
            etf_ticker: ETF ticker symbol

        Returns:
            Number of holdings records saved
        """
        logger.debug(f"{self.name}: Fetching holdings for {etf_ticker}")

        # For MVP, use simulated data
        # In production, this would call provider APIs
        holdings_data = await self._fetch_holdings(etf_ticker)

        if not holdings_data:
            logger.warning(f"{self.name}: No holdings data found for {etf_ticker}")
            return 0

        # Parse into ETFHolding model objects
        today = date.today()
        holdings = []

        for holding in holdings_data:
            etf_holding = ETFHolding(
                etf_ticker=etf_ticker,
                holding_ticker=holding["ticker"],
                shares=holding.get("shares"),
                weight_pct=holding.get("weight"),
                as_of_date=today,
            )
            holdings.append(etf_holding)

        # Save to database
        count = save_etf_holdings(self.session, holdings)
        self.session.commit()

        logger.info(f"{self.name}: Saved {count} holdings for {etf_ticker}")
        return count

    async def _fetch_holdings(self, etf_ticker: str) -> list[dict]:
        """
        Fetch holdings data for an ETF.

        For MVP, returns simulated data. In production, this would:
        - Determine the provider (Vanguard, iShares, etc.)
        - Call the appropriate API
        - Parse the response

        Args:
            etf_ticker: ETF ticker symbol

        Returns:
            List of holdings dictionaries with ticker, weight, shares
        """
        # Return simulated data for known ETFs
        if etf_ticker in self.SIMULATED_HOLDINGS:
            return self.SIMULATED_HOLDINGS[etf_ticker]

        # For unknown ETFs, return empty (would be API call in production)
        logger.debug(f"{self.name}: No simulated data for {etf_ticker}")
        return []

    def _is_etf(self, ticker: str) -> bool:
        """
        Check if ticker is a known ETF.

        Args:
            ticker: Ticker symbol to check

        Returns:
            True if ticker is a known ETF
        """
        return ticker in self.KNOWN_ETFS

    def get_holdings_changes(self, etf_ticker: str) -> dict:
        """
        Compare current holdings to previous day.

        Detects:
        - Added positions (new holdings not in previous day)
        - Removed positions (previous holdings no longer present)
        - Significant weight changes (above threshold)

        Args:
            etf_ticker: ETF ticker to analyze

        Returns:
            Dictionary with added, removed, and weight_changes lists
        """
        # Get current holdings (most recent date)
        current_holdings = get_etf_holdings(self.session, etf_ticker)

        if not current_holdings:
            logger.warning(f"{self.name}: No current holdings found for {etf_ticker}")
            return {
                "added": [],
                "removed": [],
                "weight_changes": [],
            }

        # Get the current date from holdings
        current_date = current_holdings[0].as_of_date

        # Get previous day's holdings
        previous_date = current_date - timedelta(days=1)
        previous_holdings = get_etf_holdings(
            self.session, etf_ticker, as_of_date=previous_date
        )

        if not previous_holdings:
            # No previous data to compare
            logger.debug(
                f"{self.name}: No previous holdings for {etf_ticker} on {previous_date}"
            )
            return {
                "added": [],
                "removed": [],
                "weight_changes": [],
            }

        # Build dictionaries for comparison
        current_by_ticker = {
            h.holding_ticker: h.weight_pct for h in current_holdings
        }
        previous_by_ticker = {
            h.holding_ticker: h.weight_pct for h in previous_holdings
        }

        current_tickers = set(current_by_ticker.keys())
        previous_tickers = set(previous_by_ticker.keys())

        # Find added positions
        added = [
            {"ticker": ticker, "weight": current_by_ticker[ticker]}
            for ticker in (current_tickers - previous_tickers)
        ]

        # Find removed positions
        removed = [
            {"ticker": ticker, "weight": previous_by_ticker[ticker]}
            for ticker in (previous_tickers - current_tickers)
        ]

        # Find significant weight changes
        weight_changes = []
        common_tickers = current_tickers & previous_tickers

        for ticker in common_tickers:
            old_weight = previous_by_ticker[ticker]
            new_weight = current_by_ticker[ticker]

            if old_weight is None or new_weight is None:
                continue

            change = abs(new_weight - old_weight)
            if change >= self.WEIGHT_CHANGE_THRESHOLD:
                weight_changes.append({
                    "ticker": ticker,
                    "old": old_weight,
                    "new": new_weight,
                })

        logger.info(
            f"{self.name}: {etf_ticker} changes - "
            f"added: {len(added)}, removed: {len(removed)}, "
            f"weight changes: {len(weight_changes)}"
        )

        return {
            "added": added,
            "removed": removed,
            "weight_changes": weight_changes,
        }

    def get_all_changes(self, etf_tickers: list[str] | None = None) -> dict[str, dict]:
        """
        Get holdings changes for multiple ETFs.

        Args:
            etf_tickers: List of ETF tickers to analyze (defaults to KNOWN_ETFS)

        Returns:
            Dictionary mapping ETF ticker to its changes
        """
        if etf_tickers is None:
            etf_tickers = list(self.KNOWN_ETFS)

        all_changes = {}
        for etf_ticker in etf_tickers:
            if self._is_etf(etf_ticker):
                all_changes[etf_ticker] = self.get_holdings_changes(etf_ticker)

        return all_changes
