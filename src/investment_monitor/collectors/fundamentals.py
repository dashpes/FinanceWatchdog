"""Fundamentals collector for fetching financial metrics via yfinance."""

import math
from datetime import datetime
from typing import Any

import yfinance as yf
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import Settings
from .base import BaseCollector, CollectorResult


class FundamentalsData(BaseModel):
    """Fundamental metrics for a stock."""

    ticker: str

    # Valuation metrics
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    peg_ratio: float | None = None

    # Growth metrics
    revenue_growth_yoy: float | None = None  # Year over year
    revenue_growth_3y: float | None = None  # 3-year CAGR
    eps_growth_yoy: float | None = None
    eps_growth_3y: float | None = None

    # Quality metrics
    roe: float | None = None  # Return on equity
    profit_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None

    # Income metrics
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    free_cash_flow: float | None = None

    # Other useful data
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None


class FundamentalsCollector(BaseCollector):
    """
    Collector for fundamental financial metrics using yfinance.

    Fetches:
    - Valuation: P/E, P/B, P/S, PEG
    - Growth: Revenue YoY/3Y, EPS YoY/3Y
    - Quality: ROE, profit margin, debt/equity
    - Income: Dividend yield, payout ratio, FCF

    This collector doesn't persist to database - it returns FundamentalsData
    objects that can be used by the scoring system.
    """

    name = "fundamentals"
    rate_limit_calls = 30  # yfinance rate limit
    rate_limit_period = 60
    max_retries = 3
    retry_delay = 1.0

    def __init__(self, session: Session, config: Settings):
        """
        Initialize the fundamentals collector.

        Args:
            session: SQLAlchemy database session
            config: Application settings
        """
        super().__init__(session, config)
        self._fundamentals_cache: dict[str, FundamentalsData] = {}

    def _safe_get_float(self, info: dict[str, Any], key: str) -> float | None:
        """
        Safely get a float value from yfinance info dict.

        Args:
            info: yfinance info dictionary
            key: Key to retrieve

        Returns:
            Float value if valid, otherwise None
        """
        value = info.get(key)

        # Handle None
        if value is None:
            return None

        # Must be numeric type
        if not isinstance(value, (int, float)):
            return None

        # Handle NaN and Inf
        if math.isnan(value) or math.isinf(value):
            return None

        return float(value)

    def _safe_get_str(self, info: dict[str, Any], key: str) -> str | None:
        """
        Safely get a string value from yfinance info dict.

        Args:
            info: yfinance info dictionary
            key: Key to retrieve

        Returns:
            String value if valid, otherwise None
        """
        value = info.get(key)

        # Handle None
        if value is None:
            return None

        # Must be string type
        if not isinstance(value, str):
            return None

        return value

    def _calculate_cagr(self, values: list[float], years: int) -> float | None:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            values: List of values (oldest to newest)
            years: Number of years

        Returns:
            CAGR as a decimal (e.g., 0.15 for 15%)
        """
        if not values or len(values) < 2:
            return None

        start_value = values[0]
        end_value = values[-1]

        if start_value <= 0 or end_value <= 0:
            return None

        try:
            cagr = (end_value / start_value) ** (1 / years) - 1
            return cagr
        except (ZeroDivisionError, ValueError):
            return None

    def _extract_growth_metrics(
        self, ticker_obj: yf.Ticker
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Extract growth metrics from yfinance financials.

        Args:
            ticker_obj: yfinance Ticker object

        Returns:
            Tuple of (revenue_growth_yoy, revenue_growth_3y, eps_growth_yoy, eps_growth_3y)
        """
        revenue_growth_yoy = None
        revenue_growth_3y = None
        eps_growth_yoy = None
        eps_growth_3y = None

        try:
            # Try to get from info first (pre-calculated)
            info = ticker_obj.info
            revenue_growth_yoy = self._safe_get_float(info, "revenueGrowth")
            eps_growth_yoy = self._safe_get_float(info, "earningsGrowth")

            # Calculate 3-year growth from financials
            financials = ticker_obj.financials

            if financials is not None and not financials.empty:
                # Revenue CAGR (Total Revenue row)
                if "Total Revenue" in financials.index:
                    revenues = financials.loc["Total Revenue"].dropna().tolist()
                    # Reverse to get oldest to newest
                    revenues = list(reversed(revenues))
                    if len(revenues) >= 3:
                        revenue_growth_3y = self._calculate_cagr(revenues[:4], 3)

                # EPS from Net Income / Shares Outstanding
                # This is a simplification; proper EPS would need share count
                if "Net Income" in financials.index:
                    net_incomes = financials.loc["Net Income"].dropna().tolist()
                    net_incomes = list(reversed(net_incomes))
                    if len(net_incomes) >= 3:
                        eps_growth_3y = self._calculate_cagr(
                            [abs(x) for x in net_incomes[:4]], 3
                        )

        except Exception as e:
            logger.debug(f"Error extracting growth metrics: {e}")

        return revenue_growth_yoy, revenue_growth_3y, eps_growth_yoy, eps_growth_3y

    async def get_fundamentals(self, ticker: str) -> FundamentalsData:
        """
        Fetch fundamental data for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalsData with all available metrics
        """

        def _fetch_fundamentals() -> FundamentalsData:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Extract growth metrics
            (
                revenue_growth_yoy,
                revenue_growth_3y,
                eps_growth_yoy,
                eps_growth_3y,
            ) = self._extract_growth_metrics(ticker_obj)

            return FundamentalsData(
                ticker=ticker.upper(),
                # Valuation metrics
                pe_ratio=self._safe_get_float(info, "trailingPE"),
                pb_ratio=self._safe_get_float(info, "priceToBook"),
                ps_ratio=self._safe_get_float(info, "priceToSalesTrailing12Months"),
                peg_ratio=self._safe_get_float(info, "pegRatio"),
                # Growth metrics
                revenue_growth_yoy=revenue_growth_yoy,
                revenue_growth_3y=revenue_growth_3y,
                eps_growth_yoy=eps_growth_yoy,
                eps_growth_3y=eps_growth_3y,
                # Quality metrics
                roe=self._safe_get_float(info, "returnOnEquity"),
                profit_margin=self._safe_get_float(info, "profitMargins"),
                debt_to_equity=self._safe_get_float(info, "debtToEquity"),
                current_ratio=self._safe_get_float(info, "currentRatio"),
                # Income metrics
                dividend_yield=self._safe_get_float(info, "dividendYield"),
                payout_ratio=self._safe_get_float(info, "payoutRatio"),
                free_cash_flow=self._safe_get_float(info, "freeCashflow"),
                # Other data
                market_cap=self._safe_get_float(info, "marketCap"),
                sector=self._safe_get_str(info, "sector"),
                industry=self._safe_get_str(info, "industry"),
            )

        return await self._retry_with_backoff(_fetch_fundamentals)

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect fundamentals for all tickers.

        Args:
            tickers: List of ticker symbols to collect data for

        Returns:
            CollectorResult with collection statistics
        """
        started_at = datetime.now()
        errors: list[str] = []
        records_collected = 0

        for ticker in tickers:
            try:
                fundamentals = await self.get_fundamentals(ticker)
                self._fundamentals_cache[ticker.upper()] = fundamentals
                records_collected += 1
                logger.debug(f"Collected fundamentals for {ticker}")
            except Exception as e:
                error_msg = f"Failed to collect fundamentals for {ticker}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        finished_at = datetime.now()

        logger.info(
            f"Fundamentals collection complete: {records_collected}/{len(tickers)} tickers"
        )

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Collect fundamentals for a single ticker.

        Args:
            ticker: Ticker symbol to collect data for

        Returns:
            Number of records saved (1 on success, 0 on failure)
        """
        try:
            fundamentals = await self.get_fundamentals(ticker)
            self._fundamentals_cache[ticker.upper()] = fundamentals
            return 1
        except Exception as e:
            logger.error(f"Failed to collect fundamentals for {ticker}: {e}")
            raise

    def get_cached_fundamentals(self, ticker: str) -> FundamentalsData | None:
        """
        Get cached fundamentals for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Cached FundamentalsData or None if not cached
        """
        return self._fundamentals_cache.get(ticker.upper())

    def get_all_cached_fundamentals(self) -> dict[str, FundamentalsData]:
        """
        Get all cached fundamentals data.

        Returns:
            Dictionary mapping ticker symbols to FundamentalsData
        """
        return self._fundamentals_cache.copy()

    def clear_cache(self) -> None:
        """Clear the fundamentals cache."""
        self._fundamentals_cache.clear()
