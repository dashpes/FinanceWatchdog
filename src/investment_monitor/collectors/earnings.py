"""Earnings calendar collector using yfinance."""

from datetime import date, datetime, timedelta

import yfinance as yf
from loguru import logger
from sqlalchemy import select

from .base import BaseCollector, CollectorResult
from ..storage import EarningsDate, save_earnings_date, get_upcoming_earnings


class EarningsCollector(BaseCollector):
    """
    Collector for upcoming earnings dates.

    Fetches earnings dates for portfolio tickers using yfinance
    and stores them in the database for alerting.
    """

    name = "earnings"
    rate_limit_calls = 30
    rate_limit_period = 60  # 30 calls per minute

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Fetch earnings dates for all tickers.

        Args:
            tickers: List of ticker symbols to collect earnings dates for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        started_at = datetime.now()
        records = 0
        errors = []

        for ticker in tickers:
            try:
                count = await self.collect_single(ticker)
                records += count
            except Exception as e:
                error_msg = f"{ticker}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Failed to collect earnings for {ticker}: {e}")

        finished_at = datetime.now()
        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Fetch earnings date for a single ticker using yfinance.

        Args:
            ticker: Ticker symbol to collect earnings date for

        Returns:
            Number of earnings dates saved (0 or 1)
        """

        async def fetch_earnings():
            # yfinance is synchronous, but we wrap it for consistency
            yf_ticker = yf.Ticker(ticker)
            calendar = yf_ticker.calendar

            if calendar is None:
                logger.debug(f"{ticker}: No calendar data available (likely an ETF)")
                return None

            # calendar is a dict with 'Earnings Date' as a list
            earnings_dates = calendar.get("Earnings Date", [])

            if not earnings_dates:
                logger.debug(f"{ticker}: No earnings date found")
                return None

            # Take the first (upcoming) earnings date
            earnings_date = earnings_dates[0]

            # Handle if it's already a date object or needs conversion
            if isinstance(earnings_date, datetime):
                earnings_date = earnings_date.date()
            elif isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d").date()

            return earnings_date

        # Use retry with backoff for the API call
        earnings_date = await self._retry_with_backoff(fetch_earnings)

        if earnings_date is None:
            return 0

        # Check if we already have this earnings date
        existing = self.session.scalar(
            select(EarningsDate).where(
                EarningsDate.ticker == ticker,
                EarningsDate.earnings_date == earnings_date,
            )
        )

        if existing:
            # Update the timestamp to show we've re-verified
            existing.updated_at = datetime.now()
            self.session.flush()
            logger.debug(f"{ticker}: Updated existing earnings date {earnings_date}")
            return 0  # Don't count as new record

        # Save new earnings date
        new_earnings = EarningsDate(
            ticker=ticker,
            earnings_date=earnings_date,
            confirmed=False,  # Could be updated via other sources
        )
        save_earnings_date(self.session, new_earnings)
        logger.info(f"{ticker}: Saved earnings date {earnings_date}")
        return 1

    def get_upcoming(self, tickers: list[str], days_ahead: int = 14) -> list[dict]:
        """
        Get earnings within the next N days.

        Args:
            tickers: List of ticker symbols to check
            days_ahead: Number of days to look ahead (default 14)

        Returns:
            List of dicts with ticker, date, and days_until, sorted by date
        """
        today = date.today()
        upcoming_earnings = get_upcoming_earnings(self.session, tickers, days_ahead)

        result = []
        for earnings in upcoming_earnings:
            days_until = (earnings.earnings_date - today).days
            result.append(
                {
                    "ticker": earnings.ticker,
                    "date": earnings.earnings_date.isoformat(),
                    "days_until": days_until,
                }
            )

        # Already sorted by date from database query, but ensure sorting
        result.sort(key=lambda x: x["date"])
        return result
