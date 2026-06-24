"""Price collector using yfinance to fetch daily OHLCV data."""

from datetime import date, datetime, timedelta

import yfinance as yf
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import Settings
from ..storage import InsiderTransaction, Price, get_prices, price_exists, save_price
from .base import BaseCollector, CollectorResult


class PriceCollector(BaseCollector):
    """
    Collector for daily price and volume data using yfinance.

    Fetches OHLCV (Open, High, Low, Close, Volume) data and stores
    it in the database while avoiding duplicates.
    """

    name = "prices"
    rate_limit_calls = 30
    rate_limit_period = 60

    def __init__(
        self,
        session: Session,
        config: Settings,
        days_to_fetch: int = 30,
    ):
        """
        Initialize the price collector.

        Args:
            session: SQLAlchemy database session
            config: Application settings
            days_to_fetch: Number of days of historical data to fetch
        """
        super().__init__(session, config)
        self.days_to_fetch = days_to_fetch

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Fetch prices for all tickers using batch request.

        Uses yf.download() for efficient batch fetching of multiple tickers.

        Args:
            tickers: List of ticker symbols to collect data for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        started_at = datetime.now()
        records_collected = 0
        errors: list[str] = []

        if not tickers:
            return CollectorResult(
                collector_name=self.name,
                success=True,
                records_collected=0,
                errors=[],
                started_at=started_at,
                finished_at=datetime.now(),
            )

        try:
            # Use batch download for efficiency
            records_collected = await self._batch_fetch(tickers, errors)
        except Exception as e:
            logger.exception(f"{self.name}: Batch fetch failed")
            errors.append(f"Batch fetch error: {str(e)}")

        finished_at = datetime.now()

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    # ----------------------------------------------------------------------- #
    # Broad, universe-independent collection (confluence-relevant tickers only)
    # ----------------------------------------------------------------------- #
    async def collect_all(
        self, *, window_days: int = 30, max_tickers: int | None = None
    ) -> CollectorResult:
        """Collect daily OHLCV for the CONFLUENCE-RELEVANT universe (broad collection).

        Unlike ``collect(tickers)``, this does NOT take a configured universe and does
        NOT pull all ~8000 market tickers. It fetches prices for exactly the DISTINCT
        tickers that recently showed up in ``insider_transactions`` (trade within the
        last ``window_days``) — i.e. the names the insight engine actually needs price
        context for (volume-spike + price confluence against insider clusters).

        Reuses the existing per-ticker fetch path (``_batch_fetch``), which dedups via
        Price's (ticker, date) unique constraint (skips existing), commits once, and
        fails open per ticker.

        Args:
            window_days: insider trade_date lookback that defines the relevant universe.
            max_tickers: optional cap on the number of tickers fetched this run.
        """
        started_at = datetime.now()
        records_collected = 0
        errors: list[str] = []

        tickers = self._relevant_tickers(window_days=window_days, max_tickers=max_tickers)
        logger.info(
            f"{self.name}: {len(tickers)} confluence-relevant tickers to price "
            f"(insider trades within {window_days}d)"
        )

        if not tickers:
            return CollectorResult(
                collector_name=self.name,
                success=True,
                records_collected=0,
                errors=[],
                started_at=started_at,
                finished_at=datetime.now(),
            )

        try:
            # _batch_fetch already skips existing (ticker, date) rows and commits once;
            # per-ticker errors are appended without aborting the run (fail-open).
            records_collected = await self._batch_fetch(tickers, errors)
        except Exception as e:  # noqa: BLE001 - a batch failure must not crash broad collection
            logger.exception(f"{self.name}: Broad batch fetch failed")
            errors.append(f"Broad batch fetch error: {str(e)}")

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    def _relevant_tickers(
        self, *, window_days: int = 30, max_tickers: int | None = None
    ) -> list[str]:
        """DISTINCT tickers in insider_transactions with trade_date within window_days.

        This is the confluence-relevant universe — we only need price/volume context for
        names insiders are actually trading, NOT the whole market. Newest activity first
        so ``max_tickers`` keeps the most recent names when capped.
        """
        cutoff = date.today() - timedelta(days=window_days)
        stmt = (
            select(InsiderTransaction.ticker)
            .where(InsiderTransaction.trade_date >= cutoff)
            .group_by(InsiderTransaction.ticker)
            .order_by(func.max(InsiderTransaction.trade_date).desc())
        )
        if max_tickers is not None:
            stmt = stmt.limit(max_tickers)
        return [t for t in self.session.scalars(stmt) if t]

    async def _batch_fetch(
        self, tickers: list[str], errors: list[str]
    ) -> int:
        """
        Fetch prices for multiple tickers in a batch.

        Args:
            tickers: List of ticker symbols
            errors: List to append error messages to

        Returns:
            Number of records saved
        """
        # Apply rate limiting
        await self._rate_limit()

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=self.days_to_fetch)

        logger.debug(
            f"{self.name}: Fetching {len(tickers)} tickers from {start_date} to {end_date}"
        )

        # Download data for all tickers at once
        try:
            data = yf.download(
                tickers=tickers,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"{self.name}: yfinance download failed: {e}")
            errors.append(f"yfinance download error: {str(e)}")
            return 0

        if data.empty:
            logger.warning(f"{self.name}: No data returned for tickers")
            return 0

        records_saved = 0

        # Handle single ticker case (no multi-level columns)
        if len(tickers) == 1:
            ticker = tickers[0]
            try:
                saved = self._process_ticker_data(ticker, data)
                records_saved += saved
            except Exception as e:
                logger.error(f"{self.name}: Error processing {ticker}: {e}")
                errors.append(f"{ticker}: {str(e)}")
        else:
            # Handle multiple tickers (multi-level columns)
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        saved = self._process_ticker_data(ticker, ticker_data)
                        records_saved += saved
                    else:
                        logger.warning(f"{self.name}: No data for {ticker}")
                        errors.append(f"{ticker}: No data available")
                except Exception as e:
                    logger.error(f"{self.name}: Error processing {ticker}: {e}")
                    errors.append(f"{ticker}: {str(e)}")

        # Commit all saved records
        self.session.commit()

        return records_saved

    def _process_ticker_data(self, ticker: str, data) -> int:
        """
        Process and save price data for a single ticker.

        Args:
            ticker: Ticker symbol
            data: DataFrame with OHLCV data

        Returns:
            Number of records saved
        """
        records_saved = 0

        # Iterate through each row of data
        for idx, row in data.iterrows():
            # Get the date from the index
            if hasattr(idx, "date"):
                price_date = idx.date()
            else:
                price_date = idx

            # Skip if this price already exists
            if price_exists(self.session, ticker, price_date):
                continue

            # Handle NaN values - skip rows with no close price
            close_price = row.get("Close")
            if close_price is None or (hasattr(close_price, "__float__") and str(close_price) == "nan"):
                continue

            # Create price record
            price = Price(
                ticker=ticker,
                date=price_date,
                open=self._safe_float(row.get("Open")),
                high=self._safe_float(row.get("High")),
                low=self._safe_float(row.get("Low")),
                close=float(close_price),
                volume=self._safe_int(row.get("Volume")),
            )

            save_price(self.session, price)
            records_saved += 1

        return records_saved

    def _safe_float(self, value) -> float | None:
        """Convert value to float, returning None for NaN."""
        if value is None:
            return None
        try:
            float_val = float(value)
            import math
            if math.isnan(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value) -> int | None:
        """Convert value to int, returning None for NaN."""
        if value is None:
            return None
        try:
            float_val = float(value)
            import math
            if math.isnan(float_val):
                return None
            return int(float_val)
        except (ValueError, TypeError):
            return None

    async def collect_single(self, ticker: str) -> int:
        """
        Fetch price data for a single ticker.

        Args:
            ticker: Ticker symbol to collect data for

        Returns:
            Number of records saved
        """
        # Apply rate limiting
        await self._rate_limit()

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=self.days_to_fetch)

        logger.debug(f"{self.name}: Fetching {ticker} from {start_date} to {end_date}")

        # Download data for single ticker
        data = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
        )

        if data.empty:
            logger.warning(f"{self.name}: No data for {ticker}")
            return 0

        records_saved = self._process_ticker_data(ticker, data)
        self.session.commit()

        return records_saved

    def get_price_with_change(self, ticker: str) -> dict | None:
        """
        Get latest price with daily/weekly change calculations.

        Calculates percentage changes and volume comparisons for
        price-based alerts.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with price data and derived metrics, or None if no data:
            {
                "ticker": "AAPL",
                "price": 176.66,
                "daily_change_pct": -3.2,
                "weekly_change_pct": -5.1,
                "volume": 82_000_000,
                "avg_volume_20d": 65_000_000,
            }
        """
        # Get enough price history for calculations
        prices = get_prices(self.session, ticker, days=30)

        if not prices:
            return None

        # Prices are sorted by date descending
        latest = prices[0]

        # Calculate daily change
        daily_change_pct = None
        if len(prices) >= 2:
            previous = prices[1]
            if previous.close and previous.close != 0:
                daily_change_pct = (
                    (latest.close - previous.close) / previous.close
                ) * 100

        # Calculate weekly change (5 trading days)
        weekly_change_pct = None
        if len(prices) >= 6:
            week_ago = prices[5]
            if week_ago.close and week_ago.close != 0:
                weekly_change_pct = (
                    (latest.close - week_ago.close) / week_ago.close
                ) * 100

        # Calculate 20-day average volume
        avg_volume_20d = None
        volume_prices = [p for p in prices if p.volume is not None]
        if len(volume_prices) >= 5:  # Need at least some data
            volumes = [p.volume for p in volume_prices[:20]]
            avg_volume_20d = sum(volumes) / len(volumes)

        return {
            "ticker": ticker,
            "price": latest.close,
            "date": latest.date,
            "daily_change_pct": round(daily_change_pct, 2) if daily_change_pct is not None else None,
            "weekly_change_pct": round(weekly_change_pct, 2) if weekly_change_pct is not None else None,
            "volume": latest.volume,
            "avg_volume_20d": int(avg_volume_20d) if avg_volume_20d is not None else None,
        }
