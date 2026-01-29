"""Base collector class with standardized interface for all data collectors."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, TypeVar
import asyncio
import inspect
import time

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import Settings


class CollectorResult(BaseModel):
    """Result of a collector run containing metrics and status."""

    collector_name: str
    success: bool
    records_collected: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        """Calculate the duration of the collection run in seconds."""
        return (self.finished_at - self.started_at).total_seconds()

    def __str__(self) -> str:
        """Human-readable representation of the result."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"{self.collector_name}: {status} - "
            f"{self.records_collected} records in {self.duration_seconds:.2f}s"
        )


class CollectorError(Exception):
    """Base exception for collector errors."""

    pass


class RateLimitExceededError(CollectorError):
    """Raised when rate limit is exceeded and cannot proceed."""

    pass


class MaxRetriesExceededError(CollectorError):
    """Raised when maximum retries have been exhausted."""

    pass


T = TypeVar("T")


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Provides standardized interface with built-in:
    - Rate limiting to prevent exceeding API limits
    - Retry logic with exponential backoff
    - Logging integration
    - Error handling that doesn't crash the collector

    Subclasses must implement:
    - collect(): Collect data for multiple tickers
    - collect_single(): Collect data for a single ticker

    Class attributes can be overridden by subclasses:
    - name: Identifier for the collector
    - rate_limit_calls: Maximum calls allowed per period
    - rate_limit_period: Period in seconds for rate limiting
    - max_retries: Maximum retry attempts for failed requests
    - retry_delay: Initial delay between retries (doubles each retry)
    """

    name: str = "base"
    rate_limit_calls: int = 60  # Calls per minute
    rate_limit_period: int = 60  # Period in seconds
    max_retries: int = 3
    retry_delay: float = 1.0

    def __init__(self, session: Session, config: Settings):
        """
        Initialize the collector.

        Args:
            session: SQLAlchemy database session for persisting data
            config: Application settings
        """
        self.session = session
        self.config = config
        self._call_times: list[float] = []
        self._lock = asyncio.Lock()

    async def _rate_limit(self) -> None:
        """
        Enforce rate limiting by tracking call times and sleeping if needed.

        Uses a sliding window approach to ensure we don't exceed
        rate_limit_calls within any rate_limit_period window.
        """
        async with self._lock:
            current_time = time.monotonic()

            # Remove call times outside the current window
            window_start = current_time - self.rate_limit_period
            self._call_times = [t for t in self._call_times if t > window_start]

            # Check if we need to wait
            if len(self._call_times) >= self.rate_limit_calls:
                # Calculate how long to wait until the oldest call exits the window
                oldest_call = min(self._call_times)
                wait_time = oldest_call + self.rate_limit_period - current_time

                if wait_time > 0:
                    logger.debug(
                        f"{self.name}: Rate limit reached, waiting {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)
                    # Recalculate after waiting
                    current_time = time.monotonic()
                    window_start = current_time - self.rate_limit_period
                    self._call_times = [t for t in self._call_times if t > window_start]

            # Record this call
            self._call_times.append(current_time)

    async def _retry_with_backoff(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Retry a function with exponential backoff on failure.

        Args:
            func: The async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            MaxRetriesExceededError: If all retries are exhausted
        """
        last_exception: Exception | None = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting before each attempt
                await self._rate_limit()

                # Call the function (handle both sync and async)
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"{self.name}: Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"{self.name}: All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )

        raise MaxRetriesExceededError(
            f"Max retries ({self.max_retries}) exceeded for {self.name}"
        ) from last_exception

    @abstractmethod
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect data for given tickers.

        Must be implemented by subclasses to define the collection logic
        for multiple tickers.

        Args:
            tickers: List of ticker symbols to collect data for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        ...

    @abstractmethod
    async def collect_single(self, ticker: str) -> int:
        """
        Collect data for a single ticker.

        Must be implemented by subclasses to define the collection logic
        for a single ticker.

        Args:
            ticker: Ticker symbol to collect data for

        Returns:
            Number of records saved
        """
        ...

    async def run(self, tickers: list[str]) -> CollectorResult:
        """
        Run the collector with error handling and timing.

        This is the main entry point for running a collection. It wraps
        the collect() method with timing, error handling, and logging.

        Args:
            tickers: List of ticker symbols to collect data for

        Returns:
            CollectorResult with complete run information
        """
        started_at = datetime.now()
        errors: list[str] = []
        records_collected = 0
        success = True

        logger.info(f"{self.name}: Starting collection for {len(tickers)} tickers")

        try:
            result = await self.collect(tickers)
            records_collected = result.records_collected
            errors = result.errors
            success = result.success

        except Exception as e:
            logger.exception(f"{self.name}: Collection failed with unexpected error")
            errors.append(f"Unexpected error: {str(e)}")
            success = False

        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        result = CollectorResult(
            collector_name=self.name,
            success=success,
            records_collected=records_collected,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        if success:
            logger.info(
                f"{self.name}: Completed successfully - "
                f"{records_collected} records in {duration:.2f}s"
            )
        else:
            logger.error(
                f"{self.name}: Completed with errors - "
                f"{records_collected} records, {len(errors)} errors in {duration:.2f}s"
            )

        return result

    def get_rate_limit_status(self) -> dict[str, Any]:
        """
        Get the current rate limit status.

        Returns:
            Dictionary with calls_in_window and calls_remaining
        """
        current_time = time.monotonic()
        window_start = current_time - self.rate_limit_period
        calls_in_window = len([t for t in self._call_times if t > window_start])

        return {
            "calls_in_window": calls_in_window,
            "calls_remaining": max(0, self.rate_limit_calls - calls_in_window),
            "window_seconds": self.rate_limit_period,
        }
