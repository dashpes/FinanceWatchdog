"""Tests for the collectors module."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
import time

import pytest
from sqlalchemy.orm import Session

from investment_monitor.collectors.base import (
    BaseCollector,
    CollectorResult,
    CollectorError,
    MaxRetriesExceededError,
    RateLimitExceededError,
)
from investment_monitor.config import Settings


class MockCollector(BaseCollector):
    """Mock collector for testing the base class."""

    name = "mock"
    rate_limit_calls = 5
    rate_limit_period = 1  # 1 second for faster tests
    max_retries = 2
    retry_delay = 0.1  # Short delay for faster tests

    def __init__(self, session: Session, config: Settings):
        super().__init__(session, config)
        self.collect_called = False
        self.collect_single_called = False
        self.collect_single_results: dict[str, int] = {}
        self.collect_single_errors: dict[str, Exception] = {}

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """Collect data for given tickers."""
        self.collect_called = True
        started_at = datetime.now()
        records = 0
        errors = []

        for ticker in tickers:
            try:
                count = await self.collect_single(ticker)
                records += count
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")

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
        """Collect data for a single ticker."""
        self.collect_single_called = True

        # Check if we should raise an error for this ticker
        if ticker in self.collect_single_errors:
            raise self.collect_single_errors[ticker]

        # Return configured result or default
        return self.collect_single_results.get(ticker, 10)


class FailingCollector(BaseCollector):
    """Collector that fails a configurable number of times before succeeding."""

    name = "failing"
    max_retries = 3
    retry_delay = 0.05

    def __init__(self, session: Session, config: Settings, failures_before_success: int = 2):
        super().__init__(session, config)
        self.failures_before_success = failures_before_success
        self.attempt_count = 0

    async def collect(self, tickers: list[str]) -> CollectorResult:
        started_at = datetime.now()
        records = 0
        for ticker in tickers:
            records += await self.collect_single(ticker)
        finished_at = datetime.now()
        return CollectorResult(
            collector_name=self.name,
            success=True,
            records_collected=records,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        self.attempt_count += 1
        if self.attempt_count <= self.failures_before_success:
            raise ConnectionError(f"Simulated failure {self.attempt_count}")
        return 5


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_config():
    """Create a mock settings object."""
    return Settings()


@pytest.fixture
def mock_collector(mock_session, mock_config):
    """Create a mock collector instance."""
    return MockCollector(mock_session, mock_config)


# ============================================================================
# CollectorResult Tests
# ============================================================================


class TestCollectorResult:
    """Tests for CollectorResult model."""

    def test_duration_seconds_calculation(self):
        """Should calculate duration correctly."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        finished = datetime(2024, 1, 1, 12, 0, 5)

        result = CollectorResult(
            collector_name="test",
            success=True,
            records_collected=100,
            started_at=started,
            finished_at=finished,
        )

        assert result.duration_seconds == 5.0

    def test_str_representation_success(self):
        """Should format success message correctly."""
        result = CollectorResult(
            collector_name="test_collector",
            success=True,
            records_collected=50,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            finished_at=datetime(2024, 1, 1, 12, 0, 2),
        )

        assert "test_collector" in str(result)
        assert "SUCCESS" in str(result)
        assert "50 records" in str(result)
        assert "2.00s" in str(result)

    def test_str_representation_failure(self):
        """Should format failure message correctly."""
        result = CollectorResult(
            collector_name="test_collector",
            success=False,
            records_collected=10,
            errors=["Error 1", "Error 2"],
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            finished_at=datetime(2024, 1, 1, 12, 0, 3),
        )

        assert "test_collector" in str(result)
        assert "FAILED" in str(result)

    def test_default_errors_list(self):
        """Should default to empty errors list."""
        result = CollectorResult(
            collector_name="test",
            success=True,
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )

        assert result.errors == []
        assert result.records_collected == 0


# ============================================================================
# BaseCollector Initialization Tests
# ============================================================================


class TestBaseCollectorInit:
    """Tests for BaseCollector initialization."""

    def test_initialization(self, mock_session, mock_config):
        """Should initialize with session and config."""
        collector = MockCollector(mock_session, mock_config)

        assert collector.session == mock_session
        assert collector.config == mock_config
        assert collector._call_times == []

    def test_class_attributes(self, mock_collector):
        """Should have correct class attributes."""
        assert mock_collector.name == "mock"
        assert mock_collector.rate_limit_calls == 5
        assert mock_collector.rate_limit_period == 1
        assert mock_collector.max_retries == 2


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    async def test_rate_limit_tracks_calls(self, mock_collector):
        """Should track call times."""
        await mock_collector._rate_limit()
        await mock_collector._rate_limit()

        assert len(mock_collector._call_times) == 2

    async def test_rate_limit_allows_within_limit(self, mock_collector):
        """Should allow calls within rate limit."""
        # Make calls up to the limit
        for _ in range(mock_collector.rate_limit_calls):
            await mock_collector._rate_limit()

        # All calls should be tracked
        assert len(mock_collector._call_times) == mock_collector.rate_limit_calls

    async def test_rate_limit_enforces_limit(self, mock_collector):
        """Should enforce rate limit by sleeping."""
        # Fill up the rate limit window
        for _ in range(mock_collector.rate_limit_calls):
            await mock_collector._rate_limit()

        # Next call should wait
        start = time.monotonic()
        await mock_collector._rate_limit()
        elapsed = time.monotonic() - start

        # Should have waited (at least some time, allowing for timing variance)
        # Since window is 1 second and we filled it, we need to wait
        assert elapsed >= 0.5  # At least half a second wait

    async def test_rate_limit_window_cleanup(self, mock_collector):
        """Should clean up old calls outside the window."""
        # Make some calls
        await mock_collector._rate_limit()
        await mock_collector._rate_limit()

        # Wait for window to pass
        await asyncio.sleep(mock_collector.rate_limit_period + 0.1)

        # Make another call - old ones should be cleaned
        await mock_collector._rate_limit()

        # Only the new call should be in the window
        assert len(mock_collector._call_times) == 1

    def test_get_rate_limit_status(self, mock_collector):
        """Should return correct rate limit status."""
        status = mock_collector.get_rate_limit_status()

        assert "calls_in_window" in status
        assert "calls_remaining" in status
        assert "window_seconds" in status
        assert status["calls_remaining"] == mock_collector.rate_limit_calls

    async def test_get_rate_limit_status_after_calls(self, mock_collector):
        """Should update status after calls."""
        await mock_collector._rate_limit()
        await mock_collector._rate_limit()

        status = mock_collector.get_rate_limit_status()

        assert status["calls_in_window"] == 2
        assert status["calls_remaining"] == mock_collector.rate_limit_calls - 2


# ============================================================================
# Retry with Backoff Tests
# ============================================================================


class TestRetryWithBackoff:
    """Tests for retry with exponential backoff."""

    async def test_retry_success_first_attempt(self, mock_collector):
        """Should succeed on first attempt without retry."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await mock_collector._retry_with_backoff(success_func)

        assert result == "success"
        assert call_count == 1

    async def test_retry_success_after_failures(self, mock_collector):
        """Should succeed after initial failures."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await mock_collector._retry_with_backoff(fail_then_succeed)

        assert result == "success"
        assert call_count == 2

    async def test_retry_exhausted(self, mock_collector):
        """Should raise MaxRetriesExceededError when retries exhausted."""

        async def always_fail():
            raise ConnectionError("Permanent failure")

        with pytest.raises(MaxRetriesExceededError):
            await mock_collector._retry_with_backoff(always_fail)

    async def test_retry_exponential_backoff_timing(self, mock_session, mock_config):
        """Should use exponential backoff between retries."""
        collector = MockCollector(mock_session, mock_config)
        collector.retry_delay = 0.1
        collector.max_retries = 2

        call_times = []

        async def track_and_fail():
            call_times.append(time.monotonic())
            raise ConnectionError("Failure")

        start = time.monotonic()
        with pytest.raises(MaxRetriesExceededError):
            await collector._retry_with_backoff(track_and_fail)
        total_time = time.monotonic() - start

        # Should have 3 attempts (initial + 2 retries)
        assert len(call_times) == 3

        # Total time should include backoff delays: 0.1 + 0.2 = 0.3s minimum
        assert total_time >= 0.25  # Allow some margin

    async def test_retry_with_sync_function(self, mock_collector):
        """Should handle synchronous functions."""

        def sync_func():
            return "sync_result"

        result = await mock_collector._retry_with_backoff(sync_func)
        assert result == "sync_result"

    async def test_retry_with_args_and_kwargs(self, mock_collector):
        """Should pass arguments to the function."""

        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await mock_collector._retry_with_backoff(func_with_args, "x", "y", c="z")
        assert result == "x-y-z"


# ============================================================================
# Collect Methods Tests
# ============================================================================


class TestCollectMethods:
    """Tests for collect and collect_single methods."""

    async def test_collect_single_success(self, mock_collector):
        """Should collect data for a single ticker."""
        mock_collector.collect_single_results["AAPL"] = 15

        result = await mock_collector.collect_single("AAPL")

        assert result == 15
        assert mock_collector.collect_single_called

    async def test_collect_single_with_error(self, mock_collector):
        """Should raise error for failing ticker."""
        mock_collector.collect_single_errors["FAIL"] = ValueError("API Error")

        with pytest.raises(ValueError, match="API Error"):
            await mock_collector.collect_single("FAIL")

    async def test_collect_multiple_tickers(self, mock_collector):
        """Should collect data for multiple tickers."""
        mock_collector.collect_single_results["AAPL"] = 10
        mock_collector.collect_single_results["GOOGL"] = 20

        result = await mock_collector.collect(["AAPL", "GOOGL"])

        assert result.records_collected == 30
        assert result.success
        assert len(result.errors) == 0

    async def test_collect_partial_failure(self, mock_collector):
        """Should handle partial failures gracefully."""
        mock_collector.collect_single_results["AAPL"] = 10
        mock_collector.collect_single_errors["FAIL"] = ValueError("API Error")

        result = await mock_collector.collect(["AAPL", "FAIL"])

        assert result.records_collected == 10
        assert not result.success
        assert len(result.errors) == 1
        assert "FAIL" in result.errors[0]


# ============================================================================
# Run Method Tests
# ============================================================================


class TestRunMethod:
    """Tests for the run() method."""

    async def test_run_success(self, mock_collector):
        """Should run collector successfully."""
        mock_collector.collect_single_results["AAPL"] = 5
        mock_collector.collect_single_results["GOOGL"] = 5

        result = await mock_collector.run(["AAPL", "GOOGL"])

        assert result.success
        assert result.records_collected == 10
        assert result.collector_name == "mock"
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.duration_seconds >= 0

    async def test_run_with_errors(self, mock_collector):
        """Should handle errors without crashing."""
        mock_collector.collect_single_errors["FAIL"] = ValueError("Test error")

        result = await mock_collector.run(["FAIL"])

        assert not result.success
        assert len(result.errors) == 1

    async def test_run_with_unexpected_exception(self, mock_session, mock_config):
        """Should catch unexpected exceptions in run()."""

        class CrashingCollector(MockCollector):
            async def collect(self, tickers):
                raise RuntimeError("Unexpected crash!")

        collector = CrashingCollector(mock_session, mock_config)
        result = await collector.run(["AAPL"])

        assert not result.success
        assert any("Unexpected error" in e for e in result.errors)

    async def test_run_timing(self, mock_collector):
        """Should record accurate timing."""
        result = await mock_collector.run(["AAPL"])

        assert result.started_at <= result.finished_at
        assert result.duration_seconds >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    async def test_retry_with_rate_limiting(self, mock_session, mock_config):
        """Should combine retry and rate limiting correctly."""
        collector = FailingCollector(mock_session, mock_config, failures_before_success=2)
        collector.rate_limit_calls = 10
        collector.rate_limit_period = 1

        async def failing_operation():
            return await collector.collect_single("TEST")

        result = await collector._retry_with_backoff(failing_operation)

        assert result == 5
        assert collector.attempt_count == 3  # 2 failures + 1 success

    async def test_full_collection_workflow(self, mock_collector):
        """Should handle full collection workflow."""
        # Setup
        mock_collector.collect_single_results = {
            "AAPL": 5,
            "GOOGL": 10,
            "MSFT": 8,
        }

        # Run
        result = await mock_collector.run(["AAPL", "GOOGL", "MSFT"])

        # Verify
        assert result.success
        assert result.records_collected == 23
        assert result.collector_name == "mock"
        assert len(result.errors) == 0

    async def test_concurrent_rate_limiting(self, mock_session, mock_config):
        """Should handle concurrent access to rate limiter."""
        collector = MockCollector(mock_session, mock_config)
        collector.rate_limit_calls = 3
        collector.rate_limit_period = 0.5

        # Run multiple concurrent rate limit checks
        tasks = [collector._rate_limit() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have tracked all calls and enforced limits
        status = collector.get_rate_limit_status()
        assert status["calls_in_window"] <= collector.rate_limit_calls + 1


# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_collector_error_is_exception(self):
        """CollectorError should be an Exception."""
        assert issubclass(CollectorError, Exception)

    def test_rate_limit_exceeded_error(self):
        """RateLimitExceededError should be a CollectorError."""
        assert issubclass(RateLimitExceededError, CollectorError)
        error = RateLimitExceededError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"

    def test_max_retries_exceeded_error(self):
        """MaxRetriesExceededError should be a CollectorError."""
        assert issubclass(MaxRetriesExceededError, CollectorError)
        error = MaxRetriesExceededError("Max retries exceeded")
        assert str(error) == "Max retries exceeded"
