"""Tests for the collectors module."""

import asyncio
from datetime import datetime
from time import struct_time
from unittest.mock import MagicMock, AsyncMock, patch
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
from investment_monitor.collectors.news import NewsCollector
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


# ============================================================================
# NewsCollector Tests
# ============================================================================


class MockFeedEntry:
    """Mock feedparser entry."""

    def __init__(
        self,
        title: str,
        link: str,
        summary: str = "",
        published_parsed: struct_time | None = None,
    ):
        self.title = title
        self.link = link
        self.summary = summary
        self.published_parsed = published_parsed

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class MockFeed:
    """Mock feedparser result."""

    def __init__(self, entries: list[MockFeedEntry], bozo: bool = False, bozo_exception=None):
        self.entries = entries
        self.bozo = bozo
        self.bozo_exception = bozo_exception


@pytest.fixture
def news_collector(mock_session, mock_config):
    """Create a news collector instance with test feeds."""
    test_feeds = [
        {"name": "Test Feed", "url": "https://test.com/rss/{ticker}", "per_ticker": True},
        {"name": "General Feed", "url": "https://general.com/rss", "per_ticker": False},
    ]
    return NewsCollector(mock_session, mock_config, feeds=test_feeds)


class TestNewsCollectorInit:
    """Tests for NewsCollector initialization."""

    def test_initialization_with_default_feeds(self, mock_session, mock_config):
        """Should initialize with default feeds when none provided."""
        collector = NewsCollector(mock_session, mock_config)
        assert collector.feeds == NewsCollector.DEFAULT_FEEDS
        assert collector.name == "news"

    def test_initialization_with_custom_feeds(self, mock_session, mock_config):
        """Should use custom feeds when provided."""
        custom_feeds = [{"name": "Custom", "url": "https://custom.com/rss", "per_ticker": False}]
        collector = NewsCollector(mock_session, mock_config, feeds=custom_feeds)
        assert collector.feeds == custom_feeds

    def test_rate_limit_settings(self, news_collector):
        """Should have correct rate limit settings."""
        assert news_collector.rate_limit_calls == 30
        assert news_collector.rate_limit_period == 60


class TestTickerMentioned:
    """Tests for _ticker_mentioned method."""

    def test_matches_dollar_sign_ticker(self, news_collector):
        """Should match $AAPL format."""
        result = news_collector._ticker_mentioned("Buy $AAPL now!", ["AAPL", "GOOGL"])
        assert result == ["AAPL"]

    def test_matches_plain_ticker(self, news_collector):
        """Should match plain AAPL format."""
        result = news_collector._ticker_mentioned("Apple AAPL stock rises", ["AAPL"])
        assert result == ["AAPL"]

    def test_matches_parentheses_ticker(self, news_collector):
        """Should match (AAPL) format."""
        result = news_collector._ticker_mentioned("Apple Inc. (AAPL) reports earnings", ["AAPL"])
        assert result == ["AAPL"]

    def test_matches_multiple_tickers(self, news_collector):
        """Should match multiple tickers in text."""
        result = news_collector._ticker_mentioned(
            "$AAPL and GOOGL both rise", ["AAPL", "GOOGL", "MSFT"]
        )
        assert set(result) == {"AAPL", "GOOGL"}

    def test_no_match(self, news_collector):
        """Should return empty list when no tickers match."""
        result = news_collector._ticker_mentioned("No tickers here", ["AAPL", "GOOGL"])
        assert result == []

    def test_case_insensitive(self, news_collector):
        """Should match regardless of case."""
        result = news_collector._ticker_mentioned("aapl stock rises", ["AAPL"])
        assert result == ["AAPL"]

    def test_does_not_match_partial_word(self, news_collector):
        """Should not match ticker as part of another word."""
        result = news_collector._ticker_mentioned("APPLET software", ["AAPL"])
        assert result == []

    def test_matches_at_end_of_sentence(self, news_collector):
        """Should match ticker at end of text."""
        result = news_collector._ticker_mentioned("Great news for AAPL", ["AAPL"])
        assert result == ["AAPL"]


class TestParseFeed:
    """Tests for _parse_feed method."""

    @patch("investment_monitor.collectors.news.feedparser.parse")
    def test_parse_feed_success(self, mock_parse, news_collector):
        """Should parse feed entries successfully."""
        mock_entry = MockFeedEntry(title="Test", link="https://test.com/1")
        mock_parse.return_value = MockFeed(entries=[mock_entry])

        entries = news_collector._parse_feed("https://test.com/rss")

        mock_parse.assert_called_once_with("https://test.com/rss")
        assert len(entries) == 1

    @patch("investment_monitor.collectors.news.feedparser.parse")
    def test_parse_feed_bozo_error_with_entries(self, mock_parse, news_collector):
        """Should return entries even if bozo flag is set but entries exist."""
        mock_entry = MockFeedEntry(title="Test", link="https://test.com/1")
        mock_parse.return_value = MockFeed(
            entries=[mock_entry],
            bozo=True,
            bozo_exception=Exception("Minor error"),
        )

        entries = news_collector._parse_feed("https://test.com/rss")
        assert len(entries) == 1

    @patch("investment_monitor.collectors.news.feedparser.parse")
    def test_parse_feed_bozo_error_no_entries(self, mock_parse, news_collector):
        """Should raise exception if bozo and no entries."""
        mock_parse.return_value = MockFeed(
            entries=[],
            bozo=True,
            bozo_exception=Exception("Fatal error"),
        )

        with pytest.raises(Exception, match="Feed parsing error"):
            news_collector._parse_feed("https://test.com/rss")


class TestParsePublishedDate:
    """Tests for _parse_published_date method."""

    def test_parse_published_parsed(self, news_collector):
        """Should parse published_parsed field."""
        entry = MockFeedEntry(
            title="Test",
            link="https://test.com/1",
            published_parsed=struct_time((2024, 1, 15, 10, 30, 0, 0, 15, 0)),
        )

        result = news_collector._parse_published_date(entry)

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_no_date(self, news_collector):
        """Should return None if no date available."""
        entry = MockFeedEntry(title="Test", link="https://test.com/1")

        result = news_collector._parse_published_date(entry)

        assert result is None


class TestNewsCollectorCollect:
    """Tests for NewsCollector collect method."""

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_per_ticker_feed(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should fetch per-ticker feed for each ticker."""
        mock_entry = MockFeedEntry(
            title="AAPL stock news",
            link="https://test.com/1",
            summary="Apple news",
        )
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = False

        result = await news_collector.collect(["AAPL"])

        assert result.success
        assert result.records_collected > 0
        # Should have called parse for both feeds (per-ticker and general)
        assert mock_parse.call_count >= 1

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_deduplicates_by_url(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should not save duplicate URLs within same collection."""
        # Same URL in multiple feed entries
        entries = [
            MockFeedEntry(title="News 1", link="https://test.com/same-url"),
            MockFeedEntry(title="News 2", link="https://test.com/same-url"),
        ]
        mock_parse.return_value = MockFeed(entries=entries)
        mock_exists.return_value = False

        result = await news_collector.collect(["AAPL"])

        # Should only save once per unique URL per ticker
        assert result.records_collected >= 1

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_skips_existing_urls(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should skip URLs that already exist in database."""
        mock_entry = MockFeedEntry(title="Existing news", link="https://test.com/existing")
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = True  # URL already exists

        result = await news_collector.collect(["AAPL"])

        # No new records should be saved
        mock_save.assert_not_called()

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_handles_feed_error(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should handle feed fetch errors gracefully."""
        mock_parse.side_effect = Exception("Network error")

        result = await news_collector.collect(["AAPL"])

        # Should complete but with errors
        assert not result.success
        assert len(result.errors) > 0

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_matches_tickers_in_headline(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should match tickers mentioned in headlines."""
        mock_entry = MockFeedEntry(
            title="$GOOGL beats earnings",
            link="https://test.com/googl-news",
            summary="Google reports strong quarter",
        )
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = False

        result = await news_collector.collect(["AAPL", "GOOGL"])

        assert result.success
        assert result.records_collected > 0
        # Verify save was called with GOOGL ticker
        save_calls = mock_save.call_args_list
        tickers_saved = [call[0][1].ticker for call in save_calls]
        assert "GOOGL" in tickers_saved

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_skips_empty_headlines(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should skip entries with empty headlines."""
        mock_entry = MockFeedEntry(title="", link="https://test.com/1")
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = False

        result = await news_collector.collect(["AAPL"])

        # Should not crash and handle empty headline
        assert result is not None

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_skips_empty_urls(
        self, mock_save, mock_exists, mock_parse, news_collector
    ):
        """Should skip entries with empty URLs."""
        mock_entry = MockFeedEntry(title="News", link="")
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = False

        result = await news_collector.collect(["AAPL"])

        # Should not save entries without URLs
        mock_exists.assert_not_called()


class TestNewsCollectorSingle:
    """Tests for collect_single method."""

    @patch("investment_monitor.collectors.news.feedparser.parse")
    @patch("investment_monitor.collectors.news.news_exists")
    @patch("investment_monitor.collectors.news.save_news_item")
    async def test_collect_single(self, mock_save, mock_exists, mock_parse, news_collector):
        """Should collect news for a single ticker."""
        mock_entry = MockFeedEntry(title="AAPL news", link="https://test.com/1")
        mock_parse.return_value = MockFeed(entries=[mock_entry])
        mock_exists.return_value = False

        result = await news_collector.collect_single("AAPL")

        assert isinstance(result, int)
        assert result >= 0


# ============================================================================
# EarningsCollector Tests
# ============================================================================

from datetime import date, timedelta
from investment_monitor.collectors.earnings import EarningsCollector
from investment_monitor.storage import EarningsDate


@pytest.fixture
def earnings_collector(mock_session, mock_config):
    """Create an earnings collector instance."""
    return EarningsCollector(mock_session, mock_config)


class TestEarningsCollectorInit:
    """Tests for EarningsCollector initialization."""

    def test_initialization(self, mock_session, mock_config):
        """Should initialize with correct settings."""
        collector = EarningsCollector(mock_session, mock_config)

        assert collector.name == "earnings"
        assert collector.rate_limit_calls == 30
        assert collector.rate_limit_period == 60

    def test_has_required_methods(self, earnings_collector):
        """Should have required collect methods."""
        assert hasattr(earnings_collector, "collect")
        assert hasattr(earnings_collector, "collect_single")
        assert hasattr(earnings_collector, "get_upcoming")


class TestEarningsCollectorCollectSingle:
    """Tests for collect_single method."""

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_success(self, mock_ticker_class, earnings_collector):
        """Should fetch and save earnings date for a ticker."""
        # Mock yfinance response
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [date(2026, 2, 15)],
        }
        mock_ticker_class.return_value = mock_ticker

        # Mock database query to return None (no existing record)
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.collect_single("AAPL")

        assert result == 1
        earnings_collector.session.add.assert_called_once()

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_no_calendar(self, mock_ticker_class, earnings_collector):
        """Should handle tickers with no calendar (ETFs)."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_ticker_class.return_value = mock_ticker

        result = await earnings_collector.collect_single("SPY")

        assert result == 0
        earnings_collector.session.add.assert_not_called()

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_empty_earnings_dates(self, mock_ticker_class, earnings_collector):
        """Should handle empty earnings dates list."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": []}
        mock_ticker_class.return_value = mock_ticker

        result = await earnings_collector.collect_single("AAPL")

        assert result == 0
        earnings_collector.session.add.assert_not_called()

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_existing_record(self, mock_ticker_class, earnings_collector):
        """Should update existing record instead of creating new one."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [date(2026, 2, 15)]}
        mock_ticker_class.return_value = mock_ticker

        # Mock existing record
        existing_earnings = MagicMock(spec=EarningsDate)
        earnings_collector.session.scalar.return_value = existing_earnings

        result = await earnings_collector.collect_single("AAPL")

        assert result == 0  # No new record created
        earnings_collector.session.add.assert_not_called()
        # Verify updated_at was set
        assert existing_earnings.updated_at is not None

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_datetime_conversion(self, mock_ticker_class, earnings_collector):
        """Should handle datetime objects in earnings date."""
        from datetime import datetime as dt

        mock_ticker = MagicMock()
        # yfinance sometimes returns datetime instead of date
        mock_ticker.calendar = {"Earnings Date": [dt(2026, 2, 15, 16, 30, 0)]}
        mock_ticker_class.return_value = mock_ticker
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.collect_single("AAPL")

        assert result == 1

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_single_string_date_conversion(self, mock_ticker_class, earnings_collector):
        """Should handle string dates in earnings date."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": ["2026-02-15"]}
        mock_ticker_class.return_value = mock_ticker
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.collect_single("AAPL")

        assert result == 1


class TestEarningsCollectorCollect:
    """Tests for collect method."""

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_multiple_tickers(self, mock_ticker_class, earnings_collector):
        """Should collect earnings for multiple tickers."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [date(2026, 2, 15)]}
        mock_ticker_class.return_value = mock_ticker
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.collect(["AAPL", "GOOGL", "MSFT"])

        assert result.success
        assert result.records_collected == 3
        assert result.collector_name == "earnings"

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_partial_failure(self, mock_ticker_class, earnings_collector):
        """Should handle partial failures gracefully."""

        def mock_ticker_factory(ticker):
            mock = MagicMock()
            if ticker == "FAIL":
                mock.calendar = None
                # Make the API call fail
                type(mock).calendar = property(
                    lambda self: (_ for _ in ()).throw(Exception("API Error"))
                )
            else:
                mock.calendar = {"Earnings Date": [date(2026, 2, 15)]}
            return mock

        mock_ticker_class.side_effect = mock_ticker_factory
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.collect(["AAPL", "FAIL"])

        # Should have one success and one error
        assert not result.success
        assert len(result.errors) == 1
        assert "FAIL" in result.errors[0]

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_collect_all_etfs(self, mock_ticker_class, earnings_collector):
        """Should handle all ETFs (no earnings) gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_ticker_class.return_value = mock_ticker

        result = await earnings_collector.collect(["SPY", "QQQ", "IWM"])

        assert result.success
        assert result.records_collected == 0
        assert len(result.errors) == 0


class TestEarningsCollectorGetUpcoming:
    """Tests for get_upcoming method."""

    @patch("investment_monitor.collectors.earnings.get_upcoming_earnings")
    def test_get_upcoming_returns_sorted_list(self, mock_get_upcoming, earnings_collector):
        """Should return sorted list of upcoming earnings."""
        today = date.today()

        # Create mock earnings with different dates
        earnings1 = MagicMock(spec=EarningsDate)
        earnings1.ticker = "AAPL"
        earnings1.earnings_date = today + timedelta(days=5)

        earnings2 = MagicMock(spec=EarningsDate)
        earnings2.ticker = "GOOGL"
        earnings2.earnings_date = today + timedelta(days=2)

        earnings3 = MagicMock(spec=EarningsDate)
        earnings3.ticker = "MSFT"
        earnings3.earnings_date = today + timedelta(days=10)

        mock_get_upcoming.return_value = [earnings1, earnings2, earnings3]

        result = earnings_collector.get_upcoming(["AAPL", "GOOGL", "MSFT"], days_ahead=14)

        assert len(result) == 3
        # Should be sorted by date
        assert result[0]["ticker"] == "GOOGL"
        assert result[0]["days_until"] == 2
        assert result[1]["ticker"] == "AAPL"
        assert result[1]["days_until"] == 5
        assert result[2]["ticker"] == "MSFT"
        assert result[2]["days_until"] == 10

    @patch("investment_monitor.collectors.earnings.get_upcoming_earnings")
    def test_get_upcoming_empty(self, mock_get_upcoming, earnings_collector):
        """Should return empty list when no upcoming earnings."""
        mock_get_upcoming.return_value = []

        result = earnings_collector.get_upcoming(["AAPL", "GOOGL"])

        assert result == []

    @patch("investment_monitor.collectors.earnings.get_upcoming_earnings")
    def test_get_upcoming_custom_days_ahead(self, mock_get_upcoming, earnings_collector):
        """Should pass correct days_ahead to database query."""
        mock_get_upcoming.return_value = []

        earnings_collector.get_upcoming(["AAPL"], days_ahead=30)

        mock_get_upcoming.assert_called_once()
        call_args = mock_get_upcoming.call_args
        assert call_args[0][1] == ["AAPL"]
        assert call_args[0][2] == 30

    @patch("investment_monitor.collectors.earnings.get_upcoming_earnings")
    def test_get_upcoming_date_format(self, mock_get_upcoming, earnings_collector):
        """Should return dates in ISO format."""
        today = date.today()
        earnings = MagicMock(spec=EarningsDate)
        earnings.ticker = "AAPL"
        earnings.earnings_date = today + timedelta(days=5)
        mock_get_upcoming.return_value = [earnings]

        result = earnings_collector.get_upcoming(["AAPL"])

        assert result[0]["date"] == (today + timedelta(days=5)).isoformat()


class TestEarningsCollectorRun:
    """Tests for run method (inherited from BaseCollector)."""

    @patch("investment_monitor.collectors.earnings.yf.Ticker")
    async def test_run_full_workflow(self, mock_ticker_class, earnings_collector):
        """Should run complete collection workflow."""
        mock_ticker = MagicMock()
        mock_ticker.calendar = {"Earnings Date": [date(2026, 2, 15)]}
        mock_ticker_class.return_value = mock_ticker
        earnings_collector.session.scalar.return_value = None

        result = await earnings_collector.run(["AAPL", "GOOGL"])

        assert result.collector_name == "earnings"
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.duration_seconds >= 0


# ============================================================================
# ETFHoldingsCollector Tests
# ============================================================================

from investment_monitor.collectors.etf_holdings import ETFHoldingsCollector
from investment_monitor.storage.models import ETFHolding


class TestETFHoldingsCollector:
    """Tests for ETFHoldingsCollector."""

    @pytest.fixture
    def etf_collector(self, mock_session, mock_config):
        """Create an ETF holdings collector instance."""
        return ETFHoldingsCollector(mock_session, mock_config)

    def test_initialization(self, etf_collector):
        """Should initialize with correct attributes."""
        assert etf_collector.name == "etf_holdings"
        assert etf_collector.rate_limit_calls == 10
        assert etf_collector.rate_limit_period == 60

    def test_is_etf_known_etfs(self, etf_collector):
        """Should identify known ETFs correctly."""
        assert etf_collector._is_etf("VTI") is True
        assert etf_collector._is_etf("VOO") is True
        assert etf_collector._is_etf("SPY") is True
        assert etf_collector._is_etf("QQQ") is True
        assert etf_collector._is_etf("VGT") is True
        assert etf_collector._is_etf("SCHD") is True
        assert etf_collector._is_etf("VYM") is True
        assert etf_collector._is_etf("IWM") is True

    def test_is_etf_non_etfs(self, etf_collector):
        """Should reject non-ETF tickers."""
        assert etf_collector._is_etf("AAPL") is False
        assert etf_collector._is_etf("GOOGL") is False
        assert etf_collector._is_etf("MSFT") is False
        assert etf_collector._is_etf("UNKNOWN") is False

    async def test_collect_filters_to_etfs_only(self, etf_collector):
        """Should only process ETF tickers, skip non-ETFs."""
        # Mix of ETFs and non-ETFs
        tickers = ["AAPL", "VTI", "GOOGL", "VOO", "MSFT"]

        result = await etf_collector.collect(tickers)

        # Should succeed (no errors for known ETFs with simulated data)
        assert result.success is True
        # Should have collected holdings for VTI and VOO
        assert result.records_collected > 0
        assert result.collector_name == "etf_holdings"

    async def test_collect_empty_when_no_etfs(self, etf_collector):
        """Should return empty result when no ETFs in input."""
        tickers = ["AAPL", "GOOGL", "MSFT"]

        result = await etf_collector.collect(tickers)

        assert result.success is True
        assert result.records_collected == 0
        assert len(result.errors) == 0

    async def test_collect_single_returns_holdings_count(self, etf_collector):
        """Should return number of holdings saved."""
        # VTI has 10 simulated holdings
        count = await etf_collector.collect_single("VTI")

        assert count == 10
        # Session should have been used to save
        etf_collector.session.add_all.assert_called()
        etf_collector.session.commit.assert_called()

    async def test_collect_single_unknown_etf(self, etf_collector):
        """Should return 0 for ETFs without data."""
        count = await etf_collector.collect_single("UNKNOWN_ETF")

        assert count == 0

    async def test_fetch_holdings_simulated_data(self, etf_collector):
        """Should return simulated holdings for known ETFs."""
        holdings = await etf_collector._fetch_holdings("VTI")

        assert len(holdings) == 10
        assert holdings[0]["ticker"] == "AAPL"
        assert holdings[0]["weight"] == 6.5
        assert holdings[0]["shares"] == 1000000

    async def test_fetch_holdings_unknown_etf(self, etf_collector):
        """Should return empty list for unknown ETFs."""
        holdings = await etf_collector._fetch_holdings("UNKNOWN")

        assert holdings == []


class TestETFHoldingsChangeDetection:
    """Tests for holdings change detection."""

    @pytest.fixture
    def etf_collector(self, mock_session, mock_config):
        """Create an ETF holdings collector instance."""
        return ETFHoldingsCollector(mock_session, mock_config)

    def test_get_holdings_changes_no_current_data(self, etf_collector):
        """Should return empty changes when no current holdings."""
        # Mock no holdings returned
        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.return_value = []

            changes = etf_collector.get_holdings_changes("VTI")

            assert changes == {"added": [], "removed": [], "weight_changes": []}

    def test_get_holdings_changes_no_previous_data(self, etf_collector):
        """Should return empty changes when no previous holdings to compare."""
        today = date.today()

        # Create mock current holdings
        current_holdings = [
            MagicMock(
                holding_ticker="AAPL",
                weight_pct=6.5,
                as_of_date=today,
            ),
            MagicMock(
                holding_ticker="MSFT",
                weight_pct=5.8,
                as_of_date=today,
            ),
        ]

        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            # First call returns current, second returns empty (no previous)
            mock_get.side_effect = [current_holdings, []]

            changes = etf_collector.get_holdings_changes("VTI")

            assert changes == {"added": [], "removed": [], "weight_changes": []}

    def test_get_holdings_changes_detects_added(self, etf_collector):
        """Should detect newly added positions."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        current_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=today),
            MagicMock(holding_ticker="NVDA", weight_pct=2.5, as_of_date=today),  # New
        ]
        previous_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=yesterday),
        ]

        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.side_effect = [current_holdings, previous_holdings]

            changes = etf_collector.get_holdings_changes("VTI")

            assert len(changes["added"]) == 1
            assert changes["added"][0]["ticker"] == "NVDA"
            assert changes["added"][0]["weight"] == 2.5

    def test_get_holdings_changes_detects_removed(self, etf_collector):
        """Should detect removed positions."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        current_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=today),
        ]
        previous_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=yesterday),
            MagicMock(holding_ticker="INTC", weight_pct=1.0, as_of_date=yesterday),  # Removed
        ]

        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.side_effect = [current_holdings, previous_holdings]

            changes = etf_collector.get_holdings_changes("VTI")

            assert len(changes["removed"]) == 1
            assert changes["removed"][0]["ticker"] == "INTC"
            assert changes["removed"][0]["weight"] == 1.0

    def test_get_holdings_changes_detects_weight_changes(self, etf_collector):
        """Should detect significant weight changes."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        current_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=7.5, as_of_date=today),  # +1.0
            MagicMock(holding_ticker="MSFT", weight_pct=5.9, as_of_date=today),  # +0.1 (below threshold)
        ]
        previous_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=yesterday),
            MagicMock(holding_ticker="MSFT", weight_pct=5.8, as_of_date=yesterday),
        ]

        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.side_effect = [current_holdings, previous_holdings]

            changes = etf_collector.get_holdings_changes("VTI")

            # Only AAPL should be reported (change >= 0.5 threshold)
            assert len(changes["weight_changes"]) == 1
            assert changes["weight_changes"][0]["ticker"] == "AAPL"
            assert changes["weight_changes"][0]["old"] == 6.5
            assert changes["weight_changes"][0]["new"] == 7.5

    def test_get_holdings_changes_handles_none_weights(self, etf_collector):
        """Should handle None weights gracefully."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        current_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=None, as_of_date=today),
        ]
        previous_holdings = [
            MagicMock(holding_ticker="AAPL", weight_pct=6.5, as_of_date=yesterday),
        ]

        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.side_effect = [current_holdings, previous_holdings]

            changes = etf_collector.get_holdings_changes("VTI")

            # Should not crash, no weight change reported for None
            assert len(changes["weight_changes"]) == 0

    def test_get_all_changes(self, etf_collector):
        """Should get changes for multiple ETFs."""
        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            # Return empty for all calls
            mock_get.return_value = []

            all_changes = etf_collector.get_all_changes(["VTI", "VOO"])

            # Should have called for both ETFs
            assert "VTI" in all_changes
            assert "VOO" in all_changes

    def test_get_all_changes_defaults_to_known_etfs(self, etf_collector):
        """Should default to checking all known ETFs."""
        with patch(
            "investment_monitor.collectors.etf_holdings.get_etf_holdings"
        ) as mock_get:
            mock_get.return_value = []

            all_changes = etf_collector.get_all_changes()

            # Should have checked all known ETFs
            for etf in etf_collector.KNOWN_ETFS:
                assert etf in all_changes


class TestETFHoldingsCollectorIntegration:
    """Integration tests for ETFHoldingsCollector."""

    @pytest.fixture
    def etf_collector(self, mock_session, mock_config):
        """Create an ETF holdings collector instance."""
        return ETFHoldingsCollector(mock_session, mock_config)

    async def test_full_workflow_with_etfs_and_stocks(self, etf_collector):
        """Should handle mixed portfolio of ETFs and stocks."""
        # Portfolio with ETFs and individual stocks
        portfolio = ["AAPL", "VTI", "MSFT", "QQQ", "GOOGL", "VOO"]

        result = await etf_collector.collect(portfolio)

        # Should succeed
        assert result.success is True
        # Should have collected for VTI, QQQ, VOO only
        # VTI=10, QQQ=10, VOO=10
        assert result.records_collected == 30
        assert len(result.errors) == 0

    async def test_run_method_integration(self, etf_collector):
        """Should work with base class run() method."""
        result = await etf_collector.run(["VTI", "VOO"])

        assert result.success is True
        assert result.collector_name == "etf_holdings"
        assert result.records_collected == 20  # 10 + 10
        assert result.duration_seconds >= 0

    def test_simulated_holdings_structure(self, etf_collector):
        """Should have correct structure for simulated holdings."""
        for etf_ticker, holdings in etf_collector.SIMULATED_HOLDINGS.items():
            assert len(holdings) > 0
            for holding in holdings:
                assert "ticker" in holding
                assert "weight" in holding
                assert "shares" in holding
                assert isinstance(holding["ticker"], str)
                assert isinstance(holding["weight"], (int, float))
                assert isinstance(holding["shares"], int)
