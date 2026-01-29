"""Tests for the notification system."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.models.portfolio import Holding, Portfolio
from investment_monitor.notifications import (
    AlertMessage,
    ConsoleChannel,
    NotificationChannel,
    NotificationManager,
    Priority,
    format_daily_digest,
    format_weekly_digest,
)


# ============================================================================
# AlertMessage Tests
# ============================================================================


class TestAlertMessage:
    """Tests for AlertMessage model."""

    def test_create_minimal(self):
        """Test creating message with minimal required fields."""
        msg = AlertMessage(
            title="Test Alert",
            body="This is a test alert body.",
            alert_type="test",
        )
        assert msg.title == "Test Alert"
        assert msg.body == "This is a test alert body."
        assert msg.alert_type == "test"
        assert msg.priority == Priority.MEDIUM  # default
        assert msg.ticker is None
        assert msg.url is None
        assert isinstance(msg.timestamp, datetime)

    def test_create_full(self):
        """Test creating message with all fields."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        msg = AlertMessage(
            title="AAPL Price Drop",
            body="Apple stock dropped 5% today.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
            url="https://example.com/aapl",
            timestamp=ts,
        )
        assert msg.title == "AAPL Price Drop"
        assert msg.body == "Apple stock dropped 5% today."
        assert msg.ticker == "AAPL"
        assert msg.alert_type == "price"
        assert msg.priority == Priority.HIGH
        assert msg.url == "https://example.com/aapl"
        assert msg.timestamp == ts

    def test_priority_values(self):
        """Test all priority values are accepted."""
        for priority in Priority:
            msg = AlertMessage(
                title="Test",
                body="Test body",
                alert_type="test",
                priority=priority,
            )
            assert msg.priority == priority

    def test_format_short_with_ticker(self):
        """Test short format with ticker."""
        msg = AlertMessage(
            title="Price Alert",
            body="Details here",
            ticker="MSFT",
            alert_type="price",
        )
        assert msg.format_short() == "[MSFT] Price Alert"

    def test_format_short_without_ticker(self):
        """Test short format without ticker."""
        msg = AlertMessage(
            title="General Alert",
            body="Details here",
            alert_type="system",
        )
        assert msg.format_short() == "General Alert"

    def test_format_full_with_url(self):
        """Test full format includes URL."""
        msg = AlertMessage(
            title="Test Alert",
            body="Alert body content",
            ticker="AAPL",
            alert_type="test",
            url="https://example.com",
        )
        full = msg.format_full()
        assert "[AAPL] Test Alert" in full
        assert "Alert body content" in full
        assert "More info: https://example.com" in full

    def test_format_full_without_url(self):
        """Test full format without URL."""
        msg = AlertMessage(
            title="Test Alert",
            body="Alert body content",
            alert_type="test",
        )
        full = msg.format_full()
        assert "Test Alert" in full
        assert "Alert body content" in full
        assert "More info" not in full

    def test_title_validation_empty(self):
        """Test that empty title is rejected."""
        with pytest.raises(ValueError):
            AlertMessage(title="", body="body", alert_type="test")

    def test_body_validation_empty(self):
        """Test that empty body is rejected."""
        with pytest.raises(ValueError):
            AlertMessage(title="Title", body="", alert_type="test")

    def test_alert_type_validation_empty(self):
        """Test that empty alert_type is rejected."""
        with pytest.raises(ValueError):
            AlertMessage(title="Title", body="body", alert_type="")


# ============================================================================
# Priority Tests
# ============================================================================


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_priority_string_conversion(self):
        """Test priority can be created from string."""
        assert Priority("high") == Priority.HIGH
        assert Priority("medium") == Priority.MEDIUM
        assert Priority("low") == Priority.LOW


# ============================================================================
# ConsoleChannel Tests
# ============================================================================


class TestConsoleChannel:
    """Tests for ConsoleChannel."""

    def test_channel_name(self):
        """Test channel has correct name."""
        channel = ConsoleChannel()
        assert channel.name == "console"

    @pytest.mark.asyncio
    async def test_send_high_priority(self):
        """Test sending high priority message logs at error level."""
        channel = ConsoleChannel()
        msg = AlertMessage(
            title="Critical Alert",
            body="This is critical.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
        )

        with patch.object(channel._logger, "error") as mock_error:
            result = await channel.send(msg)

            assert result is True
            # Should log title and body for HIGH priority
            assert mock_error.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_medium_priority(self):
        """Test sending medium priority message logs at warning level."""
        channel = ConsoleChannel()
        msg = AlertMessage(
            title="Notable Alert",
            body="This is notable.",
            alert_type="volume",
            priority=Priority.MEDIUM,
        )

        with patch.object(channel._logger, "warning") as mock_warning:
            result = await channel.send(msg)

            assert result is True
            mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_low_priority(self):
        """Test sending low priority message logs at debug level."""
        channel = ConsoleChannel()
        msg = AlertMessage(
            title="Minor Alert",
            body="This is minor.",
            alert_type="info",
            priority=Priority.LOW,
        )

        with patch.object(channel._logger, "debug") as mock_debug:
            result = await channel.send(msg)

            assert result is True
            mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_handles_exception(self):
        """Test that send handles exceptions gracefully."""
        channel = ConsoleChannel()
        msg = AlertMessage(
            title="Test",
            body="Test body",
            alert_type="test",
        )

        # Force an exception
        with patch.object(channel._logger, "warning", side_effect=Exception("Test error")):
            result = await channel.send(msg)

            # Should return False but not raise
            assert result is False

    @pytest.mark.asyncio
    async def test_send_digest_empty(self):
        """Test sending empty digest."""
        channel = ConsoleChannel()

        with patch.object(channel._logger, "info") as mock_info:
            result = await channel.send_digest([])

            assert result is True
            # Should log that there are no messages
            mock_info.assert_called()

    @pytest.mark.asyncio
    async def test_send_digest_with_messages(self):
        """Test sending digest with multiple messages."""
        channel = ConsoleChannel()
        messages = [
            AlertMessage(
                title="Alert 1",
                body="Body 1",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
            AlertMessage(
                title="Alert 2",
                body="Body 2",
                ticker="MSFT",
                alert_type="volume",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="Alert 3",
                body="Body 3",
                alert_type="system",
                priority=Priority.LOW,
            ),
        ]

        with patch.object(channel._logger, "info") as mock_info:
            result = await channel.send_digest(messages)

            assert result is True
            mock_info.assert_called()

    @pytest.mark.asyncio
    async def test_send_digest_groups_by_ticker(self):
        """Test digest groups messages by ticker."""
        channel = ConsoleChannel()
        messages = [
            AlertMessage(title="A1", body="B1", ticker="AAPL", alert_type="p", priority=Priority.HIGH),
            AlertMessage(title="A2", body="B2", ticker="AAPL", alert_type="v", priority=Priority.HIGH),
            AlertMessage(title="A3", body="B3", ticker="MSFT", alert_type="p", priority=Priority.HIGH),
        ]

        # Capture the digest output
        captured_digest = None

        def capture_log(*args, **kwargs):
            nonlocal captured_digest
            if "digest" in kwargs:
                captured_digest = kwargs["digest"]

        with patch.object(channel._logger, "info", side_effect=capture_log):
            result = await channel.send_digest(messages)

        assert result is True
        assert captured_digest is not None
        assert "[AAPL]" in captured_digest
        assert "[MSFT]" in captured_digest

    @pytest.mark.asyncio
    async def test_send_digest_handles_exception(self):
        """Test that digest handles exceptions gracefully."""
        channel = ConsoleChannel()
        messages = [
            AlertMessage(title="Test", body="Body", alert_type="test"),
        ]

        with patch.object(channel._logger, "info", side_effect=Exception("Test error")):
            result = await channel.send_digest(messages)

            assert result is False

    def test_supports_all_priorities(self):
        """Test console channel supports all priorities."""
        channel = ConsoleChannel()
        assert channel.supports_priority(Priority.HIGH) is True
        assert channel.supports_priority(Priority.MEDIUM) is True
        assert channel.supports_priority(Priority.LOW) is True


# ============================================================================
# NotificationManager Tests
# ============================================================================


class MockChannel(NotificationChannel):
    """Mock notification channel for testing."""

    name = "mock"

    def __init__(self, send_result: bool = True, digest_result: bool = True):
        self.send_result = send_result
        self.digest_result = digest_result
        self.sent_messages: list[AlertMessage] = []
        self.digest_calls: list[list[AlertMessage]] = []

    async def send(self, message: AlertMessage) -> bool:
        self.sent_messages.append(message)
        return self.send_result

    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        self.digest_calls.append(messages)
        return self.digest_result


class TestNotificationManager:
    """Tests for NotificationManager."""

    def test_init_empty(self):
        """Test creating manager without channels."""
        manager = NotificationManager()
        assert manager.channels == []
        assert manager.digest_queue == []

    def test_init_with_channels(self):
        """Test creating manager with channels."""
        channel1 = MockChannel()
        channel2 = MockChannel()
        manager = NotificationManager([channel1, channel2])

        assert len(manager.channels) == 2

    def test_add_channel(self):
        """Test adding a channel."""
        manager = NotificationManager()
        channel = MockChannel()

        manager.add_channel(channel)

        assert len(manager.channels) == 1
        assert manager.channels[0] is channel

    def test_remove_channel_exists(self):
        """Test removing an existing channel."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        result = manager.remove_channel("mock")

        assert result is True
        assert len(manager.channels) == 0

    def test_remove_channel_not_found(self):
        """Test removing a non-existent channel."""
        manager = NotificationManager()

        result = manager.remove_channel("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_notify_high_priority_immediate(self):
        """Test high priority messages are sent immediately."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        msg = AlertMessage(
            title="Urgent",
            body="Urgent body",
            alert_type="price",
            priority=Priority.HIGH,
        )

        await manager.notify(msg)

        assert len(channel.sent_messages) == 1
        assert channel.sent_messages[0] is msg
        assert len(manager.digest_queue) == 0

    @pytest.mark.asyncio
    async def test_notify_medium_priority_queued(self):
        """Test medium priority messages are queued for digest."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        msg = AlertMessage(
            title="Notable",
            body="Notable body",
            alert_type="volume",
            priority=Priority.MEDIUM,
        )

        await manager.notify(msg)

        assert len(channel.sent_messages) == 0
        assert len(manager.digest_queue) == 1
        assert manager.digest_queue[0] is msg

    @pytest.mark.asyncio
    async def test_notify_low_priority_log_only(self):
        """Test low priority messages are logged only."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        msg = AlertMessage(
            title="Minor",
            body="Minor body",
            alert_type="info",
            priority=Priority.LOW,
        )

        await manager.notify(msg)

        assert len(channel.sent_messages) == 0
        assert len(manager.digest_queue) == 0

    @pytest.mark.asyncio
    async def test_notify_no_channels_configured(self):
        """Test notification with no channels logs warning."""
        manager = NotificationManager()

        msg = AlertMessage(
            title="Test",
            body="Body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        # Should not raise
        await manager.notify(msg)

    @pytest.mark.asyncio
    async def test_send_daily_digest_uses_queue(self):
        """Test daily digest uses internal queue."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        msg1 = AlertMessage(title="M1", body="B1", alert_type="t", priority=Priority.MEDIUM)
        msg2 = AlertMessage(title="M2", body="B2", alert_type="t", priority=Priority.MEDIUM)

        await manager.notify(msg1)
        await manager.notify(msg2)

        assert len(manager.digest_queue) == 2

        await manager.send_daily_digest()

        assert len(channel.digest_calls) == 1
        assert len(channel.digest_calls[0]) == 2
        assert len(manager.digest_queue) == 0  # Queue cleared

    @pytest.mark.asyncio
    async def test_send_daily_digest_with_explicit_messages(self):
        """Test daily digest with explicit message list."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        # Add something to internal queue
        await manager.notify(
            AlertMessage(title="Queued", body="B", alert_type="t", priority=Priority.MEDIUM)
        )

        explicit_messages = [
            AlertMessage(title="Explicit1", body="B", alert_type="t"),
            AlertMessage(title="Explicit2", body="B", alert_type="t"),
        ]

        await manager.send_daily_digest(explicit_messages)

        # Should use explicit messages, not queue
        assert len(channel.digest_calls) == 1
        assert len(channel.digest_calls[0]) == 2
        assert channel.digest_calls[0][0].title == "Explicit1"

        # Queue should NOT be cleared when explicit messages provided
        assert len(manager.digest_queue) == 1

    @pytest.mark.asyncio
    async def test_send_daily_digest_empty(self):
        """Test daily digest with no messages."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        await manager.send_daily_digest()

        # Should not call send_digest on channel
        assert len(channel.digest_calls) == 0

    @pytest.mark.asyncio
    async def test_send_daily_digest_no_channels(self):
        """Test daily digest with no channels configured."""
        manager = NotificationManager()

        msg = AlertMessage(title="Test", body="B", alert_type="t", priority=Priority.MEDIUM)
        await manager.notify(msg)

        # Should not raise
        await manager.send_daily_digest()

    @pytest.mark.asyncio
    async def test_multiple_channels_all_receive(self):
        """Test high priority messages go to all channels."""
        channel1 = MockChannel()
        channel2 = MockChannel()
        manager = NotificationManager([channel1, channel2])

        msg = AlertMessage(
            title="Urgent",
            body="Body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        await manager.notify(msg)

        assert len(channel1.sent_messages) == 1
        assert len(channel2.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_channel_failure_does_not_stop_others(self):
        """Test one channel failing doesn't prevent others from receiving."""
        failing_channel = MockChannel(send_result=False)
        success_channel = MockChannel(send_result=True)
        manager = NotificationManager([failing_channel, success_channel])

        msg = AlertMessage(
            title="Test",
            body="Body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        await manager.notify(msg)

        # Both should receive the message attempt
        assert len(failing_channel.sent_messages) == 1
        assert len(success_channel.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_channel_exception_handled(self):
        """Test channel exception is handled gracefully."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        # Make channel.send raise an exception
        async def raise_error(msg):
            raise RuntimeError("Channel error")

        channel.send = raise_error

        msg = AlertMessage(
            title="Test",
            body="Body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        # Should not raise
        await manager.notify(msg)

    def test_clear_digest_queue(self):
        """Test manually clearing digest queue."""
        manager = NotificationManager()
        manager._digest_queue = [
            AlertMessage(title="M1", body="B", alert_type="t"),
            AlertMessage(title="M2", body="B", alert_type="t"),
        ]

        count = manager.clear_digest_queue()

        assert count == 2
        assert len(manager.digest_queue) == 0

    def test_get_digest_queue_size(self):
        """Test getting digest queue size."""
        manager = NotificationManager()

        assert manager.get_digest_queue_size() == 0

        manager._digest_queue.append(
            AlertMessage(title="M", body="B", alert_type="t")
        )

        assert manager.get_digest_queue_size() == 1

    def test_channels_property_returns_copy(self):
        """Test channels property returns a copy."""
        channel = MockChannel()
        manager = NotificationManager([channel])

        channels = manager.channels
        channels.append(MockChannel())

        # Original should not be modified
        assert len(manager.channels) == 1

    def test_digest_queue_property_returns_copy(self):
        """Test digest_queue property returns a copy."""
        manager = NotificationManager()
        manager._digest_queue.append(
            AlertMessage(title="M", body="B", alert_type="t")
        )

        queue = manager.digest_queue
        queue.append(AlertMessage(title="M2", body="B", alert_type="t"))

        # Original should not be modified
        assert len(manager.digest_queue) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestNotificationIntegration:
    """Integration tests for the notification system."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete notification workflow."""
        channel = ConsoleChannel()
        manager = NotificationManager([channel])

        # Send high priority (immediate)
        high_msg = AlertMessage(
            title="Critical Price Drop",
            body="AAPL dropped 10% in one hour!",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
            url="https://finance.example.com/aapl",
        )
        await manager.notify(high_msg)

        # Queue medium priority
        medium_msgs = [
            AlertMessage(
                title="Volume Spike",
                body="MSFT volume 2x normal",
                ticker="MSFT",
                alert_type="volume",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="Earnings Reminder",
                body="GOOGL earnings in 3 days",
                ticker="GOOGL",
                alert_type="earnings",
                priority=Priority.MEDIUM,
            ),
        ]
        for msg in medium_msgs:
            await manager.notify(msg)

        # Log low priority
        low_msg = AlertMessage(
            title="Minor Update",
            body="Market closed normally",
            alert_type="system",
            priority=Priority.LOW,
        )
        await manager.notify(low_msg)

        # Verify queue state
        assert manager.get_digest_queue_size() == 2

        # Send digest
        await manager.send_daily_digest()

        # Queue should be cleared
        assert manager.get_digest_queue_size() == 0

    @pytest.mark.asyncio
    async def test_mixed_channels(self):
        """Test manager with multiple different channels."""
        console = ConsoleChannel()
        mock = MockChannel()
        manager = NotificationManager([console, mock])

        msg = AlertMessage(
            title="Test Alert",
            body="Test body",
            ticker="TEST",
            alert_type="test",
            priority=Priority.HIGH,
        )

        await manager.notify(msg)

        # Mock channel should have received the message
        assert len(mock.sent_messages) == 1
        assert mock.sent_messages[0].title == "Test Alert"


# ============================================================================
# Digest Formatter Tests
# ============================================================================


class TestFormatDailyDigest:
    """Tests for format_daily_digest function."""

    def test_empty_messages(self):
        """Test digest with no messages."""
        plain, html = format_daily_digest([], date_value=date(2026, 1, 28))

        assert "INVESTMENT MONITOR DAILY DIGEST" in plain
        assert "January 28, 2026" in plain
        assert "No alerts for today" in plain
        assert "Generated by Investment Monitor" in plain

        assert "<!DOCTYPE html>" in html
        assert "No alerts for today" in html

    def test_single_price_alert_up(self):
        """Test digest with a single upward price alert."""
        messages = [
            AlertMessage(
                title="AAPL +2.5%",
                body="Apple stock rose 2.5% today.",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            )
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        assert "PRICE MOVEMENTS" in plain
        assert "AAPL" in plain
        # Arrow up for positive movement
        assert "\u25b2" in plain  # ▲
        assert "1 price alert" in plain  # summary

        assert "price-up" in html

    def test_single_price_alert_down(self):
        """Test digest with a single downward price alert."""
        messages = [
            AlertMessage(
                title="MSFT -3.2%",
                body="Microsoft stock dropped 3.2% today.",
                ticker="MSFT",
                alert_type="price",
                priority=Priority.MEDIUM,
            )
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        assert "PRICE MOVEMENTS" in plain
        assert "MSFT" in plain
        # Arrow down for negative movement
        assert "\u25bc" in plain  # ▼

        assert "price-down" in html

    def test_multiple_alert_types(self):
        """Test digest with multiple types of alerts."""
        messages = [
            AlertMessage(
                title="AAPL -3.2%",
                body="Apple stock dropped significantly.",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
            AlertMessage(
                title="Tim Cook sells shares",
                body="Tim Cook (CEO) sold 50,000 shares @ $180.00. Total value: $9,000,000",
                ticker="AAPL",
                alert_type="insider",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="Apple Announces New AI Features",
                body="Apple unveils new AI capabilities for iPhone.",
                ticker="AAPL",
                alert_type="news",
                priority=Priority.MEDIUM,
                url="https://example.com/apple-ai",
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # Check all sections are present
        assert "PRICE MOVEMENTS" in plain
        assert "INSIDER ACTIVITY" in plain
        assert "NEWS" in plain

        # Check summary
        assert "1 price alert" in plain
        assert "1 insider transaction" in plain
        assert "1 relevant news item" in plain

        # Check HTML has all sections
        assert "PRICE MOVEMENTS" in html
        assert "INSIDER ACTIVITY" in html
        assert "NEWS" in html

    def test_grouping_by_ticker(self):
        """Test that alerts are grouped by ticker within each type."""
        messages = [
            AlertMessage(
                title="AAPL drops",
                body="Apple fell 2%",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="MSFT rises",
                body="Microsoft up 1%",
                ticker="MSFT",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="AAPL Volume spike",
                body="3x normal volume",
                ticker="AAPL",
                alert_type="volume",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # Price alerts show ticker in a different format
        assert "AAPL" in plain
        assert "MSFT" in plain
        assert "2 price alerts" in plain
        assert "1 volume alert" in plain
        # Volume alerts use bracketed ticker format
        assert "[AAPL]" in plain

    def test_with_portfolio(self):
        """Test digest with portfolio context."""
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="AAPL", shares=Decimal("100"), cost_basis=Decimal("150.00")),
                Holding(ticker="GOOGL", shares=Decimal("50"), cost_basis=Decimal("140.00")),
            ]
        )
        messages = [
            AlertMessage(
                title="AAPL drops",
                body="Apple fell 2%",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, portfolio=portfolio, date_value=date(2026, 1, 28))

        # Portfolio doesn't affect daily digest much currently
        assert "AAPL" in plain
        assert "AAPL" in html

    def test_default_date_is_today(self):
        """Test that default date is today."""
        messages = []
        plain, html = format_daily_digest(messages)

        # Should contain today's date
        today = date.today()
        expected_date = today.strftime("%B %d, %Y")
        assert expected_date in plain

    def test_url_included(self):
        """Test that URLs are included in output."""
        messages = [
            AlertMessage(
                title="News Alert",
                body="Important news",
                ticker="AAPL",
                alert_type="news",
                priority=Priority.MEDIUM,
                url="https://example.com/news",
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        assert "https://example.com/news" in plain
        assert "https://example.com/news" in html

    def test_html_escapes_special_characters(self):
        """Test that HTML output escapes special characters."""
        messages = [
            AlertMessage(
                title="Price Alert",
                body="Stock moved <5% or >10%",
                ticker="TEST",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # HTML should have escaped < and >
        assert "&lt;5%" in html
        assert "&gt;10%" in html

    def test_general_alerts_without_ticker(self):
        """Test alerts without a ticker are handled."""
        messages = [
            AlertMessage(
                title="System Update",
                body="Market closed early today.",
                alert_type="system",
                priority=Priority.LOW,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        assert "SYSTEM NOTIFICATIONS" in plain
        assert "System Update" in plain


class TestFormatWeeklyDigest:
    """Tests for format_weekly_digest function."""

    def test_empty_messages(self):
        """Test weekly digest with no messages."""
        plain, html = format_weekly_digest(
            [],
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        assert "INVESTMENT MONITOR WEEKLY DIGEST" in plain
        assert "January 22 - 28, 2026" in plain
        assert "No alerts this week" in plain
        assert "Generated by Investment Monitor" in plain

        assert "<!DOCTYPE html>" in html
        assert "No alerts this week" in html

    def test_with_ai_synthesis(self):
        """Test weekly digest with AI synthesis."""
        messages = [
            AlertMessage(
                title="AAPL down",
                body="Apple dropped 3%",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]
        ai_synthesis = "This week saw moderate volatility in tech stocks.\nAAPL declined following mixed earnings guidance."

        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
            ai_synthesis=ai_synthesis,
        )

        assert "WEEKLY ANALYSIS" in plain
        assert "moderate volatility" in plain
        assert "AAPL declined" in plain

        assert "Weekly Analysis" in html
        assert "moderate volatility" in html

    def test_summary_includes_total_and_tickers(self):
        """Test weekly summary includes total count and tickers."""
        messages = [
            AlertMessage(
                title="AAPL drops",
                body="Apple fell",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="MSFT rises",
                body="Microsoft up",
                ticker="MSFT",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="AAPL news",
                body="Apple news",
                ticker="AAPL",
                alert_type="news",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        assert "Total alerts: 3" in plain
        assert "Tickers mentioned:" in plain
        assert "AAPL" in plain
        assert "MSFT" in plain

        assert "Total alerts: 3" in html

    def test_with_portfolio_shows_holdings(self):
        """Test weekly digest highlights portfolio holdings."""
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="AAPL", shares=Decimal("100"), cost_basis=Decimal("150.00")),
                Holding(ticker="GOOGL", shares=Decimal("50"), cost_basis=Decimal("140.00")),
            ]
        )
        messages = [
            AlertMessage(
                title="AAPL drops",
                body="Apple fell",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="MSFT rises",
                body="Microsoft up",
                ticker="MSFT",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_weekly_digest(
            messages,
            portfolio=portfolio,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        assert "Portfolio holdings with alerts: AAPL" in plain
        assert "Portfolio holdings with alerts: AAPL" in html

    def test_date_range_formatting(self):
        """Test various date range formats."""
        messages = []

        # Same month
        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )
        assert "January 22 - 28, 2026" in plain

        # Different months same year
        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 28),
            week_end=date(2026, 2, 3),
        )
        assert "January 28 - February 03, 2026" in plain

        # Different years
        plain, html = format_weekly_digest(
            messages,
            week_start=date(2025, 12, 29),
            week_end=date(2026, 1, 4),
        )
        assert "December 29, 2025 - January 04, 2026" in plain

    def test_default_dates(self):
        """Test default dates are computed correctly."""
        messages = []
        plain, html = format_weekly_digest(messages)

        # Should default to last 7 days
        today = date.today()
        assert today.strftime("%B") in plain or (today.month == 1 and "December" in plain)

    def test_multiple_alert_types_weekly(self):
        """Test weekly digest handles multiple alert types."""
        messages = [
            AlertMessage(
                title="AAPL drops",
                body="Apple fell 3%",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="Tim Cook sells",
                body="CEO sold shares",
                ticker="AAPL",
                alert_type="insider",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="AAPL earnings",
                body="Earnings next week",
                ticker="AAPL",
                alert_type="earnings",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="MSFT news",
                body="Microsoft announcement",
                ticker="MSFT",
                alert_type="news",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        assert "PRICE MOVEMENTS" in plain
        assert "INSIDER ACTIVITY" in plain
        assert "EARNINGS" in plain
        assert "NEWS" in plain

    def test_ai_synthesis_with_newlines(self):
        """Test AI synthesis with multiple lines."""
        ai_synthesis = """Key observations this week:
- Tech sector showed mixed performance
- AAPL declined on guidance concerns
- MSFT continued strong momentum

Overall market sentiment remains cautious."""

        plain, html = format_weekly_digest(
            [],
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
            ai_synthesis=ai_synthesis,
        )

        assert "Key observations this week:" in plain
        assert "Tech sector showed mixed performance" in plain
        assert "Overall market sentiment" in plain

        # HTML should have <br> for newlines
        assert "<br>" in html

    def test_ticker_grouping_in_sections(self):
        """Test tickers are grouped within each section."""
        messages = [
            AlertMessage(
                title="AAPL down 1%",
                body="First AAPL alert",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="AAPL down 2%",
                body="Second AAPL alert",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="MSFT up 1%",
                body="MSFT alert",
                ticker="MSFT",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_weekly_digest(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        # Should show ticker headers in weekly digest
        assert "[AAPL]" in plain
        assert "[MSFT]" in plain


class TestDigestEdgeCases:
    """Edge case tests for digest formatters."""

    def test_very_long_body(self):
        """Test handling of very long message body."""
        long_body = "A" * 1000
        messages = [
            AlertMessage(
                title="Long Alert",
                body=long_body,
                ticker="TEST",
                alert_type="news",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # Should not crash
        assert "Long Alert" in plain
        assert len(html) > 1000

    def test_special_characters_in_title(self):
        """Test handling of special characters in title."""
        messages = [
            AlertMessage(
                title="Price Alert: $AAPL -3.5% & More",
                body="Details here",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        assert "$AAPL" in plain
        assert "&amp;" in html  # & should be escaped in HTML

    def test_unknown_alert_type(self):
        """Test handling of unknown alert type."""
        messages = [
            AlertMessage(
                title="Custom Alert",
                body="Some custom alert type",
                ticker="TEST",
                alert_type="custom_type",
                priority=Priority.MEDIUM,
            ),
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # Unknown type should be uppercased
        assert "CUSTOM_TYPE" in plain
        assert "Custom Alert" in plain

    def test_all_alert_types(self):
        """Test all known alert types are formatted correctly."""
        alert_types = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]
        messages = [
            AlertMessage(
                title=f"{at.upper()} Alert",
                body=f"Alert for {at}",
                ticker="TEST",
                alert_type=at,
                priority=Priority.MEDIUM,
            )
            for at in alert_types
        ]

        plain, html = format_daily_digest(messages, date_value=date(2026, 1, 28))

        # Check each section header exists
        expected_headers = [
            "PRICE MOVEMENTS",
            "VOLUME ALERTS",
            "INSIDER ACTIVITY",
            "NEWS",
            "EARNINGS",
            "DIVIDENDS",
            "SEC FILINGS",
            "ANALYST UPDATES",
            "SYSTEM NOTIFICATIONS",
        ]

        for header in expected_headers:
            assert header in plain

    def test_returns_tuple(self):
        """Test that formatters return tuple of (plain, html)."""
        plain, html = format_daily_digest([], date_value=date(2026, 1, 28))
        assert isinstance(plain, str)
        assert isinstance(html, str)

        plain, html = format_weekly_digest([], week_start=date(2026, 1, 22), week_end=date(2026, 1, 28))
        assert isinstance(plain, str)
        assert isinstance(html, str)

    def test_empty_portfolio(self):
        """Test with empty portfolio."""
        portfolio = Portfolio(holdings=[], watchlist=[])
        messages = [
            AlertMessage(
                title="Alert",
                body="Body",
                ticker="TEST",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        # Should not crash
        plain, html = format_daily_digest(messages, portfolio=portfolio, date_value=date(2026, 1, 28))
        assert "Alert" in plain

        plain, html = format_weekly_digest(
            messages,
            portfolio=portfolio,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )
        assert "Alert" in plain
