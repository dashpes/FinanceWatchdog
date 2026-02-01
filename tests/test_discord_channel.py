"""Tests for Discord notification channel."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.notifications.base import AlertMessage, Priority


class TestDiscordChannel:
    """Tests for DiscordChannel."""

    def test_channel_name(self):
        """Test channel has correct name."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        assert channel.name == "discord"

    def test_init_requires_webhook_url(self):
        """Test initialization requires webhook URL."""
        from investment_monitor.notifications.discord import DiscordChannel

        with pytest.raises(ValueError):
            DiscordChannel("")

    @pytest.mark.asyncio
    async def test_send_single_alert(self):
        """Test sending a single alert."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="AAPL dropped 5%",
            body="Apple stock fell significantly.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send(msg)

            assert result is True
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert "json" in call_kwargs
            assert "embeds" in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_handles_failure(self):
        """Test send returns False on HTTP error."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="Test",
            body="Test body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.is_success = False
            mock_post.return_value = mock_response

            result = await channel.send(msg)

            assert result is False

    @pytest.mark.asyncio
    async def test_send_digest_with_pdf(self):
        """Test sending digest generates PDF and sends embed."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
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
        ]

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send_digest(messages)

            assert result is True
            mock_post.assert_called_once()
            # Should include files for PDF
            call_kwargs = mock_post.call_args[1]
            assert "files" in call_kwargs or "data" in call_kwargs

    @pytest.mark.asyncio
    async def test_send_digest_empty_messages(self):
        """Test sending empty digest."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send_digest([])

            assert result is True

    def test_format_alert_embed_price_down(self):
        """Test embed formatting for price drop."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="AAPL -5.2%",
            body="Apple stock dropped significantly.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
        )

        embed = channel._format_alert_embed(msg)

        assert embed["title"] == "[AAPL] AAPL -5.2%"
        assert embed["color"] == 0xDC3545  # Red for price drop

    def test_format_alert_embed_price_up(self):
        """Test embed formatting for price gain."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="NVDA +8.1%",
            body="NVIDIA stock rose today.",
            ticker="NVDA",
            alert_type="price",
            priority=Priority.HIGH,
        )

        embed = channel._format_alert_embed(msg)

        assert embed["color"] == 0x28A745  # Green for price gain

    def test_supports_all_priorities(self):
        """Test channel supports all priorities."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        assert channel.supports_priority(Priority.HIGH) is True
        assert channel.supports_priority(Priority.MEDIUM) is True
        assert channel.supports_priority(Priority.LOW) is True
