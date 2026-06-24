"""Tests for the iMessage notification channel."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from investment_monitor.notifications.base import AlertMessage, Priority
from investment_monitor.notifications.imessage import IMessageChannel

RECIPIENT = "+15551234567"


def _ok_proc():
    proc = MagicMock()
    proc.returncode = 0
    proc.stderr = ""
    return proc


class TestIMessageChannel:
    def test_channel_name(self):
        assert IMessageChannel(RECIPIENT).name == "imessage"

    def test_init_requires_recipient(self):
        with pytest.raises(ValueError):
            IMessageChannel("")
        with pytest.raises(ValueError):
            IMessageChannel("   ")

    def test_recipient_is_stripped(self):
        assert IMessageChannel("  foo@bar.com  ")._recipient == "foo@bar.com"

    def test_send_text_success(self):
        channel = IMessageChannel(RECIPIENT)
        with patch(
            "investment_monitor.notifications.imessage.subprocess.run",
            return_value=_ok_proc(),
        ) as mock_run:
            assert channel.send_text("hello") is True

        # Recipient and body are passed as distinct argv items (osascript `on run argv`),
        # never interpolated into the AppleScript source.
        argv = mock_run.call_args.args[0]
        assert argv[0] == "osascript"
        assert argv[-2] == RECIPIENT
        assert argv[-1] == "hello"

    def test_send_text_no_injection(self):
        """A body that looks like AppleScript stays inert — it's the last argv item."""
        channel = IMessageChannel(RECIPIENT)
        evil = 'x" \n end tell \n tell application "Finder" to delete'
        with patch(
            "investment_monitor.notifications.imessage.subprocess.run",
            return_value=_ok_proc(),
        ) as mock_run:
            assert channel.send_text(evil) is True
        assert mock_run.call_args.args[0][-1] == evil

    def test_send_text_nonzero_returns_false(self):
        channel = IMessageChannel(RECIPIENT)
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = "boom"
        with patch(
            "investment_monitor.notifications.imessage.subprocess.run",
            return_value=proc,
        ):
            assert channel.send_text("hi") is False

    def test_send_text_timeout_returns_false(self):
        channel = IMessageChannel(RECIPIENT)
        with patch(
            "investment_monitor.notifications.imessage.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=15),
        ):
            assert channel.send_text("hi") is False

    def test_send_text_no_osascript_returns_false(self):
        """Non-macOS host: osascript missing — fail-open, never raise."""
        channel = IMessageChannel(RECIPIENT)
        with patch(
            "investment_monitor.notifications.imessage.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            assert channel.send_text("hi") is False

    @pytest.mark.asyncio
    async def test_async_send_delegates_to_send_text(self):
        channel = IMessageChannel(RECIPIENT)
        msg = AlertMessage(
            title="AAPL up 3%", body="details", alert_type="price", priority=Priority.HIGH
        )
        with patch.object(channel, "send_text", return_value=True) as mock_send:
            assert await channel.send(msg) is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_send_digest_empty_is_true(self):
        channel = IMessageChannel(RECIPIENT)
        with patch.object(channel, "send_text") as mock_send:
            assert await channel.send_digest([]) is True
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_send_digest_joins_messages(self):
        channel = IMessageChannel(RECIPIENT)
        msgs = [
            AlertMessage(title="one", body="b", alert_type="x"),
            AlertMessage(title="two", body="b", alert_type="x", ticker="AAPL"),
        ]
        with patch.object(channel, "send_text", return_value=True) as mock_send:
            assert await channel.send_digest(msgs) is True
        sent = mock_send.call_args.args[0]
        assert "one" in sent and "[AAPL] two" in sent
