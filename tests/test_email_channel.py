"""Tests for the email (SMTP) notification channel."""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from investment_monitor.notifications.base import AlertMessage, Priority
from investment_monitor.notifications.email import EmailChannel

HOST = "smtp.gmail.com"
TO = "me@example.com"
USER = "robo@example.com"


def _channel(**overrides):
    kwargs = dict(host=HOST, recipient=TO, username=USER, password="app-pw")
    kwargs.update(overrides)
    return EmailChannel(**kwargs)


def _mock_smtp():
    """A context-manager mock standing in for an smtplib.SMTP instance."""
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    smtp.__exit__.return_value = False
    return smtp


class TestEmailChannelConstruction:
    def test_channel_name(self):
        assert _channel().name == "email"

    def test_requires_host(self):
        with pytest.raises(ValueError):
            EmailChannel(host="", recipient=TO, username=USER)
        with pytest.raises(ValueError):
            EmailChannel(host="   ", recipient=TO, username=USER)

    def test_requires_recipient(self):
        with pytest.raises(ValueError):
            EmailChannel(host=HOST, recipient="", username=USER)

    def test_requires_sender_or_username(self):
        # No sender and no username → cannot form a From address.
        with pytest.raises(ValueError):
            EmailChannel(host=HOST, recipient=TO)

    def test_sender_falls_back_to_username(self):
        assert _channel(sender="")._sender == USER

    def test_explicit_sender_wins(self):
        assert _channel(sender="from@x.com")._sender == "from@x.com"


class TestEmailSend:
    def test_send_text_starttls_path(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP", return_value=smtp
        ) as ctor:
            assert _channel().send_text("hello world") is True

        ctor.assert_called_once()
        assert ctor.call_args.args[0] == HOST
        assert ctor.call_args.args[1] == 587
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with(USER, "app-pw")
        smtp.send_message.assert_called_once()

    def test_send_text_no_starttls_when_disabled(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP", return_value=smtp
        ):
            assert _channel(use_tls=False).send_text("hi") is True
        smtp.starttls.assert_not_called()

    def test_port_465_uses_implicit_tls(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP_SSL", return_value=smtp
        ) as ssl_ctor, patch(
            "investment_monitor.notifications.email.smtplib.SMTP"
        ) as plain_ctor:
            assert _channel(port=465).send_text("hi") is True
        ssl_ctor.assert_called_once()
        plain_ctor.assert_not_called()
        smtp.starttls.assert_not_called()

    def test_no_login_without_username(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP", return_value=smtp
        ):
            # sender supplied so construction succeeds without a username
            assert EmailChannel(
                host=HOST, recipient=TO, sender="from@x.com", username=""
            ).send_text("hi") is True
        smtp.login.assert_not_called()
        smtp.send_message.assert_called_once()

    def test_subject_derived_from_first_line(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP", return_value=smtp
        ):
            _channel().send_text("Trade Confirmation — 2 orders executed\nBuy AAPL — $4.00")
        sent = smtp.send_message.call_args.args[0]
        assert sent["Subject"] == "Trade Confirmation — 2 orders executed"
        assert sent["To"] == TO
        assert sent["From"] == USER
        assert "Buy AAPL — $4.00" in sent.get_content()

    def test_explicit_subject_overrides(self):
        smtp = _mock_smtp()
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP", return_value=smtp
        ):
            _channel().send_text("body text", subject="Custom Subject")
        assert smtp.send_message.call_args.args[0]["Subject"] == "Custom Subject"

    def test_smtp_exception_returns_false(self):
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP",
            side_effect=smtplib.SMTPAuthenticationError(535, b"bad creds"),
        ):
            assert _channel().send_text("hi") is False

    def test_connection_refused_returns_false(self):
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP",
            side_effect=ConnectionRefusedError(),
        ):
            assert _channel().send_text("hi") is False

    def test_timeout_returns_false(self):
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP",
            side_effect=TimeoutError(),
        ):
            assert _channel().send_text("hi") is False

    def test_unexpected_exception_returns_false(self):
        with patch(
            "investment_monitor.notifications.email.smtplib.SMTP",
            side_effect=RuntimeError("boom"),
        ):
            assert _channel().send_text("hi") is False


class TestEmailAsyncAPI:
    @pytest.mark.asyncio
    async def test_send_delegates_to_send_text(self):
        channel = _channel()
        msg = AlertMessage(
            title="AAPL up 3%", body="details", alert_type="price", priority=Priority.HIGH
        )
        with patch.object(channel, "send_text", return_value=True) as mock_send:
            assert await channel.send(msg) is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_digest_empty_is_true(self):
        channel = _channel()
        with patch.object(channel, "send_text") as mock_send:
            assert await channel.send_digest([]) is True
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_digest_joins_messages(self):
        channel = _channel()
        msgs = [
            AlertMessage(title="one", body="b", alert_type="x"),
            AlertMessage(title="two", body="b", alert_type="x", ticker="AAPL"),
        ]
        with patch.object(channel, "send_text", return_value=True) as mock_send:
            assert await channel.send_digest(msgs) is True
        sent = mock_send.call_args.args[0]
        assert "one" in sent and "[AAPL] two" in sent
