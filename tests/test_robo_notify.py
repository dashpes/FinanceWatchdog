"""Tests for the robo advisor's notification dispatch and formatting."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from investment_monitor.robo import notify
from investment_monitor.robo.rebalance import RebalanceResult


def _settings(*, to="+15551234567", paper=False, smtp_host="", email_to=""):
    return SimpleNamespace(
        imessage_to=to,
        imessage_notify_paper=paper,
        db_path="data/portfolio.db",
        smtp_host=smtp_host,
        smtp_port=587,
        smtp_username="robo@example.com",
        smtp_password="pw",
        email_from="",
        email_to=email_to,
        email_use_tls=True,
    )


def _result(**kw):
    base = dict(
        run_id="run-1",
        dry_run=False,
        status="completed",
        total_value=Decimal("12345"),
        settled_cash=Decimal("100"),
        num_placed=0,
    )
    base.update(kw)
    return RebalanceResult(**base)


# --- notify_run: trades ----------------------------------------------------------

def test_live_placements_send_one_text_with_orders():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True), \
         patch.object(notify, "_placed_order_lines", return_value=["BUY AAPL $500", "SELL TSLA 3 sh"]):
        notify.notify_run(_result(num_placed=2), _settings())
    text = captured["text"]
    assert "Live" in text
    assert "2 order(s) executed" in text
    assert "BUY AAPL $500" in text
    assert "SELL TSLA 3 sh" in text
    assert "Portfolio value: $12,345" in text
    assert "📈" not in text and "🛑" not in text  # no emojis


def test_no_placements_is_silent():
    with patch.object(notify, "_send") as mock_send:
        notify.notify_run(_result(num_placed=0), _settings())
    mock_send.assert_not_called()


def test_paper_placements_silent_by_default():
    with patch.object(notify, "_send") as mock_send, \
         patch.object(notify, "_placed_order_lines", return_value=["BUY AAPL $500"]):
        notify.notify_run(_result(dry_run=True, num_placed=1), _settings(paper=False))
    mock_send.assert_not_called()


def test_paper_placements_sent_when_enabled():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True), \
         patch.object(notify, "_placed_order_lines", return_value=["BUY AAPL $500"]):
        notify.notify_run(_result(dry_run=True, num_placed=1), _settings(paper=True))
    assert "Paper" in captured["text"]


# --- notify_run: error statuses --------------------------------------------------

def test_refused_run_notifies():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True):
        notify.notify_run(
            _result(status="refused", message="account has margin"), _settings()
        )
    assert "Trading Run Refused" in captured["text"]
    assert "account has margin" in captured["text"]
    assert "⛔" not in captured["text"]  # professional, no emoji


def test_failed_run_notifies():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True):
        notify.notify_run(_result(status="failed", message="account fetch failed"), _settings())
    assert "Trading Run Failed" in captured["text"]
    assert "🛑" not in captured["text"]


def test_notify_run_never_raises():
    """A failure deep in formatting must never propagate into the CLI."""
    with patch.object(notify, "_placed_order_lines", side_effect=RuntimeError("db down")), \
         patch.object(notify, "_send", return_value=True):
        # Should not raise.
        notify.notify_run(_result(num_placed=1), _settings())


def test_notify_error_sends_message():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True):
        notify.notify_error(_settings(), message="boom", dry_run=False)
    assert "Trading Run Error" in captured["text"]
    assert "boom" in captured["text"]
    assert "Live" in captured["text"]


# --- _send gating ----------------------------------------------------------------

def test_send_is_noop_without_recipient():
    # No recipient -> no channel -> returns False, never raises.
    assert notify._send(_settings(to=""), "anything") is False


# --- _channel selection (email preferred over iMessage) --------------------------

def test_channel_prefers_email_when_configured():
    from investment_monitor.notifications.email import EmailChannel

    settings = _settings(to="+15551234567", smtp_host="smtp.gmail.com", email_to="me@x.com")
    assert isinstance(notify._channel(settings), EmailChannel)


def test_channel_falls_back_to_imessage_without_email():
    from investment_monitor.notifications.imessage import IMessageChannel

    # smtp_host set but no recipient -> email inactive -> iMessage used.
    settings = _settings(to="+15551234567", smtp_host="smtp.gmail.com", email_to="")
    assert isinstance(notify._channel(settings), IMessageChannel)


def test_channel_none_when_nothing_configured():
    assert notify._channel(_settings(to="", smtp_host="", email_to="")) is None


def test_notifications_configured_reflects_channel():
    assert notify.notifications_configured(_settings(smtp_host="smtp.x.com", email_to="me@x.com"))
    assert not notify.notifications_configured(_settings(to="", smtp_host="", email_to=""))


# --- _signed_money (sign decided after rounding) ---------------------------------

def test_signed_money_tiny_negative_is_not_negative_zero():
    # A magnitude that rounds to 0.00 must never render the misleading '-$0.00'.
    assert notify._signed_money(Decimal("-0.004")) == "+$0.00"
    assert notify._signed_money(Decimal("-0.0049")) == "+$0.00"


def test_signed_money_exact_zero_is_positive():
    assert notify._signed_money(Decimal("0")) == "+$0.00"


def test_signed_money_keeps_real_signs():
    # Real, non-zero magnitudes keep their sign and rounding behaviour.
    assert notify._signed_money(Decimal("-12.50")) == "-$12.50"
    assert notify._signed_money(Decimal("0.04")) == "+$0.04"
    assert notify._signed_money(Decimal("-0.006")) == "-$0.01"  # rounds up to a cent
    assert notify._signed_money(Decimal("1234.5")) == "+$1,234.50"


# --- format_daily_summary (pure) -------------------------------------------------

def _account():
    positions = [
        SimpleNamespace(symbol="AAPL", unrealized_gain=Decimal("120")),
        SimpleNamespace(symbol="TSLA", unrealized_gain=Decimal("-300")),
        SimpleNamespace(symbol="MSFT", unrealized_gain=Decimal("15")),
        SimpleNamespace(symbol="NVDA", unrealized_gain=None),  # no basis -> skipped
    ]
    return SimpleNamespace(
        total_value=Decimal("10000"),
        settled_cash=Decimal("2500"),
        total_unrealized_gain=Decimal("-165"),
        total_cost_basis=Decimal("8000"),
        positions=positions,
    )


def test_format_daily_summary_basics():
    text = notify.format_daily_summary(_account())
    assert "Portfolio value" in text and "$10,000.00" in text
    assert "Cash available" in text and "$2,500.00" in text
    assert "Unrealised P&L" in text and "-$165.00" in text  # British spelling
    assert "%" in text  # pct shown because cost basis present
    assert "📊" not in text  # professional, no emoji
    # Archie's letterhead + sign-off (persona branding)
    assert "ARCHIE" in text and "Yours faithfully,\nArchie" in text


def test_format_daily_summary_movers_sorted_by_magnitude():
    text = notify.format_daily_summary(_account())
    lines = text.splitlines()
    mover_lines = [ln.strip() for ln in lines if ln.startswith("  ")]
    # Biggest absolute mover (TSLA -300) ranks above AAPL (+120) and MSFT (+15).
    assert mover_lines[0].startswith("TSLA")
    assert "NVDA" not in text  # position with no unrealized gain is excluded


def test_format_daily_summary_with_realized():
    realized = SimpleNamespace(total_realized=Decimal("54"))
    text = notify.format_daily_summary(_account(), realized)
    assert "Realised P&L" in text and "+$54.00" in text


def test_send_daily_summary_uses_send():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None: captured.update(text=t, subject=subject) or True):
        assert notify.send_daily_summary(_settings(), _account()) is True
    assert "Daily Portfolio Summary" in captured["text"]
    # Email subject is brand-prefixed and never the "ARCHIE" letterhead line.
    assert captured["subject"].startswith("Archie · Daily Portfolio Summary")


def test_send_test_uses_send():
    with patch.object(notify, "_send", return_value=True) as mock_send:
        assert notify.send_test(_settings()) is True
    mock_send.assert_called_once()
