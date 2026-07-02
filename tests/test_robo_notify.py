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


def _account(**kw):
    from investment_monitor.robo.models import AccountState

    base = dict(
        account_id="A", is_cash_account=True, has_margin=False,
        settled_cash=Decimal("100"), positions=[],
    )
    base.update(kw)
    return AccountState(**base)


# --- the "why": rationale snapshot + surfacing --------------------------------------
def test_order_rationale_summarises_the_thesis():
    from investment_monitor.robo.rebalance import _order_rationale

    thesis = SimpleNamespace(id=7, conviction=0.78, narrative="Insider cluster: 3 insiders bought $937k")
    assert _order_rationale(thesis) == "78% conviction — Insider cluster: 3 insiders bought $937k"
    assert _order_rationale(None) == ""  # no owning thesis (rebalance / manual name)


def test_daily_summary_shows_todays_trades_with_why():
    trades = ["  Buy EML — $1.94", "      why: 78% conviction — Insider cluster"]
    text = notify.format_daily_summary(_account(), None, trades)
    assert "Today's trades:" in text
    assert "Buy EML — $1.94" in text
    assert "why: 78% conviction — Insider cluster" in text


def test_todays_trade_lines_reads_rationale_from_db(tmp_path):
    from investment_monitor.storage import RoboOrder, get_session, init_db, save_robo_order

    db = tmp_path / "t.db"
    init_db(db)
    with get_session() as s:
        save_robo_order(s, RoboOrder(
            run_id="r1", symbol="EML", side="buy", order_type="market", notional=1.94,
            source="deterministic", placed=True, status="placed",
            rationale="78% conviction — Insider cluster: 3 insiders bought $937k",
        ))
    lines = notify.todays_trade_lines(SimpleNamespace(db_path=str(db)))
    assert any("Buy EML" in line for line in lines)
    assert any("why: 78% conviction — Insider cluster" in line for line in lines)


# --- notify_run: trades ----------------------------------------------------------

_ROWS = [
    {"side": "Buy", "symbol": "AAPL", "size": "$500.00", "fill": "", "why": ""},
    {"side": "Sell", "symbol": "TSLA", "size": "3 shares", "fill": "at $210.00", "why": ""},
]


def test_live_placements_send_one_text_with_orders():
    captured = {}
    # Email configured -> the channel supports HTML, so the letter is built too.
    settings = _settings(smtp_host="smtp.gmail.com", email_to="me@x.com")
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject, html=html) or True), \
         patch.object(notify, "_placed_order_rows", return_value=list(_ROWS)), \
         patch.object(notify, "_rejected_order_rows", return_value=None):
        notify.notify_run(_result(num_placed=2), settings)
    text = captured["text"]
    assert "Live" in text
    assert "2 order(s) executed" in text
    assert "Buy AAPL — $500.00" in text
    assert "Sell TSLA — 3 shares at $210.00" in text
    assert "Portfolio value: $12,345" in text
    assert "📈" not in text and "🛑" not in text  # no emojis
    # The HTML alternative carries the same orders in the letter layout.
    assert captured["html"] is not None
    assert "AAPL" in captured["html"] and "ARCHIE" in captured["html"]


def test_no_placements_is_silent():
    with patch.object(notify, "_send") as mock_send:
        notify.notify_run(_result(num_placed=0), _settings())
    mock_send.assert_not_called()


def test_paper_placements_silent_by_default():
    with patch.object(notify, "_send") as mock_send, \
         patch.object(notify, "_placed_order_rows", return_value=list(_ROWS[:1])):
        notify.notify_run(_result(dry_run=True, num_placed=1), _settings(paper=False))
    mock_send.assert_not_called()


def test_paper_placements_sent_when_enabled():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject) or True), \
         patch.object(notify, "_placed_order_rows", return_value=list(_ROWS[:1])), \
         patch.object(notify, "_rejected_order_rows", return_value=None):
        notify.notify_run(_result(dry_run=True, num_placed=1), _settings(paper=True))
    assert "Paper" in captured["text"]


# --- notify_run: error statuses --------------------------------------------------

def test_refused_run_notifies():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject) or True):
        notify.notify_run(
            _result(status="refused", message="account has margin"), _settings()
        )
    assert "Trading Run Refused" in captured["text"]
    assert "account has margin" in captured["text"]
    assert "⛔" not in captured["text"]  # professional, no emoji


def test_failed_run_notifies():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject) or True):
        notify.notify_run(_result(status="failed", message="account fetch failed"), _settings())
    assert "Trading Run Failed" in captured["text"]
    assert "🛑" not in captured["text"]


def test_notify_run_never_raises():
    """A failure deep in formatting must never propagate into the CLI."""
    with patch.object(notify, "_placed_order_rows", side_effect=RuntimeError("db down")), \
         patch.object(notify, "_send", return_value=True):
        # Should not raise.
        notify.notify_run(_result(num_placed=1), _settings())


def test_notify_error_sends_message():
    captured = {}
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject) or True):
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
    with patch.object(notify, "_send", side_effect=lambda s, t, subject=None, html=None: captured.update(text=t, subject=subject) or True):
        assert notify.send_daily_summary(_settings(), _account()) is True
    assert "Daily Portfolio Summary" in captured["text"]
    # Email subject is brand-prefixed and never the "ARCHIE" letterhead line.
    assert captured["subject"].startswith("Archie · Daily Portfolio Summary")


# --- HTML letter gatherers (hit a real DB; values must be read before detach) ------

def test_html_gatherers_read_from_db(tmp_path):
    from datetime import date, datetime, timedelta, timezone

    from investment_monitor.storage import RoboRun, get_session, init_db
    from investment_monitor.storage.learning_models import LearningEvent

    db = tmp_path / "t.db"
    init_db(db)
    yesterday = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    with get_session() as s:
        s.add(RoboRun(run_id="r0", dry_run=True, status="completed",
                      total_value=100.0, started_at=yesterday))
        s.add(LearningEvent(kind="thesis_outcome", symbol="EML",
                            as_of_date=date.today(), direction_correct=1, brier=0.04))
    settings = SimpleNamespace(db_path=str(db))

    learning = notify._html_learning(settings)
    assert learning == {"n": 1, "win_rate": 1.0, "brier": 0.04}
    assert notify._prev_total_value(settings) == 100.0


def test_send_test_uses_send():
    with patch.object(notify, "_send", return_value=True) as mock_send:
        assert notify.send_test(_settings()) is True
    mock_send.assert_called_once()
