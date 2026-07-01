"""iMessage notifications for the robo advisor's live runs.

A thin, synchronous, fail-open bridge between the (sync) robo CLI and the iMessage
channel. Three triggers, matching what the operator asked to be notified about:

  * trades executed — every BUY/SELL actually placed in a run (live; paper only when
    ``IMESSAGE_NOTIFY_PAPER`` is set);
  * run errors      — a run that refused (margin!), failed, or errored, plus broker
    errors raised before a result exists;
  * daily summary   — portfolio value and P&L, sent by the ``daily-summary`` command.

Design rules:
  * Never raise. A notification problem must never affect trading — every public
    function swallows its own exceptions.
  * Silent when unconfigured. With ``IMESSAGE_TO`` blank there is no channel and the
    functions are no-ops, so the feature is opt-in and the default path is byte-identical.
  * Read placed orders from the DB by ``run_id`` rather than threading state through
    the rebalance core — the trading pipeline needs no changes to support this.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from loguru import logger

from investment_monitor.config import Settings

if TYPE_CHECKING:
    from investment_monitor.robo.models import AccountState
    from investment_monitor.robo.pnl import RealizedPnL
    from investment_monitor.robo.rebalance import RebalanceResult

# Statuses that mean "this run did not complete normally" — always worth a ping,
# regardless of mode (a silently dead or refused trader is the worst outcome).
_ERROR_STATUSES = {"failed", "refused", "errored"}


# --- channel + send (fail-open) ---------------------------------------------------

def _channel(settings: Settings):
    """Build the notification channel, or None when unconfigured/unavailable.

    Email (SMTP) is preferred because it works from a headless launchd daemon with
    no GUI/Automation dependency; iMessage is a fallback for a logged-in Mac.
    """
    # Email (SMTP) — headless, daemon-safe. Active when host + recipient are set.
    if (settings.smtp_host or "").strip() and (settings.email_to or "").strip():
        try:
            from investment_monitor.notifications.email import EmailChannel

            return EmailChannel(
                host=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                sender=(settings.email_from or settings.smtp_username),
                recipient=settings.email_to,
                use_tls=settings.email_use_tls,
            )
        except Exception as exc:  # noqa: BLE001 - setup must never break a run
            logger.warning("email channel unavailable (ignored): {e}", e=exc)

    # iMessage — GUI fallback; only delivers while logged in with Automation granted.
    to = (settings.imessage_to or "").strip()
    if to:
        try:
            from investment_monitor.notifications.imessage import IMessageChannel

            return IMessageChannel(to)
        except Exception as exc:  # noqa: BLE001 - setup must never break a run
            logger.warning("iMessage channel unavailable (ignored): {e}", e=exc)
    return None


def _send(settings: Settings, text: str, subject: str | None = None) -> bool:
    """Send one message via the active channel. True on success, False otherwise.

    ``subject`` sets the email subject explicitly (iMessage ignores it). It must be
    passed for any branded message, since the letterhead is the first body line and
    would otherwise become the subject.
    """
    channel = _channel(settings)
    if channel is None:
        return False
    try:
        return channel.send_text(text, subject=subject)
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("notify failed (ignored): {e}", e=exc)
        return False


def notifications_configured(settings: Settings) -> bool:
    """True when any notification channel is configured (email or iMessage)."""
    return _channel(settings) is not None


# --- Archie: the advisor's persona / branding ------------------------------------
# Change ADVISOR_NAME (and the firm line) to rebrand; everything else flows from it.
ADVISOR_NAME = "Archie"
_FIRM_LINE = "Personal Private Equity"
_RULE = "─" * 30


def _british_date(d: date) -> str:
    """British long form, e.g. '23 June 2026'."""
    return f"{d.day} {d.strftime('%B %Y')}"


def _subject(title: str) -> str:
    """Email subject, brand-prefixed so every note is recognisably from Archie."""
    return f"{ADVISOR_NAME} · {title}"


def _compose(title: str, body: str) -> str:
    """Wrap a message body in Archie's letterhead + sign-off."""
    return (
        f"{ADVISOR_NAME.upper()}\n{_FIRM_LINE}\n{_RULE}\n"
        f"{title}\n\n{body}\n\n"
        f"Yours faithfully,\n{ADVISOR_NAME}"
    )


# --- formatting helpers -----------------------------------------------------------

def _mode(dry_run: bool) -> str:
    return "Paper" if dry_run else "Live"


def _money(value) -> str:
    """Format a dollar amount with cents: $1,234.56."""
    return f"${value:,.2f}"


def _signed_money(value) -> str:
    """Format a signed dollar amount: +$0.04 / -$12.50.

    The sign is decided AFTER rounding to cents: a tiny magnitude that rounds to
    0.00 (e.g. Decimal('-0.004')) must render '+$0.00', never a misleading '-$0.00'.
    """
    magnitude = f"{abs(value):,.2f}"
    sign = "-" if value < 0 and float(magnitude.replace(",", "")) != 0 else "+"
    return f"{sign}${magnitude}"


def _order_size(order_row) -> str:
    if order_row.notional is not None:
        return _money(order_row.notional)
    if order_row.quantity is not None:
        return f"{order_row.quantity:g} shares"
    return "n/a"


def _fill_suffix(order_row) -> str:
    if order_row.fill_price is not None:
        return f" at {_money(order_row.fill_price)}"
    return ""


def _value_line(result: RebalanceResult) -> str:
    try:
        return (
            f"Portfolio value: {_money(result.total_value)}    "
            f"Cash: {_money(result.settled_cash)}"
        )
    except Exception:  # noqa: BLE001 - cosmetic only
        return ""


def _placed_order_lines(run_id: str, settings: Settings) -> list[str]:
    """Read the orders that actually executed in a run (placed live, or simulated)."""
    from investment_monitor.storage import (
        get_robo_orders_for_run,
        get_session,
        init_db,
    )

    init_db(settings.db_path)
    lines: list[str] = []
    with get_session() as session:
        for order_row in get_robo_orders_for_run(session, run_id):
            if not (order_row.placed or order_row.simulated):
                continue
            lines.append(
                f"  {order_row.side.title()} {order_row.symbol} — "
                f"{_order_size(order_row)}{_fill_suffix(order_row)}"
            )
            # The investment 'why', when the order had an owning thesis.
            why = (getattr(order_row, "rationale", None) or "").strip()
            if why:
                lines.append(f"      why: {why}")
    return lines


def todays_trade_lines(settings: Settings) -> list[str]:
    """Today's (UTC) placed/simulated orders with their rationale — for the daily summary."""
    from datetime import datetime, timezone

    from investment_monitor.storage import get_session, init_db
    from investment_monitor.storage.robo_models import RoboOrder

    init_db(settings.db_path)
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    lines: list[str] = []
    try:
        with get_session() as session:
            rows = (
                session.query(RoboOrder)
                .filter(RoboOrder.created_at >= start)
                .order_by(RoboOrder.created_at.asc())
                .all()
            )
            for r in rows:
                if not (r.placed or r.simulated):
                    continue
                lines.append(f"  {r.side.title()} {r.symbol} — {_order_size(r)}{_fill_suffix(r)}")
                why = (getattr(r, "rationale", None) or "").strip()
                if why:
                    lines.append(f"      why: {why}")
    except Exception as exc:  # noqa: BLE001 - the summary must never fail on the trades block
        logger.warning("todays_trade_lines failed (ignored): {e}", e=exc)
    return lines


def format_daily_summary(
    account: AccountState,
    realized: RealizedPnL | None = None,
    trades: list[str] | None = None,
) -> str:
    """Build the daily-summary text (pure: no I/O, no DB, no network).

    ``trades`` is a pre-formatted list of today's trade lines (with their 'why'); when
    non-empty it is shown as a "Today's trades" section so the summary explains what the
    advisor did and why, not just the ending balances.
    """
    body = [
        f"{'Portfolio value':<18}{_money(account.total_value)}",
        f"{'Cash available':<18}{_money(account.settled_cash)}",
    ]
    unrealized = account.total_unrealized_gain
    if unrealized is not None:
        basis = account.total_cost_basis
        pct = f"  ({unrealized / basis * 100:+.1f}%)" if basis and basis > 0 else ""
        body.append(f"{'Unrealised P&L':<18}{_signed_money(unrealized)}{pct}")
    if realized is not None:
        body.append(f"{'Realised P&L':<18}{_signed_money(realized.total_realized)}")

    if trades:
        body.append("")
        body.append("Today's trades:")
        body.extend(trades)

    movers = [p for p in account.positions if p.unrealized_gain is not None]
    movers.sort(key=lambda p: abs(p.unrealized_gain), reverse=True)
    if movers:
        body.append("")
        body.append("Top movers:")
        body.extend(f"  {p.symbol}: {_signed_money(p.unrealized_gain)}" for p in movers[:5])

    title = f"Daily Portfolio Summary · {_british_date(date.today())}"
    return _compose(title, "\n".join(body))


# --- public triggers (all fail-open) ----------------------------------------------

def notify_run(result: RebalanceResult, settings: Settings) -> None:
    """Notify on a finished run: an error status, or the trades that were placed."""
    try:
        status = (result.status or "").lower()
        if status in _ERROR_STATUSES:
            title = f"Trading Run {status.title()} ({_mode(result.dry_run)})"
            detail = result.message or result.summary_line()
            _send(settings, _compose(title, detail), subject=_subject(title))
            return

        if result.num_placed <= 0:
            return  # nothing executed — stay quiet
        if result.dry_run and not settings.imessage_notify_paper:
            return  # paper placements are silent unless explicitly enabled

        title = (
            f"Trade Confirmation — {result.num_placed} order(s) executed "
            f"({_mode(result.dry_run)})"
        )
        body_lines = list(_placed_order_lines(result.run_id, settings))
        value = _value_line(result)
        if value:
            body_lines.extend(["", value])
        _send(settings, _compose(title, "\n".join(body_lines)), subject=_subject(title))
    except Exception as exc:  # noqa: BLE001 - never let a notification break the CLI
        logger.warning("robo notify_run failed (ignored): {e}", e=exc)


def notify_error(settings: Settings, *, message: str, dry_run: bool | None = None) -> None:
    """Notify on a broker/exception error raised before a result exists."""
    try:
        mode = f" ({_mode(dry_run)})" if dry_run is not None else ""
        title = f"Trading Run Error{mode}"
        _send(settings, _compose(title, message), subject=_subject(title))
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("robo notify_error failed (ignored): {e}", e=exc)


def send_daily_summary(
    settings: Settings,
    account: AccountState,
    realized: RealizedPnL | None = None,
    trades: list[str] | None = None,
) -> bool:
    """Send the daily portfolio/P&L summary. Returns True if sent."""
    try:
        title = f"Daily Portfolio Summary · {_british_date(date.today())}"
        return _send(
            settings, format_daily_summary(account, realized, trades), subject=_subject(title)
        )
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("robo daily summary notify failed (ignored): {e}", e=exc)
        return False


def send_test(settings: Settings) -> bool:
    """Send a test message so the operator can verify notification setup."""
    title = "Notification Test"
    body = (
        "This confirms your advisory notifications are configured correctly.\n"
        "I shall be in touch with trade confirmations and your daily summary."
    )
    return _send(settings, _compose(title, body), subject=_subject(title))
