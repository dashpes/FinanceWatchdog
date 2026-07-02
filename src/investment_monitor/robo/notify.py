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


def _send(
    settings: Settings, text: str, subject: str | None = None, html: str | None = None
) -> bool:
    """Send one message via the active channel. True on success, False otherwise.

    ``subject`` sets the email subject explicitly (iMessage ignores it). It must be
    passed for any branded message, since the letterhead is the first body line and
    would otherwise become the subject. ``html`` is an optional rich alternative,
    forwarded only to channels that advertise ``supports_html`` (email); the plain
    text always remains the canonical body.
    """
    channel = _channel(settings)
    if channel is None:
        return False
    try:
        if html and getattr(channel, "supports_html", False):
            return channel.send_text(text, subject=subject, html=html)
        return channel.send_text(text, subject=subject)
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("notify failed (ignored): {e}", e=exc)
        return False


def notifications_configured(settings: Settings) -> bool:
    """True when any notification channel is configured (email or iMessage)."""
    return _channel(settings) is not None


def _wants_html(settings: Settings) -> bool:
    """True when the active channel can render an HTML alternative (email).

    Gathering the HTML letter's data touches the DB; skip all of it when the
    channel (iMessage, or none) would drop the HTML anyway.
    """
    return bool(getattr(_channel(settings), "supports_html", False))


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


def _value_line(result: RebalanceResult) -> str:
    try:
        return (
            f"Portfolio value: {_money(result.total_value)}    "
            f"Cash: {_money(result.settled_cash)}"
        )
    except Exception:  # noqa: BLE001 - cosmetic only
        return ""


def _order_row(order_row) -> dict:
    """One executed order as a structured row — the single source for text AND HTML."""
    fill = (
        f"at {_money(order_row.fill_price)}" if order_row.fill_price is not None else ""
    )
    return {
        "side": order_row.side.title(),
        "symbol": order_row.symbol,
        "size": _order_size(order_row),
        "fill": fill,
        "why": (getattr(order_row, "rationale", None) or "").strip(),
    }


def trade_text_lines(rows: list[dict]) -> list[str]:
    """Render structured order rows in the plain-text format (regression-locked)."""
    lines: list[str] = []
    for r in rows:
        fill = f" {r['fill']}" if r.get("fill") else ""
        lines.append(f"  {r['side']} {r['symbol']} — {r['size']}{fill}")
        if r.get("why"):
            lines.append(f"      why: {r['why']}")
    return lines


def _placed_order_rows(run_id: str, settings: Settings) -> list[dict]:
    """Read the orders that actually executed in a run (placed live, or simulated)."""
    from investment_monitor.storage import (
        get_robo_orders_for_run,
        get_session,
        init_db,
    )

    init_db(settings.db_path)
    with get_session() as session:
        return [
            _order_row(order_row)
            for order_row in get_robo_orders_for_run(session, run_id)
            if order_row.placed or order_row.simulated
        ]


def todays_trade_rows(settings: Settings) -> list[dict]:
    """Today's (UTC) placed/simulated orders with their rationale — for the daily summary."""
    from datetime import datetime, timezone

    from investment_monitor.storage import get_session, init_db
    from investment_monitor.storage.robo_models import RoboOrder

    init_db(settings.db_path)
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    rows: list[dict] = []
    try:
        with get_session() as session:
            for r in (
                session.query(RoboOrder)
                .filter(RoboOrder.created_at >= start)
                .order_by(RoboOrder.created_at.asc())
                .all()
            ):
                if r.placed or r.simulated:
                    rows.append(_order_row(r))
    except Exception as exc:  # noqa: BLE001 - the summary must never fail on the trades block
        logger.warning("todays_trade_rows failed (ignored): {e}", e=exc)
    return rows


def todays_trade_lines(settings: Settings) -> list[str]:
    """Today's trades as plain-text lines (kept for the CLI echo path and tests)."""
    return trade_text_lines(todays_trade_rows(settings))


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


# --- HTML letter data gatherers (each fail-open: None means "omit the section") ---

def _html_theses(settings: Settings) -> list[dict] | None:
    """Live theses as row dicts for the HTML letter (conviction delta from history)."""
    try:
        from investment_monitor.storage import get_session, init_db
        from investment_monitor.storage.thesis_operations import get_active_theses

        init_db(settings.db_path)
        rows: list[dict] = []
        with get_session() as session:
            for t in get_active_theses(session):
                delta = None
                history = t.conviction_history or []
                if len(history) >= 2:
                    try:
                        delta = float(history[-1]["conviction"]) - float(history[-2]["conviction"])
                    except (KeyError, TypeError, ValueError):
                        delta = None
                narrative = (t.narrative or "").strip()
                rows.append(
                    {
                        "symbol": t.symbol,
                        "status": t.status,
                        "conviction": t.conviction,
                        "delta": delta,
                        "target_weight": t.target_weight,
                        "excerpt": narrative[:140] + ("…" if len(narrative) > 140 else ""),
                    }
                )
        return rows
    except Exception as exc:  # noqa: BLE001 - a missing section, never a failed letter
        logger.debug("theses section unavailable (ignored): {e}", e=exc)
        return None


def _html_findings(settings: Settings, *, days: int = 2, limit: int = 5) -> list[dict] | None:
    """Recent confluence findings for the 'Signals of Note' section."""
    try:
        from investment_monitor.storage import get_session, init_db
        from investment_monitor.storage.insight_operations import get_recent_findings

        init_db(settings.db_path)
        with get_session() as session:
            return [
                {
                    "ticker": f.ticker,
                    "kind": f.kind,
                    "score": f.score,
                    "narrative": (f.narrative or "").strip(),
                }
                for f in get_recent_findings(session, limit=limit, max_age_days=days)
            ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("signals section unavailable (ignored): {e}", e=exc)
        return None


def _html_learning(settings: Settings) -> dict | None:
    """Account-wide calibration aggregate (win rate + Brier) for the appendix."""
    try:
        from investment_monitor.storage import get_session, init_db
        from investment_monitor.storage.learning_models import (
            LEARNING_KIND_OUTCOME,
            LearningEvent,
        )

        init_db(settings.db_path)
        with get_session() as session:
            events = (
                session.query(LearningEvent)
                .filter(LearningEvent.kind == LEARNING_KIND_OUTCOME)
                .all()
            )
        if not events:
            return None
        hits = [int(e.direction_correct or 0) for e in events]
        briers = [float(e.brier) for e in events if e.brier is not None]
        return {
            "n": len(events),
            "win_rate": sum(hits) / len(hits),
            "brier": (sum(briers) / len(briers)) if briers else None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("learning section unavailable (ignored): {e}", e=exc)
        return None


def _prev_total_value(settings: Settings):
    """Total value from the last run before today (UTC) — the 'since my last note' delta."""
    try:
        from datetime import datetime, timezone

        from investment_monitor.storage import get_session, init_db
        from investment_monitor.storage.robo_models import RoboRun

        init_db(settings.db_path)
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        with get_session() as session:
            row = (
                session.query(RoboRun)
                .filter(RoboRun.started_at < start, RoboRun.total_value.isnot(None))
                .order_by(RoboRun.started_at.desc())
                .first()
            )
        return row.total_value if row is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("prev total unavailable (ignored): {e}", e=exc)
        return None


def _rejected_order_rows(run_id: str, settings: Settings) -> list[dict] | None:
    """Orders the gate declined in a run — the confirmation letter's appendix."""
    try:
        from investment_monitor.storage import (
            get_robo_orders_for_run,
            get_session,
            init_db,
        )

        init_db(settings.db_path)
        with get_session() as session:
            return [
                {
                    "symbol": o.symbol,
                    "side": o.side.title(),
                    "reason": (o.gate_reason or o.preflight_reason or "").strip(),
                }
                for o in get_robo_orders_for_run(session, run_id)
                if o.gate_accepted is False or (o.gate_accepted and o.preflight_ok is False)
            ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("rejected orders unavailable (ignored): {e}", e=exc)
        return None


# --- public triggers (all fail-open) ----------------------------------------------

def notify_run(result: RebalanceResult, settings: Settings) -> None:
    """Notify on a finished run: an error status, or the trades that were placed."""
    try:
        from investment_monitor.robo import email_html

        status = (result.status or "").lower()
        today = _british_date(date.today())
        if status in _ERROR_STATUSES:
            title = f"Trading Run {status.title()} ({_mode(result.dry_run)})"
            detail = result.message or result.summary_line()
            html = None
            try:
                if _wants_html(settings):
                    html = email_html.render_error(
                        date_str=today, title=title, message=detail, mode=_mode(result.dry_run)
                    )
            except Exception as exc:  # noqa: BLE001 - HTML is optional garnish
                logger.debug("error HTML skipped: {e}", e=exc)
            _send(settings, _compose(title, detail), subject=_subject(title), html=html)
            return

        if result.num_placed <= 0:
            return  # nothing executed — stay quiet
        if result.dry_run and not settings.imessage_notify_paper:
            return  # paper placements are silent unless explicitly enabled

        title = (
            f"Trade Confirmation — {result.num_placed} order(s) executed "
            f"({_mode(result.dry_run)})"
        )
        rows = _placed_order_rows(result.run_id, settings)
        body_lines = trade_text_lines(rows)
        value = _value_line(result)
        if value:
            body_lines.extend(["", value])
        html = None
        try:
            if _wants_html(settings):
                html = email_html.render_trade_confirmation(
                    date_str=today,
                    mode=_mode(result.dry_run),
                    trade_rows=rows,
                    total_value=result.total_value,
                    settled_cash=result.settled_cash,
                    rejected_rows=_rejected_order_rows(result.run_id, settings),
                )
        except Exception as exc:  # noqa: BLE001 - HTML is optional garnish
            logger.debug("confirmation HTML skipped: {e}", e=exc)
        _send(settings, _compose(title, "\n".join(body_lines)), subject=_subject(title), html=html)
    except Exception as exc:  # noqa: BLE001 - never let a notification break the CLI
        logger.warning("robo notify_run failed (ignored): {e}", e=exc)


def notify_error(settings: Settings, *, message: str, dry_run: bool | None = None) -> None:
    """Notify on a broker/exception error raised before a result exists."""
    try:
        from investment_monitor.robo import email_html

        mode = f" ({_mode(dry_run)})" if dry_run is not None else ""
        title = f"Trading Run Error{mode}"
        html = None
        try:
            if _wants_html(settings):
                html = email_html.render_error(
                    date_str=_british_date(date.today()),
                    title=title,
                    message=message,
                    mode=_mode(dry_run) if dry_run is not None else None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("error HTML skipped: {e}", e=exc)
        _send(settings, _compose(title, message), subject=_subject(title), html=html)
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("robo notify_error failed (ignored): {e}", e=exc)


def send_daily_summary(
    settings: Settings,
    account: AccountState,
    realized: RealizedPnL | None = None,
    trades: list[str] | None = None,
    *,
    trade_rows: list[dict] | None = None,
    dry_run: bool | None = None,
) -> bool:
    """Send the daily portfolio/P&L summary. Returns True if sent.

    ``trades`` (pre-formatted lines) remains the plain-text input; ``trade_rows``
    (structured dicts from :func:`todays_trade_rows`) additionally feeds the HTML
    letter. Every extra HTML section is gathered fail-open — a data problem can
    only ever shrink the letter, never block the summary.
    """
    try:
        title = f"Daily Portfolio Summary · {_british_date(date.today())}"
        if trades is None and trade_rows is not None:
            trades = trade_text_lines(trade_rows)
        html = None
        try:
            if _wants_html(settings):
                from investment_monitor.robo import email_html

                html = email_html.render_daily_summary(
                    date_str=_british_date(date.today()),
                    mode=_mode(dry_run) if dry_run is not None else "Paper",
                    account=account,
                    realized=realized,
                    trade_rows=trade_rows,
                    theses=_html_theses(settings),
                    findings=_html_findings(settings),
                    learning=_html_learning(settings),
                    prev_total=_prev_total_value(settings),
                )
        except Exception as exc:  # noqa: BLE001 - HTML is optional garnish
            logger.debug("daily summary HTML skipped: {e}", e=exc)
        return _send(
            settings,
            format_daily_summary(account, realized, trades),
            subject=_subject(title),
            html=html,
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
    html = None
    try:
        if _wants_html(settings):
            from investment_monitor.robo import email_html

            html = email_html.render_note(
                date_str=_british_date(date.today()),
                title=title,
                paragraphs=[
                    "This confirms your advisory notifications are configured correctly.",
                    "I shall be in touch with trade confirmations and your daily summary.",
                ],
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("test HTML skipped: {e}", e=exc)
    return _send(settings, _compose(title, body), subject=_subject(title), html=html)
