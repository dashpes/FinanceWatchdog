"""HTML renderers for Archie's advisory correspondence.

Every email the robo advisor sends keeps its plain-text body (the canonical,
regression-locked part) and gains an HTML alternative built here. The design is
the firm's letterhead: ivory paper, deep-green ink, brass hairlines, Georgia
serif — a private letter first, with the hard data as an appendix below.

Email-client constraints honoured throughout: nested tables only (no flexbox or
grid), every style inline, ``bgcolor`` attributes doubled up for dark-mode
clients, no webfonts, no ``<style>`` blocks (Gmail clips them), total size kept
well under Gmail's ~102 KB clipping threshold.

All functions are pure — no I/O, no DB, no network — so they unit-test exactly
like ``notify.format_daily_summary``. Callers gather the data (fail-open) and
pass plain values; any section whose data is ``None`` is simply omitted.
"""

from __future__ import annotations

from decimal import Decimal
from html import escape
from typing import Any

# --- palette: Archie's stationery --------------------------------------------------
IVORY = "#F6F1E7"       # the paper
INK = "#1C2A24"         # body text
GREEN = "#1E3A2F"       # headings, the firm's deep green
BRASS = "#A9852F"       # section labels, accents
HAIRLINE = "#D9CFBA"    # rules
MUTED = "#5C665F"       # secondary text, appendix
GAIN = "#2E6B4F"        # understated green for gains
LOSS = "#8C3B2E"        # claret for losses

SERIF = "Georgia, 'Times New Roman', Times, serif"

_BODY_STYLE = f"margin:0;padding:0;background-color:{IVORY};"
_TD_BASE = f"font-family:{SERIF};color:{INK};"


# --- money formatting (mirrors notify's rules, incl. the -$0.00 guard) -------------

def _money(value: Any) -> str:
    return f"${value:,.2f}"


def _signed_money(value: Any) -> str:
    magnitude = f"{abs(value):,.2f}"
    sign = "-" if value < 0 and float(magnitude.replace(",", "")) != 0 else "+"
    return f"{sign}${magnitude}"


def _pnl_color(value: Any) -> str:
    try:
        return LOSS if value < 0 else GAIN
    except TypeError:
        return INK


def _pct(value: Any, digits: int = 1) -> str:
    return f"{float(value) * 100:+.{digits}f}%"


# --- building blocks ----------------------------------------------------------------

def _letterhead(date_str: str) -> str:
    return f"""
<tr><td align="center" style="{_TD_BASE}padding:34px 40px 0 40px;">
  <div style="font-size:24px;letter-spacing:8px;color:{GREEN};">ARCHIE</div>
  <div style="font-size:11px;letter-spacing:4px;color:{BRASS};padding-top:6px;">PERSONAL&nbsp;PRIVATE&nbsp;EQUITY</div>
  <div style="font-size:12px;font-style:italic;color:{MUTED};padding-top:10px;">{escape(date_str)}</div>
</td></tr>
<tr><td style="padding:18px 40px 0 40px;"><div style="border-top:1px solid {HAIRLINE};font-size:0;line-height:0;">&nbsp;</div></td></tr>
"""


def _title(text: str) -> str:
    return (
        f'<tr><td align="center" style="{_TD_BASE}padding:22px 40px 4px 40px;'
        f'font-size:18px;font-style:italic;color:{GREEN};">{escape(text)}</td></tr>'
    )


def _letter(paragraphs: list[str]) -> str:
    """The plain-English letter body — the novice layer, always up top."""
    ps = "".join(
        f'<p style="margin:0 0 12px 0;font-size:15px;line-height:1.65;">{p}</p>'
        for p in paragraphs
        if p
    )
    return f'<tr><td style="{_TD_BASE}padding:16px 40px 6px 40px;">{ps}</td></tr>'


def _section(label: str, inner: str, *, muted: bool = False) -> str:
    """A section: small-caps brass label over a hairline rule, then content."""
    color = MUTED if muted else BRASS
    return f"""
<tr><td style="padding:22px 40px 0 40px;">
  <div style="font-family:{SERIF};font-size:11px;letter-spacing:3px;color:{color};padding-bottom:5px;">{escape(label.upper())}</div>
  <div style="border-top:1px solid {HAIRLINE};font-size:0;line-height:0;">&nbsp;</div>
</td></tr>
<tr><td style="padding:10px 40px 0 40px;">{inner}</td></tr>
"""


def _kv_table(rows: list[tuple[str, str, str | None]]) -> str:
    """Label/value rows: (label, value_html, color or None)."""
    out = ['<table width="100%" cellpadding="0" cellspacing="0" border="0">']
    for label, value, color in rows:
        c = color or INK
        out.append(
            f'<tr><td style="{_TD_BASE}font-size:14px;padding:3px 0;">{escape(label)}</td>'
            f'<td align="right" style="font-family:{SERIF};font-size:14px;padding:3px 0;'
            f'color:{c};">{value}</td></tr>'
        )
    out.append("</table>")
    return "".join(out)


def _data_table(headers: list[str], rows: list[list[str]], *, size: int = 13) -> str:
    """A dense table for the appendix/detail sections. Cell values are pre-built HTML."""
    head = "".join(
        f'<td style="font-family:{SERIF};font-size:10px;letter-spacing:2px;color:{MUTED};'
        f'padding:4px 6px;border-bottom:1px solid {HAIRLINE};">{escape(h.upper())}</td>'
        for h in headers
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="{_TD_BASE}font-size:{size}px;padding:5px 6px;'
            f'border-bottom:1px solid {HAIRLINE};">{cell}</td>'
            for cell in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        f"<tr>{head}</tr>{body}</table>"
    )


def _signoff(mode: str | None = None) -> str:
    mode_line = ""
    if mode:
        note = (
            "Paper mode — no real money moved."
            if mode.lower() == "paper"
            else "Live trading."
        )
        mode_line = (
            f'<div style="font-size:11px;color:{MUTED};padding-top:14px;">'
            f"{escape(note)} Automated advisory correspondence.</div>"
        )
    return f"""
<tr><td style="{_TD_BASE}padding:26px 40px 8px 40px;font-size:15px;line-height:1.6;">
  Yours faithfully,<br/><span style="font-style:italic;font-size:17px;color:{GREEN};">Archie</span>
  {mode_line}
</td></tr>
<tr><td style="padding:14px 40px 34px 40px;"><div style="border-top:1px solid {HAIRLINE};font-size:0;line-height:0;">&nbsp;</div></td></tr>
"""


def _document(date_str: str, title: str, inner: str, *, mode: str | None = None) -> str:
    """Assemble the full letter: letterhead, title, content, sign-off."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body style="{_BODY_STYLE}" bgcolor="{IVORY}">
<table width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="{IVORY}" style="background-color:{IVORY};">
<tr><td align="center">
<table width="620" cellpadding="0" cellspacing="0" border="0" style="max-width:620px;width:100%;">
{_letterhead(date_str)}
{_title(title)}
{inner}
{_signoff(mode)}
</table>
</td></tr>
</table>
</body></html>"""


# --- sections shared between renderers ----------------------------------------------

def _trades_section(trade_rows: list[dict], label: str) -> str:
    rows: list[list[str]] = []
    for r in trade_rows:
        side = escape(str(r.get("side", "")))
        side_html = f'<span style="color:{GAIN if side.lower() == "buy" else LOSS};">{side}</span>'
        detail = escape(str(r.get("size", "")))
        fill = str(r.get("fill") or "")
        if fill:
            detail += f' <span style="color:{MUTED};">{escape(fill)}</span>'
        rows.append([side_html, escape(str(r.get("symbol", ""))), detail])
        why = str(r.get("why") or "").strip()
        if why:
            rows.append(
                [
                    "",
                    "",
                    f'<span style="font-style:italic;color:{MUTED};">{escape(why)}</span>',
                ]
            )
    return _section(label, _data_table(["Side", "Symbol", "Particulars"], rows))


def _snapshot_section(account: Any, realized: Any | None) -> str:
    rows: list[tuple[str, str, str | None]] = [
        ("Portfolio value", f"<b>{_money(account.total_value)}</b>", None),
        ("Cash available", _money(account.settled_cash), None),
    ]
    unrealized = account.total_unrealized_gain
    if unrealized is not None:
        basis = account.total_cost_basis
        pct = f"&nbsp;&nbsp;({_pct(unrealized / basis)})" if basis and basis > 0 else ""
        rows.append(
            ("Unrealised P&L", f"{_signed_money(unrealized)}{pct}", _pnl_color(unrealized))
        )
    if realized is not None:
        rows.append(
            (
                "Realised P&L",
                _signed_money(realized.total_realized),
                _pnl_color(realized.total_realized),
            )
        )
    return _section("Portfolio", _kv_table(rows))


def _holdings_section(account: Any, *, max_rows: int = 12) -> str | None:
    positions = [p for p in getattr(account, "positions", []) if p.quantity]
    if not positions:
        return None
    total = account.total_value
    positions = sorted(positions, key=lambda p: p.market_value, reverse=True)
    rows = []
    for p in positions[:max_rows]:
        weight = (p.market_value / total) if total else Decimal("0")
        gain = p.unrealized_gain
        gain_html = (
            f'<span style="color:{_pnl_color(gain)};">{_signed_money(gain)}</span>'
            if gain is not None
            else f'<span style="color:{MUTED};">—</span>'
        )
        rows.append(
            [
                f"<b>{escape(p.symbol)}</b>",
                _money(p.market_value),
                f"{float(weight) * 100:.1f}%",
                gain_html,
            ]
        )
    inner = _data_table(["Holding", "Value", "Weight", "Unrealised"], rows)
    if len(positions) > max_rows:
        inner += (
            f'<div style="font-family:{SERIF};font-size:11px;color:{MUTED};padding-top:6px;">'
            f"…and {len(positions) - max_rows} smaller holdings.</div>"
        )
    return _section("Holdings", inner)


def _theses_section(theses: list[dict]) -> str | None:
    if not theses:
        return None
    rows = []
    for t in theses:
        conviction = float(t.get("conviction") or 0.0)
        delta = t.get("delta")
        delta_html = ""
        if delta:
            delta_html = (
                f' <span style="color:{_pnl_color(delta)};font-size:11px;">({float(delta):+.2f})</span>'
            )
        rows.append(
            [
                f"<b>{escape(str(t.get('symbol', '')))}</b>",
                escape(str(t.get("status", ""))),
                f"{conviction:.2f}{delta_html}",
                f"{float(t.get('target_weight') or 0.0) * 100:.1f}%",
            ]
        )
        excerpt = str(t.get("excerpt") or "").strip()
        if excerpt:
            rows.append(
                [
                    f'<span style="font-style:italic;color:{MUTED};">{escape(excerpt)}</span>',
                    "",
                    "",
                    "",
                ]
            )
    inner = _data_table(["Idea", "Status", "Conviction", "Target"], rows)
    return _section("The Book — Current Theses", inner)


def _signals_section(findings: list[dict]) -> str | None:
    if not findings:
        return None
    rows = []
    for f in findings:
        kind = str(f.get("kind", "")).replace("_", " ")
        rows.append(
            [
                f"<b>{escape(str(f.get('ticker', '')))}</b>",
                escape(kind),
                f"{float(f.get('score') or 0.0):.2f}",
                f'<span style="color:{MUTED};">{escape(str(f.get("narrative") or "")[:120])}</span>',
            ]
        )
    return _section("Signals of Note", _data_table(["Ticker", "Kind", "Score", "Note"], rows))


def _appendix_section(
    account: Any,
    realized: Any | None,
    learning: dict | None,
) -> str | None:
    """The hedge-fund appendix: exposures, concentration, calibration, per-name P&L."""
    parts: list[str] = []

    try:
        total = account.total_value
        invested = account.positions_value
        weights = sorted(
            (float(p.market_value / total) for p in account.positions if total),
            reverse=True,
        )
        if total and total > 0 and weights:
            hhi = sum(w * w for w in weights)
            kv = [
                ("Invested / cash", f"{float(invested / total) * 100:.1f}% / {float(account.settled_cash / total) * 100:.1f}%", None),
                ("Largest position", f"{weights[0] * 100:.1f}%", None),
                ("Top-3 concentration", f"{sum(weights[:3]) * 100:.1f}%", None),
                ("HHI (concentration)", f"{hhi:.3f}", None),
            ]
            parts.append(_kv_table(kv))
    except Exception:  # noqa: BLE001 - the appendix must never sink the letter
        pass

    if learning:
        kv = [
            ("Thesis outcomes recorded", str(learning.get("n", 0)), None),
            ("Directional win rate", f"{float(learning.get('win_rate') or 0.0) * 100:.0f}%", None),
            ("Mean Brier score (lower is better)", f"{float(learning.get('brier') or 0.0):.3f}", None),
        ]
        parts.append(_kv_table(kv))

    if realized is not None:
        try:
            per_symbol = [
                sp for sp in realized.per_symbol.values() if sp.realized != 0
            ]
            per_symbol.sort(key=lambda sp: sp.realized, reverse=True)
            if per_symbol:
                rows = [
                    [
                        escape(sp.symbol),
                        f'<span style="color:{_pnl_color(sp.realized)};">{_signed_money(sp.realized)}</span>',
                    ]
                    for sp in per_symbol
                ]
                parts.append(_data_table(["Realised by name", "P&L"], rows, size=12))
        except Exception:  # noqa: BLE001
            pass

    if not parts:
        return None
    inner = "".join(
        f'<div style="padding-bottom:12px;">{p}</div>' for p in parts
    )
    return _section("Appendix — for the technically inclined", inner, muted=True)


# --- public renderers ----------------------------------------------------------------

def render_daily_summary(
    *,
    date_str: str,
    mode: str,
    account: Any,
    realized: Any | None = None,
    trade_rows: list[dict] | None = None,
    theses: list[dict] | None = None,
    findings: list[dict] | None = None,
    learning: dict | None = None,
    prev_total: Any | None = None,
) -> str:
    """The daily letter: summary paragraph, snapshot, dealings, book, signals, appendix."""
    # The letter — plain English, composed deterministically from the numbers.
    sentences = []
    day_move = ""
    if prev_total is not None:
        try:
            change = account.total_value - prev_total
            direction = "up" if change >= 0 else "down"
            day_move = f", {direction} {_money(abs(change))} since my last note"
        except Exception:  # noqa: BLE001 - cosmetic
            day_move = ""
    sentences.append(
        f"Good afternoon. Your portfolio stands at <b>{_money(account.total_value)}</b>{day_move}."
    )
    n_trades = len(trade_rows or [])
    if n_trades:
        plural = "order" if n_trades == 1 else "orders"
        sentences.append(
            f"I executed <b>{n_trades} {plural}</b> today; the particulars are below."
        )
    else:
        sentences.append("I did not trade today; the book rests as it was.")
    if theses:
        n_active = sum(1 for t in theses if str(t.get("status", "")).lower() == "active")
        if n_active:
            plural = "idea" if n_active == 1 else "ideas"
            sentences.append(f"I am presently minding <b>{n_active} active {plural}</b>.")
    if mode.lower() == "paper":
        sentences.append("We remain in paper mode — no real money moves.")

    body = [_letter(sentences), _snapshot_section(account, realized)]
    if trade_rows:
        body.append(_trades_section(trade_rows, "Today's Dealings"))
    holdings = _holdings_section(account)
    if holdings:
        body.append(holdings)
    theses_html = _theses_section(theses or [])
    if theses_html:
        body.append(theses_html)
    signals = _signals_section(findings or [])
    if signals:
        body.append(signals)
    appendix = _appendix_section(account, realized, learning)
    if appendix:
        body.append(appendix)

    return _document(date_str, "Daily Portfolio Summary", "".join(body), mode=mode)


def render_trade_confirmation(
    *,
    date_str: str,
    mode: str,
    trade_rows: list[dict],
    total_value: Any | None = None,
    settled_cash: Any | None = None,
    rejected_rows: list[dict] | None = None,
) -> str:
    """The trade-confirmation letter for a run that placed orders."""
    n = len(trade_rows)
    plural = "order" if n == 1 else "orders"
    sentences = [
        f"I have today executed <b>{n} {plural}</b> on your behalf, as set out below."
    ]
    if mode.lower() == "paper":
        sentences.append("This was a paper transaction — no real money moved.")

    body = [_letter(sentences), _trades_section(trade_rows, "Orders Executed")]

    if total_value is not None:
        rows = [("Portfolio value", f"<b>{_money(total_value)}</b>", None)]
        if settled_cash is not None:
            rows.append(("Cash available", _money(settled_cash), None))
        body.append(_section("Position After Dealing", _kv_table(rows)))

    if rejected_rows:
        rows = [
            [
                escape(str(r.get("symbol", ""))),
                escape(str(r.get("side", ""))),
                f'<span style="color:{MUTED};">{escape(str(r.get("reason") or ""))}</span>',
            ]
            for r in rejected_rows
        ]
        body.append(
            _section(
                "Appendix — orders the gate declined",
                _data_table(["Symbol", "Side", "Reason"], rows, size=12),
                muted=True,
            )
        )

    title = f"Trade Confirmation — {n} {plural.title()} Executed"
    return _document(date_str, title, "".join(body), mode=mode)


def render_error(*, date_str: str, title: str, message: str, mode: str | None = None) -> str:
    """A run-error notice, in the same letterhead."""
    inner = _letter(
        [
            "I must regretfully report a problem with today's run.",
        ]
    ) + _section(
        "Particulars",
        f'<div style="{_TD_BASE}font-size:14px;line-height:1.6;color:{LOSS};">{escape(message)}</div>',
    )
    return _document(date_str, title, inner, mode=mode)


def render_note(*, date_str: str, title: str, paragraphs: list[str]) -> str:
    """A short branded note (used by the notification test)."""
    return _document(date_str, title, _letter([escape(p) for p in paragraphs]))
