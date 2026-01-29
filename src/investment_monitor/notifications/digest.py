"""Digest formatters for daily and weekly email summaries.

This module provides functions to format AlertMessage lists into readable
digests suitable for email delivery. Both plain text and HTML formats are
supported for maximum compatibility.
"""

from collections import defaultdict
from datetime import date, timedelta

from investment_monitor.models.portfolio import Portfolio
from investment_monitor.notifications.base import AlertMessage

# Unicode box-drawing characters for plain text formatting
DOUBLE_LINE = "\u2550"  # ═
SINGLE_LINE = "\u2500"  # ─
ARROW_UP = "\u25b2"  # ▲
ARROW_DOWN = "\u25bc"  # ▼


def _format_date(d: date) -> str:
    """Format a date as 'Month Day, Year'."""
    return d.strftime("%B %d, %Y")


def _format_date_range(start: date, end: date) -> str:
    """Format a date range as 'Month Day - Month Day, Year'."""
    if start.year == end.year:
        if start.month == end.month:
            return f"{start.strftime('%B %d')} - {end.day}, {end.year}"
        return f"{start.strftime('%B %d')} - {end.strftime('%B %d')}, {end.year}"
    return f"{_format_date(start)} - {_format_date(end)}"


def _group_by_type(messages: list[AlertMessage]) -> dict[str, list[AlertMessage]]:
    """Group messages by alert_type."""
    grouped: dict[str, list[AlertMessage]] = defaultdict(list)
    for msg in messages:
        grouped[msg.alert_type].append(msg)
    return dict(grouped)


def _group_by_ticker(messages: list[AlertMessage]) -> dict[str, list[AlertMessage]]:
    """Group messages by ticker (None grouped as 'General')."""
    grouped: dict[str, list[AlertMessage]] = defaultdict(list)
    for msg in messages:
        key = msg.ticker if msg.ticker else "General"
        grouped[key].append(msg)
    return dict(grouped)


def _get_summary_stats(messages: list[AlertMessage]) -> dict[str, int]:
    """Get count of messages by alert type."""
    by_type = _group_by_type(messages)
    return {alert_type: len(msgs) for alert_type, msgs in by_type.items()}


def _format_alert_type_header(alert_type: str) -> str:
    """Convert alert_type to a display header."""
    type_mappings = {
        "price": "PRICE MOVEMENTS",
        "volume": "VOLUME ALERTS",
        "insider": "INSIDER ACTIVITY",
        "news": "NEWS",
        "earnings": "EARNINGS",
        "dividend": "DIVIDENDS",
        "filing": "SEC FILINGS",
        "analyst": "ANALYST UPDATES",
        "system": "SYSTEM NOTIFICATIONS",
    }
    return type_mappings.get(alert_type.lower(), alert_type.upper())


def _format_summary_item(alert_type: str, count: int) -> str:
    """Format a summary line item."""
    type_labels = {
        "price": "price alert" if count == 1 else "price alerts",
        "volume": "volume alert" if count == 1 else "volume alerts",
        "insider": "insider transaction" if count == 1 else "insider transactions",
        "news": "relevant news item" if count == 1 else "relevant news items",
        "earnings": "earnings update" if count == 1 else "earnings updates",
        "dividend": "dividend announcement" if count == 1 else "dividend announcements",
        "filing": "SEC filing" if count == 1 else "SEC filings",
        "analyst": "analyst update" if count == 1 else "analyst updates",
        "system": "system notification" if count == 1 else "system notifications",
    }
    label = type_labels.get(alert_type.lower(), alert_type)
    return f"{count} {label}"


def _format_plain_header(title: str, width: int = 67) -> str:
    """Format a centered header with double lines."""
    line = DOUBLE_LINE * width
    centered_title = title.center(width)
    return f"{line}\n{centered_title}\n{line}"


def _format_section_header(title: str) -> str:
    """Format a section header with a single underline."""
    return f"\n{title}\n{SINGLE_LINE * len(title)}"


def _format_message_plain(msg: AlertMessage) -> str:
    """Format a single message for plain text output."""
    lines = []
    ticker_prefix = f"[{msg.ticker}] " if msg.ticker else ""
    lines.append(f"{ticker_prefix}{msg.title}")

    # Add body indented
    body_lines = msg.body.split("\n")
    for line in body_lines:
        lines.append(f"       {line}")

    if msg.url:
        lines.append(f"       Link: {msg.url}")

    return "\n".join(lines)


def _format_price_message_plain(msg: AlertMessage) -> str:
    """Format a price alert message with arrows."""
    # Try to extract price change info from the message
    # The body might contain percentage change information
    body = msg.body.lower()

    # Determine direction from body content
    if "drop" in body or "fell" in body or "down" in body or "-" in msg.title:
        arrow = ARROW_DOWN
    elif "rose" in body or "up" in body or "gain" in body or "+" in msg.title:
        arrow = ARROW_UP
    else:
        arrow = ""

    ticker = msg.ticker or "N/A"
    lines = [f"{arrow} {ticker}: {msg.title}"]

    # Add body if it provides additional context
    if msg.body and msg.body != msg.title:
        body_lines = msg.body.split("\n")
        for line in body_lines:
            lines.append(f"  {line.strip()}")

    return "\n".join(lines)


def format_daily_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio | None = None,
    date_value: date | None = None,
) -> tuple[str, str]:
    """Format messages into a daily digest.

    Args:
        messages: List of alert messages to include in the digest.
        portfolio: Optional portfolio for context (holdings info).
        date_value: Date for the digest header. Defaults to today.

    Returns:
        tuple of (plain_text, html) formatted digests.
    """
    if date_value is None:
        date_value = date.today()

    # Build plain text version
    plain_lines = []

    # Header
    plain_lines.append(_format_plain_header("INVESTMENT MONITOR DAILY DIGEST"))
    plain_lines.append(_format_date(date_value).center(67))
    plain_lines.append(DOUBLE_LINE * 67)

    if not messages:
        plain_lines.append("\nNo alerts for today.")
        plain_lines.append("")
        plain_lines.append(SINGLE_LINE * 67)
        plain_lines.append("Generated by Investment Monitor")

        plain_text = "\n".join(plain_lines)
        html = _format_daily_digest_html(messages, portfolio, date_value)
        return plain_text, html

    # Summary section
    stats = _get_summary_stats(messages)
    plain_lines.append(_format_section_header("SUMMARY"))
    for alert_type, count in sorted(stats.items()):
        plain_lines.append(f"  {_format_summary_item(alert_type, count)}")

    # Group by type and format each section
    by_type = _group_by_type(messages)

    # Define preferred order for alert types
    type_order = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]

    # Sort types: known types first in order, then unknown types alphabetically
    sorted_types = sorted(
        by_type.keys(),
        key=lambda t: (type_order.index(t.lower()) if t.lower() in type_order else len(type_order), t.lower())
    )

    for alert_type in sorted_types:
        type_messages = by_type[alert_type]
        header = _format_alert_type_header(alert_type)
        plain_lines.append(_format_section_header(header))

        # Group by ticker within each type
        by_ticker = _group_by_ticker(type_messages)

        for ticker in sorted(by_ticker.keys()):
            ticker_messages = by_ticker[ticker]
            for msg in ticker_messages:
                if alert_type.lower() == "price":
                    plain_lines.append(_format_price_message_plain(msg))
                else:
                    plain_lines.append(_format_message_plain(msg))
                plain_lines.append("")  # Empty line between messages

    # Footer
    plain_lines.append(SINGLE_LINE * 67)
    plain_lines.append("Generated by Investment Monitor")

    plain_text = "\n".join(plain_lines)
    html = _format_daily_digest_html(messages, portfolio, date_value)

    return plain_text, html


def _format_daily_digest_html(
    messages: list[AlertMessage],
    portfolio: Portfolio | None,
    date_value: date,
) -> str:
    """Format daily digest as HTML."""
    html_parts = []

    # Start HTML document
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
.container { background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 20px; }
.header h1 { margin: 0; color: #333; font-size: 24px; }
.header .date { color: #666; font-size: 14px; margin-top: 5px; }
.summary { background-color: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
.summary h2 { margin: 0 0 10px 0; font-size: 16px; color: #333; }
.summary ul { margin: 0; padding-left: 20px; }
.summary li { color: #555; margin: 5px 0; }
.section { margin-bottom: 25px; }
.section h2 { border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 15px; font-size: 18px; color: #333; }
.alert { padding: 10px; margin-bottom: 10px; border-left: 3px solid #ddd; background-color: #fafafa; }
.alert.price-up { border-left-color: #28a745; }
.alert.price-down { border-left-color: #dc3545; }
.alert.insider { border-left-color: #ffc107; }
.alert.news { border-left-color: #17a2b8; }
.alert .ticker { font-weight: bold; color: #333; }
.alert .title { font-weight: 500; }
.alert .body { color: #666; font-size: 14px; margin-top: 5px; }
.alert .url { font-size: 12px; margin-top: 5px; }
.alert .url a { color: #007bff; }
.arrow-up { color: #28a745; font-weight: bold; }
.arrow-down { color: #dc3545; font-weight: bold; }
.footer { text-align: center; color: #999; font-size: 12px; border-top: 1px solid #ddd; padding-top: 15px; margin-top: 20px; }
.no-alerts { text-align: center; color: #666; padding: 40px 0; }
</style>
</head>
<body>
<div class="container">
""")

    # Header
    html_parts.append(f"""<div class="header">
<h1>Investment Monitor Daily Digest</h1>
<div class="date">{_format_date(date_value)}</div>
</div>
""")

    if not messages:
        html_parts.append('<div class="no-alerts">No alerts for today.</div>')
    else:
        # Summary
        stats = _get_summary_stats(messages)
        html_parts.append('<div class="summary">')
        html_parts.append('<h2>Summary</h2>')
        html_parts.append('<ul>')
        for alert_type, count in sorted(stats.items()):
            html_parts.append(f'<li>{_format_summary_item(alert_type, count)}</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')

        # Sections by type
        by_type = _group_by_type(messages)
        type_order = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]
        sorted_types = sorted(
            by_type.keys(),
            key=lambda t: (type_order.index(t.lower()) if t.lower() in type_order else len(type_order), t.lower())
        )

        for alert_type in sorted_types:
            type_messages = by_type[alert_type]
            header = _format_alert_type_header(alert_type)
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h2>{header}</h2>')

            for msg in type_messages:
                alert_class = "alert"
                body = msg.body.lower()

                if alert_type.lower() == "price":
                    if "drop" in body or "fell" in body or "down" in body or "-" in msg.title:
                        alert_class += " price-down"
                        arrow = f'<span class="arrow-down">{ARROW_DOWN}</span>'
                    elif "rose" in body or "up" in body or "gain" in body or "+" in msg.title:
                        alert_class += " price-up"
                        arrow = f'<span class="arrow-up">{ARROW_UP}</span>'
                    else:
                        arrow = ""
                elif alert_type.lower() == "insider":
                    alert_class += " insider"
                    arrow = ""
                elif alert_type.lower() == "news":
                    alert_class += " news"
                    arrow = ""
                else:
                    arrow = ""

                html_parts.append(f'<div class="{alert_class}">')

                ticker_html = f'<span class="ticker">[{msg.ticker}]</span> ' if msg.ticker else ""
                title_escaped = msg.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if alert_type.lower() == "price" and arrow:
                    html_parts.append(f'<div class="title">{arrow} {ticker_html}{title_escaped}</div>')
                else:
                    html_parts.append(f'<div class="title">{ticker_html}{title_escaped}</div>')

                if msg.body:
                    body_escaped = msg.body.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    html_parts.append(f'<div class="body">{body_escaped}</div>')

                if msg.url:
                    html_parts.append(f'<div class="url"><a href="{msg.url}">More info</a></div>')

                html_parts.append('</div>')

            html_parts.append('</div>')

    # Footer
    html_parts.append("""<div class="footer">
Generated by Investment Monitor
</div>
</div>
</body>
</html>""")

    return "\n".join(html_parts)


def format_weekly_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio | None = None,
    week_start: date | None = None,
    week_end: date | None = None,
    ai_synthesis: str | None = None,
) -> tuple[str, str]:
    """Format messages into weekly digest with optional AI synthesis.

    Args:
        messages: List of alert messages from the week.
        portfolio: Optional portfolio for context.
        week_start: Start date of the week. Defaults to 7 days ago.
        week_end: End date of the week. Defaults to today.
        ai_synthesis: Optional AI-generated summary/analysis of the week.

    Returns:
        tuple of (plain_text, html) formatted digests.
    """
    if week_end is None:
        week_end = date.today()
    if week_start is None:
        week_start = week_end - timedelta(days=6)

    # Build plain text version
    plain_lines = []

    # Header
    plain_lines.append(_format_plain_header("INVESTMENT MONITOR WEEKLY DIGEST"))
    plain_lines.append(_format_date_range(week_start, week_end).center(67))
    plain_lines.append(DOUBLE_LINE * 67)

    # AI Synthesis section (if provided)
    if ai_synthesis:
        plain_lines.append(_format_section_header("WEEKLY ANALYSIS"))
        # Word wrap the AI synthesis
        synthesis_lines = ai_synthesis.split("\n")
        for line in synthesis_lines:
            plain_lines.append(f"  {line}")
        plain_lines.append("")

    if not messages:
        plain_lines.append("\nNo alerts this week.")
        plain_lines.append("")
        plain_lines.append(SINGLE_LINE * 67)
        plain_lines.append("Generated by Investment Monitor")

        plain_text = "\n".join(plain_lines)
        html = _format_weekly_digest_html(messages, portfolio, week_start, week_end, ai_synthesis)
        return plain_text, html

    # Summary section
    stats = _get_summary_stats(messages)
    plain_lines.append(_format_section_header("WEEK SUMMARY"))
    total_alerts = sum(stats.values())
    plain_lines.append(f"  Total alerts: {total_alerts}")
    for alert_type, count in sorted(stats.items()):
        plain_lines.append(f"    {_format_summary_item(alert_type, count)}")

    # Get unique tickers
    tickers = set()
    for msg in messages:
        if msg.ticker:
            tickers.add(msg.ticker)
    if tickers:
        plain_lines.append(f"  Tickers mentioned: {', '.join(sorted(tickers))}")

    # Portfolio context (if provided)
    if portfolio and portfolio.holdings:
        holding_tickers = set(portfolio.holding_tickers)
        mentioned_holdings = tickers & holding_tickers
        if mentioned_holdings:
            plain_lines.append(f"  Portfolio holdings with alerts: {', '.join(sorted(mentioned_holdings))}")

    # Group by type and format each section
    by_type = _group_by_type(messages)

    type_order = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]
    sorted_types = sorted(
        by_type.keys(),
        key=lambda t: (type_order.index(t.lower()) if t.lower() in type_order else len(type_order), t.lower())
    )

    for alert_type in sorted_types:
        type_messages = by_type[alert_type]
        header = _format_alert_type_header(alert_type)
        plain_lines.append(_format_section_header(header))

        # Group by ticker within each type
        by_ticker = _group_by_ticker(type_messages)

        for ticker in sorted(by_ticker.keys()):
            ticker_messages = by_ticker[ticker]
            if ticker != "General":
                plain_lines.append(f"  [{ticker}]")
            for msg in ticker_messages:
                if alert_type.lower() == "price":
                    formatted = _format_price_message_plain(msg)
                    # Indent for weekly digest
                    for line in formatted.split("\n"):
                        plain_lines.append(f"    {line}")
                else:
                    plain_lines.append(f"    {msg.title}")
                    if msg.body and msg.body != msg.title:
                        body_lines = msg.body.split("\n")
                        for bline in body_lines[:2]:  # Limit body lines in weekly
                            plain_lines.append(f"      {bline.strip()}")
            plain_lines.append("")  # Empty line between tickers

    # Footer
    plain_lines.append(SINGLE_LINE * 67)
    plain_lines.append("Generated by Investment Monitor")

    plain_text = "\n".join(plain_lines)
    html = _format_weekly_digest_html(messages, portfolio, week_start, week_end, ai_synthesis)

    return plain_text, html


def _format_weekly_digest_html(
    messages: list[AlertMessage],
    portfolio: Portfolio | None,
    week_start: date,
    week_end: date,
    ai_synthesis: str | None,
) -> str:
    """Format weekly digest as HTML."""
    html_parts = []

    # Start HTML document
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
.container { background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 20px; }
.header h1 { margin: 0; color: #333; font-size: 24px; }
.header .date { color: #666; font-size: 14px; margin-top: 5px; }
.ai-synthesis { background-color: #e8f4fd; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #2196F3; }
.ai-synthesis h2 { margin: 0 0 10px 0; font-size: 16px; color: #1976D2; }
.ai-synthesis .content { color: #333; line-height: 1.6; }
.summary { background-color: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
.summary h2 { margin: 0 0 10px 0; font-size: 16px; color: #333; }
.summary ul { margin: 0; padding-left: 20px; }
.summary li { color: #555; margin: 5px 0; }
.summary .total { font-weight: bold; color: #333; }
.summary .tickers { margin-top: 10px; font-size: 14px; color: #666; }
.section { margin-bottom: 25px; }
.section h2 { border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-bottom: 15px; font-size: 18px; color: #333; }
.ticker-group { margin-bottom: 15px; }
.ticker-group h3 { margin: 0 0 10px 0; font-size: 14px; color: #555; background-color: #f0f0f0; padding: 5px 10px; border-radius: 4px; display: inline-block; }
.alert { padding: 10px; margin-bottom: 10px; border-left: 3px solid #ddd; background-color: #fafafa; }
.alert.price-up { border-left-color: #28a745; }
.alert.price-down { border-left-color: #dc3545; }
.alert.insider { border-left-color: #ffc107; }
.alert.news { border-left-color: #17a2b8; }
.alert .ticker { font-weight: bold; color: #333; }
.alert .title { font-weight: 500; }
.alert .body { color: #666; font-size: 14px; margin-top: 5px; }
.alert .url { font-size: 12px; margin-top: 5px; }
.alert .url a { color: #007bff; }
.arrow-up { color: #28a745; font-weight: bold; }
.arrow-down { color: #dc3545; font-weight: bold; }
.footer { text-align: center; color: #999; font-size: 12px; border-top: 1px solid #ddd; padding-top: 15px; margin-top: 20px; }
.no-alerts { text-align: center; color: #666; padding: 40px 0; }
</style>
</head>
<body>
<div class="container">
""")

    # Header
    html_parts.append(f"""<div class="header">
<h1>Investment Monitor Weekly Digest</h1>
<div class="date">{_format_date_range(week_start, week_end)}</div>
</div>
""")

    # AI Synthesis section (if provided)
    if ai_synthesis:
        synthesis_escaped = ai_synthesis.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html_parts.append(f"""<div class="ai-synthesis">
<h2>Weekly Analysis</h2>
<div class="content">{synthesis_escaped}</div>
</div>
""")

    if not messages:
        html_parts.append('<div class="no-alerts">No alerts this week.</div>')
    else:
        # Summary
        stats = _get_summary_stats(messages)
        total_alerts = sum(stats.values())

        tickers = set()
        for msg in messages:
            if msg.ticker:
                tickers.add(msg.ticker)

        html_parts.append('<div class="summary">')
        html_parts.append('<h2>Week Summary</h2>')
        html_parts.append(f'<div class="total">Total alerts: {total_alerts}</div>')
        html_parts.append('<ul>')
        for alert_type, count in sorted(stats.items()):
            html_parts.append(f'<li>{_format_summary_item(alert_type, count)}</li>')
        html_parts.append('</ul>')
        if tickers:
            html_parts.append(f'<div class="tickers">Tickers mentioned: {", ".join(sorted(tickers))}</div>')

        if portfolio and portfolio.holdings:
            holding_tickers = set(portfolio.holding_tickers)
            mentioned_holdings = tickers & holding_tickers
            if mentioned_holdings:
                html_parts.append(f'<div class="tickers">Portfolio holdings with alerts: {", ".join(sorted(mentioned_holdings))}</div>')

        html_parts.append('</div>')

        # Sections by type
        by_type = _group_by_type(messages)
        type_order = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]
        sorted_types = sorted(
            by_type.keys(),
            key=lambda t: (type_order.index(t.lower()) if t.lower() in type_order else len(type_order), t.lower())
        )

        for alert_type in sorted_types:
            type_messages = by_type[alert_type]
            header = _format_alert_type_header(alert_type)
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h2>{header}</h2>')

            # Group by ticker
            by_ticker = _group_by_ticker(type_messages)

            for ticker in sorted(by_ticker.keys()):
                ticker_messages = by_ticker[ticker]

                if ticker != "General":
                    html_parts.append(f'<div class="ticker-group">')
                    html_parts.append(f'<h3>{ticker}</h3>')

                for msg in ticker_messages:
                    alert_class = "alert"
                    body = msg.body.lower()

                    if alert_type.lower() == "price":
                        if "drop" in body or "fell" in body or "down" in body or "-" in msg.title:
                            alert_class += " price-down"
                            arrow = f'<span class="arrow-down">{ARROW_DOWN}</span>'
                        elif "rose" in body or "up" in body or "gain" in body or "+" in msg.title:
                            alert_class += " price-up"
                            arrow = f'<span class="arrow-up">{ARROW_UP}</span>'
                        else:
                            arrow = ""
                    elif alert_type.lower() == "insider":
                        alert_class += " insider"
                        arrow = ""
                    elif alert_type.lower() == "news":
                        alert_class += " news"
                        arrow = ""
                    else:
                        arrow = ""

                    html_parts.append(f'<div class="{alert_class}">')

                    title_escaped = msg.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    if alert_type.lower() == "price" and arrow:
                        html_parts.append(f'<div class="title">{arrow} {title_escaped}</div>')
                    else:
                        html_parts.append(f'<div class="title">{title_escaped}</div>')

                    if msg.body:
                        body_escaped = msg.body.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                        html_parts.append(f'<div class="body">{body_escaped}</div>')

                    if msg.url:
                        html_parts.append(f'<div class="url"><a href="{msg.url}">More info</a></div>')

                    html_parts.append('</div>')

                if ticker != "General":
                    html_parts.append('</div>')

            html_parts.append('</div>')

    # Footer
    html_parts.append("""<div class="footer">
Generated by Investment Monitor
</div>
</div>
</body>
</html>""")

    return "\n".join(html_parts)
