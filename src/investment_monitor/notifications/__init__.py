"""Notification system for investment alerts.

This module provides the infrastructure for sending alert notifications
through various channels (console, Discord, Slack, email, etc.).

Priority levels:
    HIGH: Send immediately via all channels
    MEDIUM: Include in next digest
    LOW: Log only (debug level)

Example usage:
    from investment_monitor.notifications import (
        AlertMessage,
        ConsoleChannel,
        DiscordChannel,
        NotificationManager,
        Priority,
    )

    # Create a manager with Discord output
    manager = NotificationManager([
        ConsoleChannel(),
        DiscordChannel("https://discord.com/api/webhooks/xxx/yyy"),
    ])

    # Send a high-priority alert
    await manager.notify(AlertMessage(
        title="AAPL dropped 5%",
        body="Apple stock dropped significantly today.",
        ticker="AAPL",
        alert_type="price",
        priority=Priority.HIGH,
    ))
"""

from .base import AlertMessage, NotificationChannel, Priority
from .console import ConsoleChannel
from .digest import format_daily_digest, format_weekly_digest
from .discord import DiscordChannel
from .manager import NotificationManager
from .pdf_report import PDFReportGenerator

__all__ = [
    "AlertMessage",
    "ConsoleChannel",
    "DiscordChannel",
    "NotificationChannel",
    "NotificationManager",
    "PDFReportGenerator",
    "Priority",
    "format_daily_digest",
    "format_weekly_digest",
]
