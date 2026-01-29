"""Notification system for investment alerts.

This module provides the infrastructure for sending alert notifications
through various channels (console, Slack, email, etc.).

Priority levels:
    HIGH: Send immediately via all channels
    MEDIUM: Include in next digest
    LOW: Log only (debug level)

Example usage:
    from investment_monitor.notifications import (
        AlertMessage,
        ConsoleChannel,
        NotificationManager,
        Priority,
    )

    # Create a manager with console output
    manager = NotificationManager([ConsoleChannel()])

    # Send a high-priority alert
    await manager.notify(AlertMessage(
        title="AAPL dropped 5%",
        body="Apple stock dropped significantly today.",
        ticker="AAPL",
        alert_type="price",
        priority=Priority.HIGH,
    ))

    # Queue medium-priority alerts
    await manager.notify(AlertMessage(
        title="Volume spike detected",
        body="Trading volume 3x normal.",
        ticker="MSFT",
        alert_type="volume",
        priority=Priority.MEDIUM,
    ))

    # Send the daily digest
    await manager.send_daily_digest()
"""

from .base import AlertMessage, NotificationChannel, Priority
from .console import ConsoleChannel
from .digest import format_daily_digest, format_weekly_digest
from .manager import NotificationManager

__all__ = [
    "AlertMessage",
    "ConsoleChannel",
    "NotificationChannel",
    "NotificationManager",
    "Priority",
    "format_daily_digest",
    "format_weekly_digest",
]
