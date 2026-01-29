"""Alert deduplication to prevent sending the same alert multiple times.

This module provides logic to:
1. Track sent alerts in the database
2. Define deduplication windows per alert type
3. Allow re-alerting after sufficient time passes

Example usage:
    from sqlalchemy.orm import Session
    from investment_monitor.alerts import AlertDeduplicator
    from investment_monitor.notifications import AlertMessage

    deduplicator = AlertDeduplicator(session)

    # Filter out duplicates from a list of alerts
    unique_alerts = deduplicator.filter_duplicates(alerts)

    # After sending, mark as sent
    for alert in unique_alerts:
        deduplicator.mark_sent(alert, "console")
"""

from datetime import datetime, timedelta
from hashlib import sha256

from sqlalchemy.orm import Session

from investment_monitor.notifications import AlertMessage
from investment_monitor.storage import AlertSent, alert_exists_by_dedup_key, save_alert

# Deduplication windows per alert type
# After this time passes, the same alert can be sent again
DEDUP_WINDOWS: dict[str, timedelta] = {
    "price_drop": timedelta(hours=24),
    "price_rise": timedelta(hours=24),
    "price": timedelta(hours=24),  # Generic price alert
    "volume_spike": timedelta(hours=12),
    "volume": timedelta(hours=12),  # Generic volume alert
    "insider_transaction": timedelta(days=7),
    "insider": timedelta(days=7),  # Generic insider alert
    "earnings_upcoming": timedelta(days=3),
    "earnings": timedelta(days=3),  # Generic earnings alert
    "news_keyword": timedelta(days=1),
    "news": timedelta(days=1),  # Generic news alert
    "dividend": timedelta(days=7),
    "filing": timedelta(days=7),
    "analyst": timedelta(days=1),
    "system": timedelta(hours=1),
}

# Default window for unknown alert types
DEFAULT_DEDUP_WINDOW = timedelta(hours=24)


class AlertDeduplicator:
    """Handles deduplication of alerts to prevent sending duplicates.

    This class provides methods to:
    - Generate unique deduplication keys for alerts
    - Check if an alert is a duplicate
    - Mark alerts as sent
    - Filter duplicate alerts from a list

    Attributes:
        session: SQLAlchemy database session
    """

    def __init__(self, session: Session) -> None:
        """Initialize the deduplicator.

        Args:
            session: SQLAlchemy database session for tracking sent alerts
        """
        self.session = session

    def generate_dedup_key(self, alert: AlertMessage) -> str:
        """Generate a unique key for deduplication.

        The key is based on:
        - Alert type
        - Ticker (if present)
        - A hash of the title (for uniqueness within type)

        Examples:
            - "price_drop:AAPL:abc123" (price alert for AAPL)
            - "insider:AAPL:def456" (insider alert for AAPL)
            - "system::ghi789" (system alert without ticker)

        Args:
            alert: The alert message to generate a key for

        Returns:
            A unique deduplication key string
        """
        # Create a hash of the title for uniqueness
        title_hash = sha256(alert.title.encode()).hexdigest()[:8]

        # Build the key components
        ticker = alert.ticker or ""
        key = f"{alert.alert_type}:{ticker}:{title_hash}"

        return key

    def get_dedup_window(self, alert_type: str) -> timedelta:
        """Get the deduplication window for an alert type.

        Args:
            alert_type: The type of alert

        Returns:
            The deduplication window as a timedelta
        """
        return DEDUP_WINDOWS.get(alert_type, DEFAULT_DEDUP_WINDOW)

    def is_duplicate(self, alert: AlertMessage) -> bool:
        """Check if a similar alert was sent within the dedup window.

        Args:
            alert: The alert message to check

        Returns:
            True if a similar alert was recently sent, False otherwise
        """
        dedup_key = self.generate_dedup_key(alert)
        window = self.get_dedup_window(alert.alert_type)
        hours = int(window.total_seconds() / 3600)

        return alert_exists_by_dedup_key(self.session, dedup_key, hours=hours)

    def mark_sent(self, alert: AlertMessage, channel: str) -> int:
        """Record that an alert was sent.

        Args:
            alert: The alert message that was sent
            channel: The channel through which it was sent (e.g., "console", "slack")

        Returns:
            The ID of the saved alert record
        """
        dedup_key = self.generate_dedup_key(alert)

        alert_record = AlertSent(
            alert_type=alert.alert_type,
            ticker=alert.ticker or "",
            message=alert.body,
            priority=alert.priority.value,
            channel=channel,
            dedup_key=dedup_key,
            sent_at=datetime.now(),
        )

        return save_alert(self.session, alert_record)

    def filter_duplicates(self, alerts: list[AlertMessage]) -> list[AlertMessage]:
        """Remove duplicates from a list of alerts.

        This filters out alerts that:
        1. Have already been sent within their dedup window
        2. Are duplicates of other alerts in the same batch

        Args:
            alerts: List of alert messages to filter

        Returns:
            List of unique alerts that haven't been recently sent
        """
        unique_alerts: list[AlertMessage] = []
        seen_keys: set[str] = set()

        for alert in alerts:
            # Generate key for this alert
            dedup_key = self.generate_dedup_key(alert)

            # Skip if we've already seen this key in this batch
            if dedup_key in seen_keys:
                continue

            # Skip if this alert was recently sent
            if self.is_duplicate(alert):
                continue

            # Mark as seen and add to results
            seen_keys.add(dedup_key)
            unique_alerts.append(alert)

        return unique_alerts
