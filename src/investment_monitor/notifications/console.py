"""Console notification channel using loguru."""

from datetime import datetime

from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority


class ConsoleChannel(NotificationChannel):
    """Logs alerts to console using loguru.

    Priority mapping:
        HIGH: ERROR level (red, immediate attention)
        MEDIUM: WARNING level (yellow, notable)
        LOW: DEBUG level (detailed logging only)
    """

    name = "console"

    def __init__(self) -> None:
        """Initialize the console channel."""
        self._logger = logger.bind(channel="console")

    async def send(self, message: AlertMessage) -> bool:
        """Log a single message with appropriate log level.

        Args:
            message: The alert message to log

        Returns:
            True (console logging doesn't fail)
        """
        try:
            log_method = self._get_log_method(message.priority)
            ticker_context = f"[{message.ticker}] " if message.ticker else ""

            log_method(
                "{ticker_context}{title} | {alert_type}",
                ticker_context=ticker_context,
                title=message.title,
                alert_type=message.alert_type,
            )

            # For HIGH priority, also log the body
            if message.priority == Priority.HIGH:
                log_method("{body}", body=message.body)
                if message.url:
                    log_method("More info: {url}", url=message.url)

            return True
        except Exception as e:
            # Graceful failure - log error but don't crash
            logger.exception("Failed to send console notification: {error}", error=str(e))
            return False

    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Format and log a digest of messages.

        Groups messages by priority and formats as a readable summary.

        Args:
            messages: List of alert messages to include in digest

        Returns:
            True if successful, False otherwise
        """
        if not messages:
            self._logger.info("No messages to include in digest")
            return True

        try:
            # Group messages by priority
            high_priority = [m for m in messages if m.priority == Priority.HIGH]
            medium_priority = [m for m in messages if m.priority == Priority.MEDIUM]
            low_priority = [m for m in messages if m.priority == Priority.LOW]

            # Format the digest
            digest_lines = [
                "",
                "=" * 60,
                f"DAILY DIGEST - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Total alerts: {len(messages)}",
                "=" * 60,
            ]

            if high_priority:
                digest_lines.extend(self._format_priority_section("HIGH PRIORITY", high_priority))

            if medium_priority:
                digest_lines.extend(self._format_priority_section("MEDIUM PRIORITY", medium_priority))

            if low_priority:
                digest_lines.extend(self._format_priority_section("LOW PRIORITY", low_priority))

            digest_lines.append("=" * 60)

            # Log the digest
            digest_text = "\n".join(digest_lines)
            self._logger.info("Daily digest:\n{digest}", digest=digest_text)

            return True
        except Exception as e:
            logger.exception("Failed to send digest notification: {error}", error=str(e))
            return False

    def _get_log_method(self, priority: Priority):
        """Get the appropriate loguru log method for priority."""
        if priority == Priority.HIGH:
            return self._logger.error
        elif priority == Priority.MEDIUM:
            return self._logger.warning
        else:  # LOW
            return self._logger.debug

    def _format_priority_section(
        self, header: str, messages: list[AlertMessage]
    ) -> list[str]:
        """Format a section of the digest for a priority level."""
        lines = [
            "",
            f"--- {header} ({len(messages)} alerts) ---",
        ]

        # Group by ticker
        by_ticker: dict[str | None, list[AlertMessage]] = {}
        for msg in messages:
            ticker = msg.ticker or "General"
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(msg)

        # Format each ticker group
        for ticker, ticker_messages in sorted(by_ticker.items()):
            if ticker != "General":
                lines.append(f"\n  [{ticker}]")
            else:
                lines.append(f"\n  [General]")

            for msg in ticker_messages:
                lines.append(f"    - {msg.title}")
                lines.append(f"      Type: {msg.alert_type}")
                # Truncate body for digest view
                body_preview = msg.body[:100] + "..." if len(msg.body) > 100 else msg.body
                lines.append(f"      {body_preview}")

        return lines
