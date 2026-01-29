"""Notification manager for routing messages to channels."""

import asyncio
from collections.abc import Sequence

from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority


class NotificationManager:
    """Routes messages to appropriate channels based on priority and config.

    The manager maintains a queue of medium-priority messages for digest
    compilation and sends high-priority messages immediately.

    Priority routing:
        HIGH: Send immediately via all channels
        MEDIUM: Queue for next digest
        LOW: Log only (debug level, no channel dispatch)

    Attributes:
        channels: List of notification channels to use
        digest_queue: Queue of medium-priority messages awaiting digest
    """

    def __init__(self, channels: Sequence[NotificationChannel] | None = None) -> None:
        """Initialize the notification manager.

        Args:
            channels: List of notification channels. If None, no channels configured.
        """
        self._channels: list[NotificationChannel] = list(channels) if channels else []
        self._digest_queue: list[AlertMessage] = []
        self._logger = logger.bind(component="notification_manager")

    @property
    def channels(self) -> list[NotificationChannel]:
        """Get list of configured channels."""
        return self._channels.copy()

    @property
    def digest_queue(self) -> list[AlertMessage]:
        """Get copy of current digest queue."""
        return self._digest_queue.copy()

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.

        Args:
            channel: The channel to add
        """
        self._channels.append(channel)
        self._logger.info("Added notification channel: {name}", name=channel.name)

    def remove_channel(self, channel_name: str) -> bool:
        """Remove a channel by name.

        Args:
            channel_name: Name of the channel to remove

        Returns:
            True if channel was removed, False if not found
        """
        for i, channel in enumerate(self._channels):
            if channel.name == channel_name:
                self._channels.pop(i)
                self._logger.info("Removed notification channel: {name}", name=channel_name)
                return True
        return False

    async def notify(self, message: AlertMessage) -> None:
        """Send notification via configured channels based on priority.

        HIGH priority: Send immediately via all channels
        MEDIUM priority: Add to digest queue
        LOW priority: Debug log only

        Args:
            message: The alert message to process
        """
        self._logger.debug(
            "Processing notification: {title} (priority={priority})",
            title=message.title,
            priority=message.priority.value,
        )

        if message.priority == Priority.HIGH:
            await self._send_immediate(message)
        elif message.priority == Priority.MEDIUM:
            self._queue_for_digest(message)
        else:  # LOW
            self._log_only(message)

    async def _send_immediate(self, message: AlertMessage) -> None:
        """Send a high-priority message immediately to all channels.

        Args:
            message: The alert message to send
        """
        if not self._channels:
            self._logger.warning(
                "No channels configured for immediate notification: {title}",
                title=message.title,
            )
            return

        self._logger.info(
            "Sending immediate notification: {title}",
            title=message.title,
        )

        # Send to all channels concurrently
        tasks = []
        for channel in self._channels:
            if channel.supports_priority(message.priority):
                tasks.append(self._send_to_channel(channel, message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            self._logger.debug(
                "Sent to {success}/{total} channels",
                success=success_count,
                total=len(tasks),
            )

    async def _send_to_channel(
        self, channel: NotificationChannel, message: AlertMessage
    ) -> bool:
        """Send message to a specific channel with error handling.

        Args:
            channel: The channel to send to
            message: The message to send

        Returns:
            True if successful, False otherwise
        """
        try:
            result = await channel.send(message)
            if not result:
                self._logger.warning(
                    "Channel {name} failed to send message: {title}",
                    name=channel.name,
                    title=message.title,
                )
            return result
        except Exception as e:
            self._logger.exception(
                "Error sending to channel {name}: {error}",
                name=channel.name,
                error=str(e),
            )
            return False

    def _queue_for_digest(self, message: AlertMessage) -> None:
        """Add a medium-priority message to the digest queue.

        Args:
            message: The alert message to queue
        """
        self._digest_queue.append(message)
        self._logger.debug(
            "Queued for digest: {title} (queue size: {size})",
            title=message.title,
            size=len(self._digest_queue),
        )

    def _log_only(self, message: AlertMessage) -> None:
        """Log a low-priority message without sending to channels.

        Args:
            message: The alert message to log
        """
        ticker_context = f"[{message.ticker}] " if message.ticker else ""
        self._logger.debug(
            "Low priority alert: {ticker}{title} | {alert_type}",
            ticker=ticker_context,
            title=message.title,
            alert_type=message.alert_type,
        )

    async def send_daily_digest(self, messages: list[AlertMessage] | None = None) -> None:
        """Compile and send daily digest.

        If no messages provided, uses the internal digest queue.

        Args:
            messages: Optional list of messages. If None, uses digest queue.
        """
        digest_messages = messages if messages is not None else self._digest_queue.copy()

        if not digest_messages:
            self._logger.info("No messages for daily digest")
            return

        self._logger.info(
            "Sending daily digest with {count} messages",
            count=len(digest_messages),
        )

        if not self._channels:
            self._logger.warning("No channels configured for digest delivery")
            return

        # Send digest to all channels concurrently
        tasks = []
        for channel in self._channels:
            tasks.append(self._send_digest_to_channel(channel, digest_messages))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            self._logger.info(
                "Digest sent to {success}/{total} channels",
                success=success_count,
                total=len(tasks),
            )

        # Clear the queue after sending (only if using internal queue)
        if messages is None:
            self._digest_queue.clear()
            self._logger.debug("Digest queue cleared")

    async def _send_digest_to_channel(
        self, channel: NotificationChannel, messages: list[AlertMessage]
    ) -> bool:
        """Send digest to a specific channel with error handling.

        Args:
            channel: The channel to send to
            messages: The messages to include in digest

        Returns:
            True if successful, False otherwise
        """
        try:
            result = await channel.send_digest(messages)
            if not result:
                self._logger.warning(
                    "Channel {name} failed to send digest",
                    name=channel.name,
                )
            return result
        except Exception as e:
            self._logger.exception(
                "Error sending digest to channel {name}: {error}",
                name=channel.name,
                error=str(e),
            )
            return False

    def clear_digest_queue(self) -> int:
        """Clear the digest queue manually.

        Returns:
            Number of messages that were cleared
        """
        count = len(self._digest_queue)
        self._digest_queue.clear()
        self._logger.debug("Manually cleared {count} messages from digest queue", count=count)
        return count

    def get_digest_queue_size(self) -> int:
        """Get the current size of the digest queue.

        Returns:
            Number of messages in the digest queue
        """
        return len(self._digest_queue)
