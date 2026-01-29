"""Base classes for notification system."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Alert priority levels.

    HIGH: Send immediately via all channels
    MEDIUM: Include in next digest
    LOW: Log only (debug level)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertMessage(BaseModel):
    """A notification message to be sent through channels.

    Attributes:
        title: Brief summary of the alert
        body: Detailed alert information
        ticker: Stock ticker if applicable
        alert_type: Category of alert (price, volume, insider, etc.)
        priority: Message priority level
        url: Optional link for more information
        timestamp: When the alert was generated
    """

    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)
    ticker: str | None = None
    alert_type: str = Field(..., min_length=1)
    priority: Priority = Priority.MEDIUM
    url: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def format_short(self) -> str:
        """Format message as a single line summary."""
        ticker_prefix = f"[{self.ticker}] " if self.ticker else ""
        return f"{ticker_prefix}{self.title}"

    def format_full(self) -> str:
        """Format message with full details."""
        lines = [self.format_short(), "-" * 40, self.body]
        if self.url:
            lines.append(f"More info: {self.url}")
        return "\n".join(lines)


class NotificationChannel(ABC):
    """Abstract base class for notification channels.

    Subclasses must implement send() and send_digest() methods.
    Each channel handles its own error handling and should not raise
    exceptions - instead return False on failure.
    """

    name: str = "base"

    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """Send a single message.

        Args:
            message: The alert message to send

        Returns:
            True if successful, False otherwise
        """
        ...

    @abstractmethod
    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Send a batch of messages as a digest.

        Args:
            messages: List of alert messages to include in digest

        Returns:
            True if successful, False otherwise
        """
        ...

    def supports_priority(self, priority: Priority) -> bool:
        """Check if this channel handles the given priority level.

        Override in subclasses to customize priority handling.
        Default: supports all priorities.
        """
        return True
