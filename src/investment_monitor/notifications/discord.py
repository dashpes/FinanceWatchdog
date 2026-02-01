"""Discord notification channel using webhooks."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority
from .pdf_report import PDFReportGenerator

if TYPE_CHECKING:
    from investment_monitor.models.portfolio import Portfolio


class DiscordChannel(NotificationChannel):
    """Discord notification channel using webhooks.

    Sends individual alerts as embeds and digests as embed + PDF attachment.
    """

    name = "discord"

    # Discord embed colors
    COLOR_DANGER = 0xDC3545  # Red
    COLOR_SUCCESS = 0x28A745  # Green
    COLOR_WARNING = 0xFFC107  # Amber
    COLOR_INFO = 0x17A2B8  # Cyan
    COLOR_DEFAULT = 0x6C757D  # Gray

    def __init__(self, webhook_url: str) -> None:
        """Initialize Discord channel.

        Args:
            webhook_url: Discord webhook URL.

        Raises:
            ValueError: If webhook_url is empty.
        """
        if not webhook_url:
            raise ValueError("Discord webhook URL is required")

        self._webhook_url = webhook_url
        self._pdf_generator = PDFReportGenerator()
        self._logger = logger.bind(component="discord_channel")

    async def send(self, message: AlertMessage) -> bool:
        """Send a single alert message as a Discord embed.

        Args:
            message: The alert to send.

        Returns:
            True if successful, False otherwise.
        """
        embed = self._format_alert_embed(message)
        payload = {"embeds": [embed]}

        return await self._post_webhook(payload)

    async def send_digest(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        is_weekly: bool = False,
        ai_synthesis: str | None = None,
    ) -> bool:
        """Send a digest with embed summary and PDF attachment.

        Args:
            messages: Alert messages to include.
            portfolio: Optional portfolio context.
            is_weekly: True for weekly digest, False for daily.
            ai_synthesis: Optional AI synthesis for weekly reports.

        Returns:
            True if successful, False otherwise.
        """
        # Create embed summary
        if is_weekly:
            embed = self._format_weekly_embed(messages, ai_synthesis)
            pdf_bytes = self._pdf_generator.generate_weekly_report(
                messages,
                portfolio=portfolio,
                ai_synthesis=ai_synthesis,
            )
            filename = f"weekly-report-{date.today().isoformat()}.pdf"
        else:
            embed = self._format_daily_embed(messages)
            pdf_bytes = self._pdf_generator.generate_daily_report(
                messages,
                portfolio=portfolio,
            )
            filename = f"daily-report-{date.today().isoformat()}.pdf"

        # Send with PDF attachment
        return await self._post_webhook_with_file(
            {"embeds": [embed]},
            pdf_bytes,
            filename,
        )

    async def _post_webhook(self, payload: dict[str, Any]) -> bool:
        """Post JSON payload to Discord webhook.

        Args:
            payload: JSON payload to send.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    timeout=30.0,
                )
                if response.is_success:
                    self._logger.debug("Discord webhook sent successfully")
                    return True
                else:
                    self._logger.warning(
                        "Discord webhook failed: {status}",
                        status=response.status_code,
                    )
                    return False
        except Exception as e:
            self._logger.exception("Discord webhook error: {error}", error=str(e))
            return False

    async def _post_webhook_with_file(
        self,
        payload: dict[str, Any],
        file_bytes: bytes,
        filename: str,
    ) -> bool:
        """Post payload with file attachment to Discord webhook.

        Args:
            payload: JSON payload (embeds, etc.).
            file_bytes: File contents.
            filename: Name for the attachment.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Discord requires multipart form data for file uploads
                files = {"file": (filename, file_bytes, "application/pdf")}
                data = {"payload_json": json.dumps(payload)}

                response = await client.post(
                    self._webhook_url,
                    data=data,
                    files=files,
                    timeout=60.0,
                )
                if response.is_success:
                    self._logger.debug("Discord webhook with file sent successfully")
                    return True
                else:
                    self._logger.warning(
                        "Discord webhook with file failed: {status}",
                        status=response.status_code,
                    )
                    return False
        except Exception as e:
            self._logger.exception("Discord webhook error: {error}", error=str(e))
            return False

    def _format_alert_embed(self, message: AlertMessage) -> dict[str, Any]:
        """Format an alert message as a Discord embed.

        Args:
            message: The alert to format.

        Returns:
            Discord embed dict.
        """
        # Determine color based on alert type and content
        color = self._get_alert_color(message)

        # Build title with ticker prefix
        title = f"[{message.ticker}] {message.title}" if message.ticker else message.title

        embed: dict[str, Any] = {
            "title": title,
            "description": message.body[:2000],  # Discord limit
            "color": color,
            "timestamp": message.timestamp.isoformat(),
            "footer": {"text": f"Priority: {message.priority.value.upper()}"},
        }

        if message.url:
            embed["url"] = message.url

        return embed

    def _format_daily_embed(self, messages: list[AlertMessage]) -> dict[str, Any]:
        """Format daily digest summary embed.

        Args:
            messages: Alert messages.

        Returns:
            Discord embed dict.
        """
        high_priority = [m for m in messages if m.priority == Priority.HIGH]

        # Build description
        lines = []
        if not messages:
            lines.append("No alerts for today.")
        else:
            lines.append(f"**{len(messages)} total alerts**")
            if high_priority:
                lines.append(f"\n**HIGH Priority ({len(high_priority)}):**")
                for m in high_priority[:5]:  # Limit to 5
                    ticker = f"[{m.ticker}] " if m.ticker else ""
                    lines.append(f"- {ticker}{m.title[:50]}")
                if len(high_priority) > 5:
                    lines.append(f"  ... and {len(high_priority) - 5} more")

        description = "\n".join(lines)[:2000]

        return {
            "title": f"Daily Investment Report - {date.today().strftime('%B %d, %Y')}",
            "description": description,
            "color": self.COLOR_INFO,
            "footer": {"text": "Full report attached as PDF"},
        }

    def _format_weekly_embed(
        self,
        messages: list[AlertMessage],
        ai_synthesis: str | None,
    ) -> dict[str, Any]:
        """Format weekly digest summary embed.

        Args:
            messages: Alert messages.
            ai_synthesis: Optional AI synthesis.

        Returns:
            Discord embed dict.
        """
        # Use AI synthesis as description if available
        if ai_synthesis:
            description = ai_synthesis[:2000]
        elif not messages:
            description = "No alerts this week."
        else:
            description = f"**{len(messages)} total alerts this week.**\nSee attached PDF for full details."

        return {
            "title": f"Weekly Investment Report - Week of {date.today().strftime('%B %d, %Y')}",
            "description": description,
            "color": self.COLOR_INFO,
            "footer": {"text": "Full report attached as PDF"},
        }

    def _get_alert_color(self, message: AlertMessage) -> int:
        """Determine embed color based on alert content.

        Args:
            message: The alert message.

        Returns:
            Discord color integer.
        """
        body_lower = message.body.lower()
        title_lower = message.title.lower()

        # Price alerts: red for drops, green for gains
        if message.alert_type == "price":
            if any(x in body_lower or x in title_lower for x in ["drop", "fell", "down", "-"]):
                return self.COLOR_DANGER
            if any(x in body_lower or x in title_lower for x in ["rose", "up", "gain", "+"]):
                return self.COLOR_SUCCESS

        # Insider alerts: warning color
        if message.alert_type == "insider":
            return self.COLOR_WARNING

        # Earnings: info color
        if message.alert_type == "earnings":
            return self.COLOR_INFO

        return self.COLOR_DEFAULT
