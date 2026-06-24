"""iMessage notification channel (macOS only).

Sends texts through Messages.app by driving it with AppleScript via ``osascript``.
The recipient and message are passed as separate ``argv`` items to the script (not
interpolated into the AppleScript source), so a message body can never break out of
the string or inject AppleScript — important because bodies carry market data.

Requirements on the host:
  * macOS with Messages.app signed in to an iMessage account.
  * Automation permission for whatever runs this (Terminal, or the launchd job's
    runner) to control Messages — System Settings > Privacy & Security > Automation.
    The first send triggers the permission prompt; grant it once.

Every method is fail-open: a send failure returns ``False`` and is logged, never
raised, so notifications can never affect trading or the monitor run.
"""

from __future__ import annotations

import asyncio
import subprocess

from loguru import logger

from .base import AlertMessage, NotificationChannel

# `on run argv` receives [recipient, message]. Sending to a `participant` of the
# iMessage service is the form that reliably reaches an arbitrary handle (phone
# number or Apple ID email), including one with no prior conversation thread.
_APPLESCRIPT = """on run argv
    set targetHandle to item 1 of argv
    set targetMessage to item 2 of argv
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant targetHandle of targetService
        send targetMessage to targetBuddy
    end tell
end run"""


class IMessageChannel(NotificationChannel):
    """Sends alerts as iMessages via macOS Messages.app.

    Args:
        recipient: Destination handle — a phone number (e.g. ``+15551234567``) or
            the Apple ID email registered with iMessage. Texting your own number is
            the common case.
        timeout: Hard cap (seconds) on the ``osascript`` call so a wedged Messages
            app can never hang the caller.
    """

    name = "imessage"

    def __init__(self, recipient: str, *, timeout: float = 15.0) -> None:
        if not recipient or not recipient.strip():
            raise ValueError(
                "iMessage recipient (phone number or Apple ID email) is required"
            )
        self._recipient = recipient.strip()
        self._timeout = timeout
        self._logger = logger.bind(component="imessage_channel")

    def send_text(self, text: str, subject: str | None = None) -> bool:
        """Send a raw string via Messages.app. Synchronous; fail-open.

        ``subject`` is accepted for a uniform channel interface but ignored — iMessage
        has no subject line. Returns True only when ``osascript`` exits 0.
        """
        try:
            proc = subprocess.run(
                ["osascript", "-e", _APPLESCRIPT, self._recipient, text],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError:
            self._logger.warning("osascript not found — iMessage only works on macOS")
            return False
        except subprocess.TimeoutExpired:
            self._logger.warning("iMessage send timed out after {t}s", t=self._timeout)
            return False
        except Exception as exc:  # noqa: BLE001 - never raise from a notification
            self._logger.exception("iMessage send error: {e}", e=str(exc))
            return False

        if proc.returncode != 0:
            self._logger.warning(
                "iMessage send failed (rc={rc}): {err}",
                rc=proc.returncode,
                err=(proc.stderr or "").strip()[:300],
            )
            return False
        self._logger.debug("iMessage sent to {to}", to=self._recipient)
        return True

    async def send(self, message: AlertMessage) -> bool:
        """Send a single alert (NotificationChannel API; offloads the blocking call)."""
        body = message.format_full() if message.ticker or message.url else (
            f"{message.title}\n{message.body}"
        )
        return await asyncio.to_thread(self.send_text, body)

    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Send a batch of alerts as one text (NotificationChannel API)."""
        if not messages:
            return True
        lines = [m.format_short() for m in messages]
        return await asyncio.to_thread(self.send_text, "\n".join(lines))
