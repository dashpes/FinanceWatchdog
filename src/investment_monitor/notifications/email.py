"""Email notification channel (SMTP).

Sends notifications through any SMTP server (Gmail, Fastmail, a private relay, …).
Unlike the iMessage channel, this needs no GUI session, no Messages.app, and no
Automation grant — so it works from a background launchd *daemon* that runs whether
or not anyone is logged in. That makes it the transport of choice for the autonomous
robo advisor once it runs headless.

Gmail setup: turn on 2-Step Verification, then create an App Password (Google
Account > Security > App passwords) and use that 16-char value as ``SMTP_PASSWORD``
with ``SMTP_HOST=smtp.gmail.com`` and ``SMTP_PORT=587``. A normal account password
will not authenticate once 2FA is on.

Every method is fail-open: a send failure returns ``False`` and is logged, never
raised, so notifications can never affect trading or a monitor run.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.message import EmailMessage

from loguru import logger

from .base import AlertMessage, NotificationChannel


class EmailChannel(NotificationChannel):
    """Sends alerts as plain-text email over SMTP.

    Args:
        host: SMTP server hostname (e.g. ``smtp.gmail.com``).
        port: SMTP port. 587 → STARTTLS (default), 465 → implicit TLS (SMTPS).
        username: SMTP login user. Blank skips authentication (open relay).
        password: SMTP password / app password.
        sender: ``From`` address. Falls back to ``username`` when blank.
        recipient: ``To`` address.
        use_tls: Issue STARTTLS on a plain (587) connection. Ignored for port 465,
            which is always implicit TLS.
        timeout: Hard cap (seconds) on the whole SMTP exchange, so a wedged or
            unreachable server can never hang the caller.
    """

    name = "email"
    # The robo notify layer checks this to decide whether to build/pass an HTML
    # alternative (iMessage and other channels never see one).
    supports_html = True

    def __init__(
        self,
        *,
        host: str,
        recipient: str,
        port: int = 587,
        username: str = "",
        password: str = "",
        sender: str = "",
        use_tls: bool = True,
        timeout: float = 20.0,
    ) -> None:
        if not host or not host.strip():
            raise ValueError("SMTP host (SMTP_HOST) is required for the email channel")
        if not recipient or not recipient.strip():
            raise ValueError("email recipient (EMAIL_TO) is required")
        self._sender = (sender or username or "").strip()
        if not self._sender:
            raise ValueError(
                "email sender (EMAIL_FROM, or SMTP_USERNAME as a fallback) is required"
            )
        self._host = host.strip()
        self._port = int(port)
        self._username = (username or "").strip()
        self._password = password or ""
        self._recipient = recipient.strip()
        self._use_tls = use_tls
        self._timeout = timeout
        self._logger = logger.bind(component="email_channel")

    @staticmethod
    def _subject_and_body(text: str, subject: str | None) -> tuple[str, str]:
        """Use an explicit subject, else derive one from the first non-empty line."""
        if subject:
            return subject[:150], text
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return (first[:150] or "FinanceWatchdog robo"), text

    def _build(self, text: str, subject: str | None, html: str | None = None) -> EmailMessage:
        subj, body = self._subject_and_body(text, subject)
        msg = EmailMessage()
        msg["From"] = self._sender
        msg["To"] = self._recipient
        msg["Subject"] = subj
        msg.set_content(body)
        if html:
            # multipart/alternative: plain part first, HTML last (preferred by clients).
            msg.add_alternative(html, subtype="html")
        return msg

    def send_text(self, text: str, subject: str | None = None, *, html: str | None = None) -> bool:
        """Send an email. Synchronous; fail-open.

        ``text`` is always the canonical body; when ``html`` is given the message is
        sent as multipart/alternative so text-only clients still get the plain part.
        Returns True only when the SMTP server accepted the message for delivery.
        """
        msg = self._build(text, subject, html)
        try:
            if self._port == 465:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self._host, self._port, timeout=self._timeout, context=ctx
                ) as smtp:
                    if self._username:
                        smtp.login(self._username, self._password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                    smtp.ehlo()
                    if self._use_tls:
                        smtp.starttls(context=ssl.create_default_context())
                        smtp.ehlo()
                    if self._username:
                        smtp.login(self._username, self._password)
                    smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            self._logger.warning("email send failed: {e}", e=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - never raise from a notification
            self._logger.exception("email send error: {e}", e=str(exc))
            return False
        self._logger.debug("email sent to {to}", to=self._recipient)
        return True

    async def send(self, message: AlertMessage) -> bool:
        """Send a single alert (NotificationChannel API; offloads the blocking call)."""
        body = message.format_full() if message.ticker or message.url else (
            f"{message.title}\n{message.body}"
        )
        return await asyncio.to_thread(self.send_text, body, message.format_short())

    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Send a batch of alerts as one email (NotificationChannel API)."""
        if not messages:
            return True
        lines = [m.format_short() for m in messages]
        return await asyncio.to_thread(self.send_text, "\n".join(lines))
