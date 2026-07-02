"""TTL-cached live account state from the broker.

One broker call per TTL window no matter how many browser tabs poll the API
(single-flight behind an asyncio lock). On broker failure the last good
snapshot is served marked ``stale``; the router layer degrades further to the
latest persisted run when there has never been a successful fetch.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from investment_monitor.config import Settings


class AccountCache:
    def __init__(self, settings: Settings, ttl_seconds: float = 60.0) -> None:
        self._settings = settings
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        self._account: Any = None
        self._fetched_at: float = 0.0
        self._fetched_iso: str = ""

    def _fetch(self) -> Any:
        """Blocking broker call — runs in a worker thread."""
        from investment_monitor.robo.broker import PublicBroker
        from investment_monitor.robo.config import RoboConfig

        cfg = RoboConfig.from_yaml(self._settings.config_dir / "robo.yaml")
        broker = PublicBroker(
            api_token=self._settings.public_api_token,
            account_id=cfg.account_id,
            base_url=self._settings.public_api_base_url,
            dry_run=True,  # read-only snapshot; the dashboard never trades
        )
        return broker.get_account_state()

    async def get(self) -> dict:
        """{"account": AccountState|None, "stale": bool, "as_of": iso|None}."""
        async with self._lock:
            now = time.monotonic()
            if self._account is not None and (now - self._fetched_at) < self._ttl:
                return {"account": self._account, "stale": False, "as_of": self._fetched_iso}
            try:
                account = await asyncio.to_thread(self._fetch)
            except Exception as exc:  # noqa: BLE001 - degrade, never 500 the page
                logger.warning("account fetch failed (serving stale): {e}", e=exc)
                return {
                    "account": self._account,
                    "stale": self._account is not None,
                    "as_of": self._fetched_iso or None,
                }
            self._account = account
            self._fetched_at = time.monotonic()
            self._fetched_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return {"account": account, "stale": False, "as_of": self._fetched_iso}
