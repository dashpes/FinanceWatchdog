"""Congressional Trades Collector for House and Senate stock trading disclosures."""

from datetime import date, datetime

import httpx
from loguru import logger
from sqlalchemy import select

from .base import BaseCollector, CollectorResult
from ..storage import CongressionalTrade, save_congressional_trade


class CongressTradesCollector(BaseCollector):
    """
    Collector for congressional stock trading disclosures.

    Fetches trade data from:
    - House Stock Watcher (public S3 data)
    - Senate Stock Watcher (public S3 data)

    Saves discovered trades as CongressionalTrade records, filtering by
    provided ticker list and deduplicating based on the unique constraint.
    """

    name = "congress_trades"
    rate_limit_calls = 10  # Conservative for public APIs
    rate_limit_period = 60
    max_retries = 3
    retry_delay = 2.0

    # Public API URLs (no auth required)
    HOUSE_URL = (
        "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com"
        "/data/all_transactions.json"
    )
    SENATE_URL = (
        "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
        "/aggregate/all_transactions.json"
    )

    # User agent for requests
    USER_AGENT = "InvestmentMonitor/1.0 (Congressional Trade Tracker)"

    async def fetch_house_trades(self) -> list[dict]:
        """
        Fetch House trades from the public S3 API.

        Returns:
            List of raw trade dictionaries from House Stock Watcher
        """

        async def _fetch() -> list[dict]:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.HOUSE_URL,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Fetched {len(data)} House trades from API")
                return data

        return await self._retry_with_backoff(_fetch)

    async def fetch_senate_trades(self) -> list[dict]:
        """
        Fetch Senate trades from the public S3 API.

        Returns:
            List of raw trade dictionaries from Senate Stock Watcher
        """

        async def _fetch() -> list[dict]:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.SENATE_URL,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Fetched {len(data)} Senate trades from API")
                return data

        return await self._retry_with_backoff(_fetch)

    def _normalize_trade_type(self, raw_type: str | None) -> str:
        """
        Normalize trade type to standard format (buy/sell/exchange).

        Args:
            raw_type: Raw trade type from API (e.g., "purchase", "Sale", "exchange")

        Returns:
            Normalized trade type: "buy", "sell", or "exchange"
        """
        if not raw_type:
            return "unknown"

        raw_lower = raw_type.strip().lower()

        # Map various forms to standard types
        if raw_lower in ("purchase", "buy", "bought"):
            return "buy"
        elif raw_lower in ("sale", "sell", "sold", "sale (full)", "sale (partial)"):
            return "sell"
        elif raw_lower in ("exchange",):
            return "exchange"
        else:
            # Keep original for unknown types
            return raw_lower

    def _parse_date(self, date_str: str | None) -> date | None:
        """
        Parse a date string into a date object.

        Handles multiple date formats commonly found in the APIs.

        Args:
            date_str: Date string in various formats

        Returns:
            Parsed date object or None if parsing fails
        """
        if not date_str:
            return None

        date_str = date_str.strip()
        if not date_str or date_str.lower() in ("n/a", "none", "--"):
            return None

        # Try multiple formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        logger.debug(f"Could not parse date: {date_str}")
        return None

    def parse_trade(self, raw: dict, chamber: str) -> CongressionalTrade | None:
        """
        Parse raw trade data into a CongressionalTrade object.

        Handles different field names between House and Senate APIs.

        Args:
            raw: Raw trade dictionary from API
            chamber: "House" or "Senate"

        Returns:
            CongressionalTrade object or None if parsing fails or data is invalid
        """
        try:
            # Get politician name (different field names)
            if chamber == "House":
                politician = raw.get("representative", "").strip()
            else:
                politician = raw.get("senator", "").strip()

            if not politician:
                return None

            # Get ticker - required field
            ticker = raw.get("ticker", "").strip().upper()
            if not ticker or ticker == "--" or ticker == "N/A":
                return None

            # Get trade type and normalize
            raw_trade_type = raw.get("type", "")
            trade_type = self._normalize_trade_type(raw_trade_type)

            # Get amount range - required field
            amount_range = raw.get("amount", "").strip()
            if not amount_range or amount_range == "--":
                return None

            # Get trade date - required field
            trade_date_str = raw.get("transaction_date", "")
            trade_date = self._parse_date(trade_date_str)
            if not trade_date:
                return None

            # Get disclosure date - optional
            disclosure_date_str = raw.get("disclosure_date", "")
            disclosure_date = self._parse_date(disclosure_date_str)

            # Get party - optional
            party = raw.get("party", "").strip() or None

            # Get description - optional
            description = raw.get("asset_description", "").strip() or None

            # Build source URL for tracking
            # These APIs don't provide direct URLs, so we construct a reference
            source_url = None
            if chamber == "House":
                source_url = (
                    f"https://housestockwatcher.com/summary_by_rep/{politician}"
                )
            else:
                source_url = (
                    f"https://senatestockwatcher.com/summary_by_senator/{politician}"
                )

            return CongressionalTrade(
                ticker=ticker,
                politician=politician,
                party=party,
                chamber=chamber,
                trade_type=trade_type,
                amount_range=amount_range,
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                description=description,
                source_url=source_url,
            )

        except Exception as e:
            logger.debug(f"Failed to parse trade: {e} - raw: {raw}")
            return None

    def _trade_exists(self, trade: CongressionalTrade) -> bool:
        """
        Check if a trade already exists in the database.

        Uses the unique constraint fields: ticker, politician, trade_date,
        trade_type, amount_range.

        Args:
            trade: CongressionalTrade to check

        Returns:
            True if trade exists, False otherwise
        """
        stmt = select(CongressionalTrade.id).where(
            CongressionalTrade.ticker == trade.ticker,
            CongressionalTrade.politician == trade.politician,
            CongressionalTrade.trade_date == trade.trade_date,
            CongressionalTrade.trade_type == trade.trade_type,
            CongressionalTrade.amount_range == trade.amount_range,
        )
        return self.session.scalar(stmt) is not None

    def _trade_key(self, trade: CongressionalTrade) -> tuple:
        """Dedup key matching the table's unique constraint."""
        return (
            trade.ticker, trade.politician, trade.trade_date,
            trade.trade_type, trade.amount_range,
        )

    def _existing_trade_keys(self) -> set[tuple]:
        """One-shot snapshot of existing trade keys for O(1) in-memory dedup.

        Broad ingestion processes the full House+Senate history; a per-row
        ``_trade_exists`` SELECT would be tens of thousands of queries, so we load
        all existing keys once and check membership in memory instead.
        """
        rows = self.session.execute(
            select(
                CongressionalTrade.ticker, CongressionalTrade.politician,
                CongressionalTrade.trade_date, CongressionalTrade.trade_type,
                CongressionalTrade.amount_range,
            )
        ).all()
        return {tuple(r) for r in rows}

    async def collect_all(self, *, since: date | None = None) -> CollectorResult:
        """Retain ALL congressional trades market-wide (broad multi-source collection).

        Unlike ``collect(tickers)``, this does NOT filter to a configured universe —
        it is the broad ingestion the insight engine needs (what is Congress quietly
        buying across the WHOLE market, not just names we already hold?). Dedup uses a
        single in-memory snapshot of existing keys, and rows are added without per-row
        flush then committed once, so ingesting the full history stays fast.

        Args:
            since: if set, only retain trades on/after this date (bounds run volume).
        """
        started_at = datetime.now()
        records = 0
        errors: list[str] = []
        seen = self._existing_trade_keys()

        for chamber, fetch in (
            ("House", self.fetch_house_trades),
            ("Senate", self.fetch_senate_trades),
        ):
            try:
                raw_trades = await fetch()
                added = 0
                for raw in raw_trades:
                    trade = self.parse_trade(raw, chamber)
                    if trade is None:
                        continue
                    if since is not None and trade.trade_date < since:
                        continue
                    key = self._trade_key(trade)
                    if key in seen:
                        continue
                    seen.add(key)
                    self.session.add(trade)  # no per-row flush; commit once below
                    added += 1
                records += added
                logger.info(f"{self.name}: retained {added} {chamber} trades (broad, market-wide)")
            except Exception as e:  # noqa: BLE001 - one chamber failing must not abort the other
                error_msg = f"{chamber} broad fetch failed: {str(e)}"
                errors.append(error_msg)
                logger.error(f"{self.name}: {error_msg}")

        try:
            self.session.commit()
        except Exception as e:  # noqa: BLE001
            self.session.rollback()
            error_msg = f"Failed to commit broad trades: {str(e)}"
            errors.append(error_msg)
            logger.error(f"{self.name}: {error_msg}")

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect congressional trades for given tickers.

        Fetches all trades from both House and Senate APIs, then filters
        to only save trades for the specified tickers.

        Args:
            tickers: List of ticker symbols to collect trades for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        started_at = datetime.now()
        records = 0
        errors: list[str] = []

        # Normalize tickers to uppercase for matching
        ticker_set = {t.upper() for t in tickers}

        # Fetch House trades
        try:
            house_trades = await self.fetch_house_trades()
            house_count = 0

            for raw in house_trades:
                trade = self.parse_trade(raw, "House")
                if trade and trade.ticker in ticker_set:
                    if not self._trade_exists(trade):
                        save_congressional_trade(self.session, trade)
                        house_count += 1

            records += house_count
            logger.debug(f"{self.name}: Saved {house_count} House trades")

        except Exception as e:
            error_msg = f"House trades fetch failed: {str(e)}"
            errors.append(error_msg)
            logger.error(f"{self.name}: {error_msg}")

        # Fetch Senate trades
        try:
            senate_trades = await self.fetch_senate_trades()
            senate_count = 0

            for raw in senate_trades:
                trade = self.parse_trade(raw, "Senate")
                if trade and trade.ticker in ticker_set:
                    if not self._trade_exists(trade):
                        save_congressional_trade(self.session, trade)
                        senate_count += 1

            records += senate_count
            logger.debug(f"{self.name}: Saved {senate_count} Senate trades")

        except Exception as e:
            error_msg = f"Senate trades fetch failed: {str(e)}"
            errors.append(error_msg)
            logger.error(f"{self.name}: {error_msg}")

        # Commit all changes
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            error_msg = f"Failed to commit: {str(e)}"
            errors.append(error_msg)
            logger.error(f"{self.name}: {error_msg}")

        finished_at = datetime.now()

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=records,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Collect trades for a single ticker.

        This is a convenience method that fetches all trades and filters
        to the specified ticker. Note that this still fetches all trades
        from both APIs since they don't support filtering by ticker.

        Args:
            ticker: Ticker symbol to collect trades for

        Returns:
            Number of records saved
        """
        result = await self.collect([ticker])
        return result.records_collected
