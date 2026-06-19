"""News collector that aggregates RSS feeds for portfolio tickers."""

import re
from datetime import datetime
from time import mktime
from typing import Any

import feedparser
from loguru import logger

from ..storage import NewsItem, news_exists, save_news_item
from .base import BaseCollector, CollectorResult


class NewsCollector(BaseCollector):
    """
    Collector for news articles from RSS feeds.

    Fetches from multiple RSS sources, matches headlines to portfolio tickers,
    deduplicates by URL, and stores with source and timestamp.
    """

    name = "news"
    rate_limit_calls = 30
    rate_limit_period = 60

    # Default RSS feeds configuration
    DEFAULT_FEEDS: list[dict[str, Any]] = [
        {
            "name": "Yahoo Finance",
            "url": "https://finance.yahoo.com/rss/headline?s={ticker}",
            "per_ticker": True,
        },
        {
            "name": "Seeking Alpha",
            "url": "https://seekingalpha.com/market_currents.xml",
            "per_ticker": False,
        },
    ]

    def __init__(self, session: Any, config: Any, feeds: list[dict[str, Any]] | None = None):
        """
        Initialize the news collector.

        Args:
            session: SQLAlchemy database session
            config: Application settings
            feeds: Optional list of RSS feed configurations. If not provided,
                   uses DEFAULT_FEEDS.
        """
        super().__init__(session, config)
        self.feeds = feeds if feeds is not None else self.DEFAULT_FEEDS

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect news from all configured RSS feeds.

        1. Fetch all configured RSS feeds
        2. For per-ticker feeds, substitute ticker symbol
        3. Parse entries and match to portfolio tickers
        4. Deduplicate and save new items

        Args:
            tickers: List of ticker symbols to collect news for

        Returns:
            CollectorResult with success status, records count, and any errors
        """
        started_at = datetime.now()
        errors: list[str] = []
        total_saved = 0
        seen_urls: set[str] = set()

        for feed_config in self.feeds:
            feed_name = feed_config["name"]
            feed_url_template = feed_config["url"]
            per_ticker = feed_config.get("per_ticker", False)

            if per_ticker:
                # Fetch feed for each ticker
                for ticker in tickers:
                    feed_url = feed_url_template.format(ticker=ticker)
                    try:
                        saved = await self._process_feed(
                            feed_url=feed_url,
                            feed_name=feed_name,
                            tickers=tickers,
                            seen_urls=seen_urls,
                            primary_ticker=ticker,
                        )
                        total_saved += saved
                    except Exception as e:
                        error_msg = f"Failed to fetch {feed_name} for {ticker}: {e}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
            else:
                # Fetch general feed once
                try:
                    saved = await self._process_feed(
                        feed_url=feed_url_template,
                        feed_name=feed_name,
                        tickers=tickers,
                        seen_urls=seen_urls,
                    )
                    total_saved += saved
                except Exception as e:
                    error_msg = f"Failed to fetch {feed_name}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

        finished_at = datetime.now()
        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=total_saved,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def collect_single(self, ticker: str) -> int:
        """
        Collect news for a single ticker.

        Args:
            ticker: Ticker symbol to collect news for

        Returns:
            Number of records saved
        """
        result = await self.collect([ticker])
        return result.records_collected

    # ----------------------------------------------------------------------- #
    # Broad, universe-independent collection (market-wide news)
    # ----------------------------------------------------------------------- #

    # $AAPL / $BRK.B style cashtag — the universe-independent way feeds name an
    # issuer. 1-5 letters with an optional .X class suffix keeps false positives
    # (e.g. "$5", "$1,000") out while still catching real tickers.
    CASHTAG_RE = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")

    async def collect_all(self) -> CollectorResult:
        """Retain market-wide news instead of filtering to a configured universe.

        ``collect(tickers)`` fetches broad RSS then NARROWS to the passed tickers;
        this broad variant keeps EVERY item with a resolvable ticker — the
        non-directional market context the insight engine wants (what is the WHOLE
        market reacting to, not just names we already hold?). Only the
        universe-independent feeds (``per_ticker`` is False) are fetched, since a
        per-ticker template needs a universe to fill in. Tickers are resolved from
        the item's own cashtags, deduped by ``NewsItem.url`` (unique), added without
        per-row flush, then committed once. Each feed fails open.
        """
        started_at = datetime.now()
        records = 0
        errors: list[str] = []
        seen_urls: set[str] = set()

        broad_feeds = [f for f in self.feeds if not f.get("per_ticker", False)]
        for feed_config in broad_feeds:
            feed_name = feed_config["name"]
            feed_url = feed_config["url"]
            try:
                added = self._retain_feed(feed_url, feed_name, seen_urls)
                records += added
                logger.info(f"{self.name}: retained {added} items from {feed_name} (broad, market-wide)")
            except Exception as e:  # noqa: BLE001 - one feed failing must not abort the rest
                error_msg = f"{feed_name} broad fetch failed: {str(e)}"
                errors.append(error_msg)
                logger.error(f"{self.name}: {error_msg}")

        try:
            self.session.commit()
        except Exception as e:  # noqa: BLE001
            self.session.rollback()
            error_msg = f"Failed to commit broad news: {str(e)}"
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

    def _retain_feed(self, feed_url: str, feed_name: str, seen_urls: set[str]) -> int:
        """Parse one broad feed and retain every item with a resolvable ticker.

        Reuses the shared ``_parse_feed`` helper. A single item naming several
        issuers (e.g. ``$AAPL`` and ``$MSFT``) is retained once per ticker, each
        with a per-ticker url so the unique constraint never collides.
        """
        entries = self._parse_feed(feed_url)
        added = 0
        for entry in entries:
            url = entry.get("link", "")
            headline = entry.get("title", "")
            if not url or not headline:
                continue

            text = f"{headline} {entry.get('summary', '')}"
            tickers = self._extract_tickers(text)
            if not tickers:
                continue  # broad keeps items with a RESOLVABLE ticker, not all noise

            published_at = self._parse_published_date(entry)
            for ticker in tickers:
                # Per-ticker url keeps the (unique) url distinct when one article
                # names multiple issuers; dedup still collapses repeats market-wide.
                row_url = url if len(tickers) == 1 else f"{url}#{ticker}"
                if row_url in seen_urls:
                    continue
                seen_urls.add(row_url)
                if news_exists(self.session, row_url):
                    continue
                self.session.add(  # no per-row flush; commit once in collect_all
                    NewsItem(
                        ticker=ticker,
                        headline=headline,
                        source=feed_name,
                        url=row_url,
                        published_at=published_at,
                    )
                )
                added += 1
        return added

    def _extract_tickers(self, text: str) -> list[str]:
        """Resolve issuer tickers from item text via cashtags (universe-independent).

        Returns de-duplicated, order-preserving uppercase symbols. Unlike
        ``_ticker_mentioned`` (which needs a candidate list), this discovers tickers
        with no portfolio to match against — what "broad" requires.
        """
        out: list[str] = []
        for sym in self.CASHTAG_RE.findall(text.upper()):
            if sym not in out:
                out.append(sym)
        return out

    async def _process_feed(
        self,
        feed_url: str,
        feed_name: str,
        tickers: list[str],
        seen_urls: set[str],
        primary_ticker: str | None = None,
    ) -> int:
        """
        Process a single RSS feed.

        Args:
            feed_url: URL of the RSS feed
            feed_name: Name of the feed source
            tickers: List of tickers to match against
            seen_urls: Set of URLs already processed (for deduplication)
            primary_ticker: If this is a per-ticker feed, the primary ticker

        Returns:
            Number of news items saved
        """
        await self._rate_limit()

        entries = self._parse_feed(feed_url)
        saved = 0

        for entry in entries:
            url = entry.get("link", "")
            if not url:
                continue

            # Deduplicate by URL within this collection run
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Check if already exists in database
            if news_exists(self.session, url):
                continue

            headline = entry.get("title", "")
            if not headline:
                continue

            # Match tickers mentioned in headline/summary
            text_to_search = f"{headline} {entry.get('summary', '')}"
            matched_tickers = self._ticker_mentioned(text_to_search, tickers)

            # If this is a per-ticker feed, include the primary ticker
            if primary_ticker and primary_ticker not in matched_tickers:
                matched_tickers.append(primary_ticker)

            # Parse publication date
            published_at = self._parse_published_date(entry)

            # Save for each matched ticker (or once if no tickers matched)
            if matched_tickers:
                for ticker in matched_tickers:
                    news_item = NewsItem(
                        ticker=ticker,
                        headline=headline,
                        source=feed_name,
                        url=url,
                        published_at=published_at,
                    )
                    save_news_item(self.session, news_item)
                    saved += 1
                    logger.debug(f"Saved news item for {ticker}: {headline[:50]}...")
            else:
                # Save without ticker association
                news_item = NewsItem(
                    ticker=None,
                    headline=headline,
                    source=feed_name,
                    url=url,
                    published_at=published_at,
                )
                save_news_item(self.session, news_item)
                saved += 1
                logger.debug(f"Saved news item (no ticker): {headline[:50]}...")

        return saved

    def _ticker_mentioned(self, text: str, tickers: list[str]) -> list[str]:
        """
        Check if any portfolio tickers are mentioned in text.

        Handles variations like $AAPL, AAPL, (AAPL), etc.

        Args:
            text: Text to search for ticker mentions
            tickers: List of tickers to look for

        Returns:
            List of matched tickers
        """
        matched = []
        text_upper = text.upper()

        for ticker in tickers:
            ticker_upper = ticker.upper()
            # Pattern matches:
            # - $AAPL (cash tag)
            # - AAPL at word boundaries
            # - (AAPL) in parentheses
            # Avoid matching partial words (e.g., "APPLET" should not match "AAPL")
            patterns = [
                rf'\${ticker_upper}\b',  # $AAPL
                rf'\b{ticker_upper}\b',  # AAPL as word
                rf'\({ticker_upper}\)',  # (AAPL)
            ]

            for pattern in patterns:
                if re.search(pattern, text_upper):
                    matched.append(ticker)
                    break  # Found this ticker, move to next

        return matched

    def _parse_feed(self, feed_url: str) -> list[dict[str, Any]]:
        """
        Parse RSS feed and return entries.

        Args:
            feed_url: URL of the RSS feed to parse

        Returns:
            List of feed entries as dictionaries

        Raises:
            Exception: If feed parsing fails
        """
        feed = feedparser.parse(feed_url)

        # Check for parsing errors
        if feed.bozo and feed.bozo_exception:
            # Some bozo exceptions are recoverable (e.g., CharacterEncodingOverride)
            if not feed.entries:
                raise Exception(f"Feed parsing error: {feed.bozo_exception}")

        return feed.entries

    def _parse_published_date(self, entry: dict[str, Any]) -> datetime | None:
        """
        Parse the publication date from a feed entry.

        Args:
            entry: Feed entry dictionary

        Returns:
            datetime object or None if parsing fails
        """
        # Try parsed time struct first
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime.fromtimestamp(mktime(entry.published_parsed))
            except (TypeError, ValueError, OverflowError):
                pass

        # Try updated_parsed as fallback
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime.fromtimestamp(mktime(entry.updated_parsed))
            except (TypeError, ValueError, OverflowError):
                pass

        return None
