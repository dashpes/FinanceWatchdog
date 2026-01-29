"""AI-enhanced news processing for relevance scoring and filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from investment_monitor.storage import NewsItem, get_recent_news, get_unscored_news

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from investment_monitor.models import Portfolio

    from .local_llm import LocalLLM


class NewsProcessor:
    """Process news items using local LLM for relevance scoring and filtering.

    This processor scores news items for relevance using a local LLM,
    persists scores to the database, and filters news based on relevance
    thresholds for alerts and digests.

    Example usage:
        processor = NewsProcessor(
            session=db_session,
            llm=local_llm,
            portfolio=portfolio,
            min_relevance=5.0,
        )

        # Score unscored news items
        count = await processor.process_unscored_news()
        print(f"Scored {count} news items")

        # Get relevant news for alerts
        relevant = await processor.get_relevant_news(ticker="AAPL", hours=24)
    """

    def __init__(
        self,
        session: Session,
        llm: LocalLLM,
        portfolio: Portfolio,
        min_relevance: float = 5.0,
    ) -> None:
        """Initialize the NewsProcessor.

        Args:
            session: SQLAlchemy session for database operations.
            llm: LocalLLM instance for scoring relevance.
            portfolio: Portfolio for getting investment context.
            min_relevance: Minimum relevance score (0-10) for filtering.
                          Items below this threshold are filtered out.
        """
        self.session = session
        self.llm = llm
        self.portfolio = portfolio
        self.min_relevance = min_relevance

    async def process_unscored_news(self, batch_size: int = 100) -> int:
        """Find news items without relevance scores and score them.

        Retrieves unscored news items from the database, scores each using
        the local LLM, and persists the scores back to the database.

        Args:
            batch_size: Maximum number of items to process in one call.
                       Default is 100 to handle high volume efficiently.

        Returns:
            Number of items successfully processed and scored.
        """
        items = get_unscored_news(self.session, limit=batch_size)

        if not items:
            logger.debug("No unscored news items found")
            return 0

        processed = 0
        skipped = 0

        for item in items:
            try:
                score = await self._score_item(item)
                item.relevance_score = score

                if score < 0:
                    skipped += 1
                    logger.debug(f"Skipped scoring for: {item.headline[:50]}...")
                else:
                    processed += 1
                    logger.debug(
                        f"Scored {score:.1f}: {item.headline[:50]}..."
                    )
            except Exception as e:
                logger.warning(f"Error scoring news item {item.id}: {e}")
                skipped += 1
                continue

        # Commit all changes
        try:
            self.session.commit()
        except Exception as e:
            logger.error(f"Failed to commit news scores: {e}")
            self.session.rollback()
            return 0

        if processed > 0:
            logger.info(f"Scored {processed} news items, skipped {skipped}")
        elif skipped > 0:
            logger.info(f"Skipped {skipped} news items (LLM unavailable)")

        return processed

    async def get_relevant_news(
        self,
        ticker: str | None = None,
        hours: int = 24,
    ) -> list[NewsItem]:
        """Get news items above the relevance threshold.

        Retrieves recent news from the database and filters to include
        only items with relevance scores at or above the minimum threshold.
        Unscored items (relevance_score is None) are excluded.

        Args:
            ticker: Optional ticker to filter by. If None, returns all.
            hours: Number of hours to look back. Default is 24.

        Returns:
            List of NewsItem objects with relevance >= min_relevance,
            sorted by most recent first.
        """
        items = get_recent_news(self.session, ticker=ticker, hours=hours)

        relevant = [
            item
            for item in items
            if item.relevance_score is not None
            and item.relevance_score >= self.min_relevance
        ]

        logger.debug(
            f"Found {len(relevant)}/{len(items)} relevant news items "
            f"(min_relevance={self.min_relevance})"
        )

        return relevant

    async def get_news_by_priority(
        self,
        ticker: str | None = None,
        hours: int = 24,
        min_score: float | None = None,
    ) -> list[NewsItem]:
        """Get news items sorted by relevance score (highest first).

        This method retrieves news that meets the minimum score requirement
        and returns them sorted by relevance, useful for prioritizing
        which news to include in alerts or digests.

        Args:
            ticker: Optional ticker to filter by.
            hours: Number of hours to look back.
            min_score: Minimum score override. If None, uses min_relevance.

        Returns:
            List of NewsItem objects sorted by relevance_score descending.
        """
        threshold = min_score if min_score is not None else self.min_relevance
        items = get_recent_news(self.session, ticker=ticker, hours=hours)

        # Filter and sort by relevance
        relevant = [
            item
            for item in items
            if item.relevance_score is not None and item.relevance_score >= threshold
        ]

        # Sort by relevance score descending (highest first)
        relevant.sort(key=lambda x: x.relevance_score or 0, reverse=True)

        return relevant

    async def _score_item(self, item: NewsItem) -> float:
        """Score a single news item for relevance.

        Uses the local LLM to evaluate how relevant the news headline
        is to the ticker/company it's associated with.

        Args:
            item: The NewsItem to score.

        Returns:
            Relevance score from 0-10, or -1 if LLM unavailable.
        """
        if not self.llm.is_available():
            return -1.0

        # Get company name from portfolio thesis if available
        company_name = ""
        if item.ticker:
            thesis = self.portfolio.get_thesis(item.ticker)
            if thesis:
                # Use thesis as context hint for better relevance scoring
                company_name = f"(thesis: {thesis})"

        return await self.llm.score_relevance(
            headline=item.headline,
            ticker=item.ticker or "",
            company_name=company_name,
        )

    async def get_unscored_count(self) -> int:
        """Get count of news items pending scoring.

        Returns:
            Number of news items without relevance scores.
        """
        items = get_unscored_news(self.session, limit=1000)
        return len(items)

    async def score_single_item(self, item: NewsItem) -> float:
        """Score a single news item and persist to database.

        Convenience method for scoring individual items as they arrive,
        useful for real-time processing pipelines.

        Args:
            item: The NewsItem to score.

        Returns:
            The relevance score assigned (0-10, or -1 if unavailable).
        """
        score = await self._score_item(item)
        item.relevance_score = score

        try:
            self.session.commit()
        except Exception as e:
            logger.error(f"Failed to save score for item {item.id}: {e}")
            self.session.rollback()

        return score
