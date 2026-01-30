"""Research Orchestrator for coordinating full research flow.

This module provides the ResearchOrchestrator class that coordinates the
complete research pipeline:
1. Get/create StockCandidate
2. Fetch fundamentals (FundamentalsCollector)
3. Fetch news (NewsCollector)
4. Fetch congressional trades (CongressTradesCollector)
5. Generate deep report (ResearchReportGenerator using Claude)
6. Save report and update candidate status to "researched"
"""

from dataclasses import dataclass
import time

from loguru import logger
from sqlalchemy.orm import Session

from ..analysis import ResearchReportGenerator, ReportResult
from ..collectors import (
    CongressTradesCollector,
    FundamentalsCollector,
    FundamentalsData,
    NewsCollector,
)
from ..config import Settings
from ..models import ResearchConfig
from ..storage import (
    CANDIDATE_STATUSES,
    CandidateScore,
    CongressionalTrade,
    NewsItem,
    ResearchReport,
    StockCandidate,
    get_candidate_by_ticker,
    get_latest_score,
    get_trades_for_ticker,
    save_candidate,
    save_report,
)
from .queue import ResearchQueue


@dataclass
class ResearchResult:
    """Result of a research operation for a single ticker.

    Attributes:
        ticker: Stock ticker symbol that was researched
        success: Whether the research completed successfully
        report: The generated ResearchReport, or None if failed
        error: Error message if research failed, or None if successful
        duration: Time taken to complete the research in seconds
    """

    ticker: str
    success: bool
    report: ResearchReport | None
    error: str | None
    duration: float


class ResearchOrchestrator:
    """Coordinates the full research flow for stock candidates.

    This orchestrator manages the complete research pipeline:
    - Fetches fundamentals, news, and congressional trading data
    - Generates deep research reports using Claude API
    - Manages the research queue
    - Handles budget constraints for Claude API usage
    - Updates candidate status after research

    Example:
        orchestrator = ResearchOrchestrator(session, config, research_config)
        result = await orchestrator.research_ticker("AAPL")
        if result.success:
            print(result.report.summary)
    """

    # Status constants
    STATUS_DISCOVERED = CANDIDATE_STATUSES[0]  # "discovered"
    STATUS_SCREENING = CANDIDATE_STATUSES[1]  # "screening"
    STATUS_RESEARCHED = CANDIDATE_STATUSES[2]  # "researched"

    # Configuration constants
    MAX_TRADES_IN_SUMMARY = 10  # Maximum number of trades to include in congress summary
    DEFAULT_COMPOSITE_SCORE = 50.0  # Default score when no score exists for a candidate

    def __init__(
        self,
        session: Session,
        config: Settings,
        research_config: ResearchConfig,
    ):
        """Initialize the Research Orchestrator.

        Args:
            session: SQLAlchemy database session
            config: Application settings
            research_config: Research configuration with budget and weights
        """
        self.session = session
        self.config = config
        self.research_config = research_config

        # Initialize collectors
        self._fundamentals_collector = FundamentalsCollector(session, config)
        self._news_collector = NewsCollector(session, config)
        self._congress_collector = CongressTradesCollector(session, config)

        # Initialize report generator
        self._report_generator = ResearchReportGenerator(
            api_key=config.anthropic_api_key,
            max_monthly_spend=research_config.claude_budget.monthly_limit_usd,
        )

        # Initialize research queue
        self._queue = ResearchQueue(session)

    async def research_ticker(self, ticker: str) -> ResearchResult:
        """Perform full research for a single ticker.

        This executes the complete research flow:
        1. Get or create the StockCandidate
        2. Fetch fundamentals using FundamentalsCollector
        3. Fetch news using NewsCollector (if available)
        4. Fetch congressional trades using CongressTradesCollector
        5. Generate deep report using ResearchReportGenerator (Claude)
        6. Save report and update candidate status to "researched"

        Args:
            ticker: Stock ticker symbol to research

        Returns:
            ResearchResult with the research outcome
        """
        start_time = time.time()
        ticker = ticker.upper()

        logger.info(f"Starting research for {ticker}")

        try:
            # Step 1: Get or create candidate
            candidate = self._get_or_create_candidate(ticker)

            # Step 2: Check budget before proceeding
            if not self._check_budget():
                duration = time.time() - start_time
                error_msg = "Claude API budget exceeded - cannot generate report"
                logger.warning(f"Research for {ticker} skipped: {error_msg}")
                return ResearchResult(
                    ticker=ticker,
                    success=False,
                    report=None,
                    error=error_msg,
                    duration=duration,
                )

            # Step 3: Fetch fundamentals
            fundamentals = await self._fetch_fundamentals(ticker)

            # Step 4: Fetch news
            # Note: News is collected and stored in the database for historical tracking
            # and potential future sentiment analysis. The report generator currently
            # doesn't accept news items directly, but they are available via database queries.
            _ = await self._fetch_news(ticker)  # Stored in DB by NewsCollector

            # Step 5: Fetch congressional trades
            congress_trades = await self._fetch_congress_trades(ticker)

            # Step 6: Get score for report generation
            score = get_latest_score(self.session, ticker)
            if score is None:
                # Create a minimal score if none exists
                score = CandidateScore(
                    ticker=ticker,
                    composite_score=self.DEFAULT_COMPOSITE_SCORE,
                )

            # Step 7: Generate report
            report_result = await self._generate_report(
                ticker=ticker,
                fundamentals=fundamentals,
                score=score,
                congress_trades=congress_trades,
            )

            if not report_result.success:
                duration = time.time() - start_time
                error_msg = report_result.error_message or "Report generation failed"
                logger.error(f"Report generation failed for {ticker}: {error_msg}")
                return ResearchResult(
                    ticker=ticker,
                    success=False,
                    report=None,
                    error=error_msg,
                    duration=duration,
                )

            # Step 8: Save report and update candidate status
            report = report_result.report
            if report:
                save_report(self.session, report)
                candidate.status = self.STATUS_RESEARCHED
                self.session.flush()
                logger.info(f"Research completed for {ticker}: {report.recommendation}")

            duration = time.time() - start_time
            return ResearchResult(
                ticker=ticker,
                success=True,
                report=report,
                error=None,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.exception(f"Research failed for {ticker}: {error_msg}")
            return ResearchResult(
                ticker=ticker,
                success=False,
                report=None,
                error=error_msg,
                duration=duration,
            )

    async def process_queue(self, max_items: int = 5) -> list[ResearchResult]:
        """Process multiple items from the research queue.

        Gets items from the queue in priority order (highest composite_score first),
        performs research on each, and removes them from the queue after processing.

        Errors are handled gracefully - if one item fails, processing continues
        with the remaining items.

        Args:
            max_items: Maximum number of items to process (default: 5)

        Returns:
            List of ResearchResult objects for each processed item
        """
        results: list[ResearchResult] = []

        # Get items from queue
        queue_items = self._queue.get_queue(limit=max_items)

        if not queue_items:
            logger.info("Research queue is empty, nothing to process")
            return results

        logger.info(f"Processing {len(queue_items)} items from research queue")

        for candidate in queue_items:
            ticker = candidate.ticker

            try:
                # Research the ticker
                result = await self.research_ticker(ticker)
                results.append(result)

                if result.success:
                    logger.info(
                        f"Successfully researched {ticker} "
                        f"(duration: {result.duration:.1f}s)"
                    )
                else:
                    logger.warning(
                        f"Research failed for {ticker}: {result.error}"
                    )

            except Exception as e:
                # Create failure result and continue
                logger.exception(f"Unexpected error researching {ticker}: {e}")
                results.append(
                    ResearchResult(
                        ticker=ticker,
                        success=False,
                        report=None,
                        error=str(e),
                        duration=0.0,
                    )
                )
            finally:
                # Always remove from queue after processing, regardless of success/failure
                self._queue.remove_from_queue(ticker)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        logger.info(
            f"Queue processing complete: {successful} successful, {failed} failed"
        )

        return results

    def _get_or_create_candidate(self, ticker: str) -> StockCandidate:
        """Get existing candidate or create a new one.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockCandidate object
        """
        candidate = get_candidate_by_ticker(self.session, ticker)

        if candidate is None:
            candidate = StockCandidate(
                ticker=ticker,
                status=self.STATUS_DISCOVERED,
                discovery_source="research_orchestrator",
            )
            save_candidate(self.session, candidate)
            self.session.flush()
            logger.debug(f"Created new candidate for {ticker}")

        return candidate

    def _check_budget(self) -> bool:
        """Check if Claude API budget allows report generation.

        Returns:
            True if within budget, False if budget exceeded
        """
        if not self.research_config.claude_budget.enabled:
            return True

        remaining = self._report_generator.get_remaining_budget()
        return remaining > 0

    async def _fetch_fundamentals(self, ticker: str) -> FundamentalsData:
        """Fetch fundamentals for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalsData object
        """
        logger.debug(f"Fetching fundamentals for {ticker}")
        return await self._fundamentals_collector.get_fundamentals(ticker)

    async def _fetch_news(self, ticker: str) -> list[NewsItem]:
        """Fetch news for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of NewsItem objects
        """
        logger.debug(f"Fetching news for {ticker}")
        try:
            result = await self._news_collector.collect([ticker])
            return result.records_collected
        except Exception as e:
            logger.warning(f"Failed to fetch news for {ticker}: {e}")
            return []

    async def _fetch_congress_trades(self, ticker: str) -> list[CongressionalTrade]:
        """Fetch congressional trades for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of CongressionalTrade objects
        """
        logger.debug(f"Fetching congressional trades for {ticker}")
        try:
            # Get recent trades from database
            trades = get_trades_for_ticker(self.session, ticker)
            return trades
        except Exception as e:
            logger.warning(f"Failed to fetch congressional trades for {ticker}: {e}")
            return []

    def _build_congress_summary(self, trades: list[CongressionalTrade]) -> str:
        """Build a summary of congressional trades for the report prompt.

        Args:
            trades: List of CongressionalTrade objects

        Returns:
            Formatted string summary of trades
        """
        if not trades:
            return "No congressional trading activity found"

        lines = []
        for trade in trades[:self.MAX_TRADES_IN_SUMMARY]:
            lines.append(
                f"- {trade.politician} ({trade.party or 'Unknown'}): "
                f"{trade.trade_type} {trade.amount_range} on {trade.trade_date}"
            )

        return "\n".join(lines)

    async def _generate_report(
        self,
        ticker: str,
        fundamentals: FundamentalsData,
        score: CandidateScore,
        congress_trades: list[CongressionalTrade],
    ) -> ReportResult:
        """Generate a research report using Claude API.

        Args:
            ticker: Stock ticker symbol
            fundamentals: FundamentalsData for the stock
            score: CandidateScore with factor scores
            congress_trades: List of congressional trades

        Returns:
            ReportResult from the report generator
        """
        logger.debug(f"Generating report for {ticker}")

        # Build congress summary
        congress_summary = self._build_congress_summary(congress_trades)

        # Get company name from fundamentals or use ticker
        company_name = ticker  # Could be enhanced to get from fundamentals

        return await self._report_generator.generate_report(
            ticker=ticker,
            company_name=company_name,
            fundamentals=fundamentals,
            score_result=score,
            congress_summary=congress_summary,
        )
