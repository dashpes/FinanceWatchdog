"""Discovery pipeline for stock candidate screening and scoring.

This module orchestrates the full discovery process:
1. Collect universe (S&P 500, NASDAQ 100, ETFs)
2. Apply filters (market cap, sectors)
3. Fetch fundamentals
4. Score candidates (Ollama)
5. Identify top candidates
6. Auto-add to watchlist if above threshold
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from ..analysis import ResearchScorer
from ..collectors import (
    FundamentalsCollector,
    FundamentalsData,
    PriceCollector,
    UniverseCollector,
)
from ..config import Settings
from ..models import ResearchConfig, ScoringWeights
from ..storage import (
    CANDIDATE_STATUSES,
    CandidateScore,
    StockCandidate,
    get_candidate_by_ticker,
    get_candidates_by_status,
    get_insider_transactions,
    get_latest_score,
    get_prices,
    get_recent_news,
    get_top_candidates,
    save_candidate,
    save_score,
)

# Enough daily history for 1y momentum + 52-week levels (calendar days).
_PRICE_HISTORY_DAYS = 400


def _close_on_or_before(prices: list, target: date) -> float | None:
    """Latest close at or before ``target`` (``prices`` newest-first)."""
    for p in prices:
        if p.date <= target and p.close:
            return float(p.close)
    return None


def _rsi(closes_old_to_new: list[float], period: int = 14) -> float | None:
    """Simple (non-smoothed) 14-day RSI over the most recent ``period`` deltas."""
    if len(closes_old_to_new) < period + 1:
        return None
    recent = closes_old_to_new[-(period + 1):]
    gains = [max(0.0, b - a) for a, b in zip(recent, recent[1:])]
    losses = [max(0.0, a - b) for a, b in zip(recent, recent[1:])]
    avg_gain, avg_loss = sum(gains) / period, sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_momentum_inputs(prices: list, *, today: date | None = None) -> dict:
    """Derive the scorer's momentum inputs from stored daily prices (newest-first).

    Every field degrades to None when history is insufficient — the scorer already
    treats None as 'not available', so partial history never fabricates momentum.
    52-week levels require >=300 days of span so a young series can't fake them.
    """
    today = today or date.today()
    out: dict = {
        "price_change_1m": None, "price_change_3m": None, "price_change_6m": None,
        "price_change_1y": None, "rsi": None, "vs_52w_high": None, "vs_52w_low": None,
    }
    priced = [p for p in prices if p.close]
    if not priced:
        return out
    latest = float(priced[0].close)
    oldest = min(p.date for p in priced)
    for key, span in (("price_change_1m", 30), ("price_change_3m", 91),
                      ("price_change_6m", 182), ("price_change_1y", 365)):
        target = today - timedelta(days=span)
        if oldest > target:
            continue  # series doesn't reach back that far
        base = _close_on_or_before(priced, target)
        if base and base > 0:
            out[key] = (latest / base - 1.0) * 100.0
    out["rsi"] = _rsi([float(p.close) for p in reversed(priced)])
    if (today - oldest).days >= 300:
        year = [float(p.close) for p in priced if p.date >= today - timedelta(days=365)]
        if year:
            out["vs_52w_high"] = (latest / max(year) - 1.0) * 100.0  # <=0: below the high
            out["vs_52w_low"] = (latest / min(year) - 1.0) * 100.0   # >=0: above the low
    return out


def summarize_insider_activity(txns: list) -> str:
    """Compact 90d open-market buy/sell summary for the sentiment prompt."""
    buys = [t for t in txns if t.raw_code == "P"]
    sells = [t for t in txns if t.raw_code == "S"]
    if not buys and not sells:
        return "No insider activity data"
    buy_val = sum(float(t.total_value or 0.0) for t in buys)
    sell_val = sum(float(t.total_value or 0.0) for t in sells)
    if buy_val > sell_val:
        lean = "net buying"
    elif sell_val > buy_val:
        lean = "net selling"
    else:
        lean = "mixed"
    return (
        f"Last 90d: {len(buys)} open-market buys (${buy_val:,.0f}) vs "
        f"{len(sells)} sells (${sell_val:,.0f}) — {lean}"
    )


def summarize_recent_news(items: list, *, max_headlines: int = 3) -> str:
    """Headline count + most-relevant titles for the sentiment prompt."""
    if not items:
        return "No recent news available"
    ranked = sorted(items, key=lambda i: (i.relevance_score or 0.0), reverse=True)
    heads = "; ".join((i.headline or "")[:120] for i in ranked[:max_headlines])
    return f"{len(items)} headlines in the last 14d. Most relevant: {heads}"


@dataclass
class DiscoveryResult:
    """Result of a discovery pipeline run."""

    total_candidates: int = 0
    scored_candidates: int = 0
    top_candidates: list[str] = field(default_factory=list)
    watchlist_additions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate duration of the pipeline run in seconds."""
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        """Check if pipeline completed without critical errors."""
        # Consider success if we scored at least one candidate
        return self.scored_candidates > 0 or len(self.errors) == 0


class DiscoveryPipeline:
    """Orchestrates stock discovery and initial scoring.

    The pipeline runs through 6 steps:
    1. Collect universe - gather tickers from indices and ETFs
    2. Apply filters - remove excluded sectors, market cap filter
    3. Fetch fundamentals - get financial data for candidates
    4. Score candidates - use LLM to score on 5 factors
    5. Identify top candidates - rank by composite score
    6. Auto-add to watchlist - promote high-scoring candidates
    """

    # Maximum number of candidates to retrieve from database
    MAX_DISCOVERED_CANDIDATES = 10000

    # Status constants from CANDIDATE_STATUSES
    STATUS_DISCOVERED = CANDIDATE_STATUSES[0]  # "discovered"
    STATUS_SCREENING = CANDIDATE_STATUSES[1]  # "screening"
    STATUS_WATCHLIST = CANDIDATE_STATUSES[3]  # "watchlist"

    def __init__(
        self,
        session: Session,
        config: Settings,
        research_config: ResearchConfig,
        ollama_model: str = "phi3:mini",
    ):
        """Initialize the discovery pipeline.

        Args:
            session: SQLAlchemy database session
            config: Application settings
            research_config: Research configuration with weights and thresholds
            ollama_model: Ollama model for scoring (default: phi3:mini)
        """
        self.session = session
        self.config = config
        self.research_config = research_config

        # Initialize collectors
        self.universe_collector = UniverseCollector(
            session=session,
            config=config,
            collect_sp500=research_config.universe.include_sp500,
            collect_nasdaq100=research_config.universe.include_nasdaq100,
        )
        self.fundamentals_collector = FundamentalsCollector(
            session=session,
            config=config,
        )
        # Deep history for the scoring batch only, so momentum/RSI/52w inputs are real.
        self.price_collector = PriceCollector(
            session=session, config=config, days_to_fetch=_PRICE_HISTORY_DAYS
        )

        # Initialize scorer
        self.scorer = ResearchScorer(model=ollama_model)

    async def run_discovery(self, dry_run: bool = False) -> DiscoveryResult:
        """Run the full discovery pipeline.

        Args:
            dry_run: If True, don't persist changes to database

        Returns:
            DiscoveryResult with collection statistics and identified candidates
        """
        result = DiscoveryResult()

        try:
            # Step 1: Collect universe
            logger.info("Step 1: Collecting stock universe...")
            etf_tickers = self.research_config.universe.etf_tickers
            universe_result = await self.universe_collector.collect(etf_tickers)
            logger.info(
                f"Universe collection: {universe_result.records_collected} new records"
            )

            # Get all discovered tickers from database
            candidates = self._get_discovered_tickers()
            result.total_candidates = len(candidates)
            logger.info(f"Total candidates in universe: {result.total_candidates}")

            # Step 2: Apply filters
            logger.info("Step 2: Applying filters...")
            filtered_candidates = self._apply_filters(candidates)
            logger.info(
                f"After filtering: {len(filtered_candidates)} candidates "
                f"(filtered out {len(candidates) - len(filtered_candidates)})"
            )

            # Step 3: Fetch fundamentals
            logger.info("Step 3: Fetching fundamentals...")
            batch_size = self.research_config.discovery_batch_size
            candidates_to_process = filtered_candidates[:batch_size]
            fundamentals_result = await self.fundamentals_collector.collect(
                candidates_to_process
            )
            logger.info(
                f"Fundamentals collected for {fundamentals_result.records_collected} candidates"
            )

            # Step 3.5: Price history for the scoring batch (momentum/RSI/52w inputs).
            # Best-effort: scoring proceeds on whatever history lands.
            try:
                prices_result = await self.price_collector.collect(candidates_to_process)
                logger.info(
                    f"Price history: {prices_result.records_collected} rows for scoring batch"
                )
            except Exception as e:
                logger.warning(f"Price history collection failed (scoring degrades): {e}")

            # Step 4: Score candidates
            logger.info("Step 4: Scoring candidates...")
            scored_count = 0
            for ticker in candidates_to_process:
                try:
                    fundamentals = self.fundamentals_collector.get_cached_fundamentals(
                        ticker
                    )
                    if fundamentals:
                        score = await self._score_candidate(ticker, fundamentals)
                        if not dry_run:
                            self._save_score(ticker, score)
                        scored_count += 1
                        logger.debug(
                            f"Scored {ticker}: {score.composite_score:.1f}"
                        )
                except Exception as e:
                    error_msg = f"{ticker}: {str(e)}"
                    logger.warning(f"Failed to score {ticker}: {e}")
                    result.errors.append(error_msg)

            result.scored_candidates = scored_count
            logger.info(f"Successfully scored {scored_count} candidates")

            # Step 5: Identify top candidates
            logger.info("Step 5: Identifying top candidates...")
            min_score = self.research_config.thresholds.min_composite_score
            result.top_candidates = self._get_top_candidates_list(min_score=min_score)
            logger.info(
                f"Found {len(result.top_candidates)} candidates above "
                f"minimum score threshold ({min_score})"
            )

            # Step 6: Auto-add to watchlist
            if not dry_run:
                logger.info("Step 6: Auto-adding to watchlist...")
                watchlist_threshold = (
                    self.research_config.thresholds.auto_watchlist_score
                )
                for ticker in result.top_candidates:
                    score = self._get_candidate_score(ticker)
                    if score and score >= watchlist_threshold:
                        self._update_candidate_status(ticker, self.STATUS_WATCHLIST)
                        result.watchlist_additions.append(ticker)
                        logger.info(
                            f"Added {ticker} to watchlist (score: {score:.1f})"
                        )

                # Commit all changes
                self.session.commit()
                logger.info(
                    f"Added {len(result.watchlist_additions)} candidates to watchlist"
                )
            else:
                logger.info("Step 6: Skipped (dry run)")

        except Exception as e:
            logger.exception("Discovery pipeline failed")
            result.errors.append(f"Pipeline error: {str(e)}")

        result.finished_at = datetime.now()
        logger.info(
            f"Discovery pipeline completed in {result.duration_seconds:.1f}s - "
            f"scored {result.scored_candidates}, "
            f"top candidates: {len(result.top_candidates)}, "
            f"watchlist additions: {len(result.watchlist_additions)}, "
            f"errors: {len(result.errors)}"
        )
        return result

    def _get_discovered_tickers(self) -> list[str]:
        """Get all discovered stock candidates.

        Returns:
            List of ticker symbols for discovered candidates
        """
        # Get candidates with "discovered" status
        candidates = get_candidates_by_status(
            self.session, self.STATUS_DISCOVERED, limit=self.MAX_DISCOVERED_CANDIDATES
        )
        return [c.ticker for c in candidates]

    def _apply_filters(self, tickers: list[str]) -> list[str]:
        """Apply universe filters (market cap, excluded sectors).

        Args:
            tickers: List of ticker symbols to filter

        Returns:
            Filtered list of ticker symbols
        """
        universe_config = self.research_config.universe
        excluded_tickers = set(
            t.upper() for t in universe_config.excluded_tickers
        )
        excluded_sectors = set(
            s.lower() for s in universe_config.excluded_sectors
        )

        filtered = []
        for ticker in tickers:
            # Skip explicitly excluded tickers
            if ticker.upper() in excluded_tickers:
                logger.debug(f"Filtered out {ticker}: explicitly excluded")
                continue

            # Check cached fundamentals for sector/market cap filters
            fundamentals = self.fundamentals_collector.get_cached_fundamentals(ticker)
            if fundamentals:
                # Check market cap
                if fundamentals.market_cap is not None:
                    if fundamentals.market_cap < universe_config.min_market_cap:
                        logger.debug(
                            f"Filtered out {ticker}: market cap "
                            f"${fundamentals.market_cap:,.0f} below minimum"
                        )
                        continue

                # Check sector
                if fundamentals.sector:
                    if fundamentals.sector.lower() in excluded_sectors:
                        logger.debug(
                            f"Filtered out {ticker}: sector "
                            f"'{fundamentals.sector}' excluded"
                        )
                        continue

            filtered.append(ticker)

        return filtered

    async def _score_candidate(
        self, ticker: str, fundamentals: FundamentalsData
    ) -> CandidateScore:
        """Score a single candidate using the research scorer.

        Args:
            ticker: Stock ticker symbol
            fundamentals: FundamentalsData for the stock

        Returns:
            CandidateScore with all factor scores and composite
        """
        weights = self.research_config.scoring_weights

        # Derive real momentum/sentiment inputs from data already in the DB. Each
        # degrades independently to the scorer's "not available" default.
        momentum = compute_momentum_inputs(
            get_prices(self.session, ticker, days=_PRICE_HISTORY_DAYS)
        )
        insider_summary = summarize_insider_activity(
            get_insider_transactions(self.session, ticker, days=90)
        )
        news_summary = summarize_recent_news(
            get_recent_news(self.session, ticker=ticker, hours=14 * 24)
        )

        # Use the scorer's convenience method that handles all 5 factors
        score = await self.scorer.score_stock(
            fundamentals=fundamentals,
            weights=weights,
            **momentum,
            recent_news_summary=news_summary,
            insider_activity=insider_summary,
            # Still unsourced (no analyst/short-interest collectors yet).
            analyst_rating="No analyst data",
            short_interest=None,
        )

        return score

    def _save_score(self, ticker: str, score: CandidateScore) -> None:
        """Save candidate score to database.

        Args:
            ticker: Stock ticker symbol
            score: CandidateScore to save
        """
        save_score(self.session, score)

        # Also update the candidate's composite score
        candidate = get_candidate_by_ticker(self.session, ticker)
        if candidate:
            candidate.composite_score = score.composite_score
            candidate.status = self.STATUS_SCREENING
            # Flush to make changes visible in subsequent queries
            self.session.flush()

    def _get_top_candidates_list(
        self, min_score: float = 60.0, limit: int = 20
    ) -> list[str]:
        """Get top scoring candidates.

        Args:
            min_score: Minimum composite score threshold
            limit: Maximum number of candidates to return

        Returns:
            List of ticker symbols for top candidates
        """
        candidates = get_top_candidates(
            self.session, limit=limit, min_score=min_score
        )
        return [c.ticker for c in candidates]

    def _get_candidate_score(self, ticker: str) -> float | None:
        """Get composite score for a candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Composite score or None if not scored
        """
        score = get_latest_score(self.session, ticker)
        if score:
            return score.composite_score
        return None

    def _update_candidate_status(self, ticker: str, status: str) -> None:
        """Update candidate status.

        Args:
            ticker: Stock ticker symbol
            status: New status (e.g., "watchlist", "rejected")
        """
        candidate = get_candidate_by_ticker(self.session, ticker)
        if candidate:
            candidate.status = status
            logger.debug(f"Updated {ticker} status to '{status}'")
