"""WatchlistSync - Syncs high-scoring candidates to portfolio.yaml watchlist.

This module provides the WatchlistSync class that:
1. Reads and writes to config/portfolio.yaml (or path from Settings)
2. Adds high-scoring candidates to the watchlist section
3. Preserves existing watchlist items
4. Stores optional metadata (score, date added, report summary)
"""

from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger
from sqlalchemy.orm import Session

from ..storage import ResearchReport, StockCandidate, get_top_candidates


class WatchlistSync:
    """Syncs high-scoring stock candidates to portfolio.yaml watchlist.

    This class manages the synchronization between the research database
    and the portfolio.yaml watchlist section. It ensures that:
    - High-scoring candidates are added to the watchlist
    - Existing watchlist items are preserved
    - Duplicate entries are not created
    - Optional metadata (score, date, summary) is stored

    Example:
        sync = WatchlistSync(session, portfolio_path=Path("config/portfolio.yaml"))
        sync.add_candidate_to_watchlist(candidate, report=report)
        count = sync.sync_from_candidates(min_score=75.0)
    """

    # Default minimum composite score for sync_from_candidates
    DEFAULT_MIN_SCORE = 70.0

    # Maximum number of candidates to fetch during sync
    MAX_SYNC_CANDIDATES = 1000

    # Maximum length for summary truncation in watchlist entries
    MAX_SUMMARY_LENGTH = 100

    def __init__(
        self,
        session: Session,
        portfolio_path: Path | None = None,
    ):
        """Initialize the WatchlistSync.

        Args:
            session: SQLAlchemy database session
            portfolio_path: Path to portfolio.yaml file. If None, uses
                           config/portfolio.yaml
        """
        self.session = session
        self.portfolio_path = portfolio_path or Path("config/portfolio.yaml")

    def add_candidate_to_watchlist(
        self,
        candidate: StockCandidate,
        report: ResearchReport | None = None,
    ) -> bool:
        """Add a stock candidate to the portfolio.yaml watchlist.

        Adds the candidate's ticker to the watchlist section if not already
        present. Optionally stores metadata from the candidate and report.

        Args:
            candidate: StockCandidate to add to watchlist
            report: Optional ResearchReport with additional context

        Returns:
            True if the ticker was added, False if it already exists
        """
        ticker = candidate.ticker.upper()

        # Load existing portfolio data
        portfolio_data = self._load_portfolio()

        # Ensure watchlist exists
        if "watchlist" not in portfolio_data:
            portfolio_data["watchlist"] = []

        # Check if ticker already exists
        existing_tickers = {
            item.get("ticker", "").upper()
            for item in portfolio_data["watchlist"]
        }
        if ticker in existing_tickers:
            logger.debug(f"{ticker} already in watchlist, skipping")
            return False

        # Build watchlist entry with metadata
        watchlist_entry = self._build_watchlist_entry(candidate, report)

        # Add to watchlist
        portfolio_data["watchlist"].append(watchlist_entry)

        # Save updated portfolio
        self._save_portfolio(portfolio_data)

        logger.info(f"Added {ticker} to watchlist (score: {candidate.composite_score})")
        return True

    def sync_from_candidates(self, min_score: float = DEFAULT_MIN_SCORE) -> int:
        """Sync all high-scoring candidates to the watchlist.

        Gets all candidates with composite_score >= min_score and adds
        them to the watchlist if not already present.

        Args:
            min_score: Minimum composite score threshold (default: 70.0)

        Returns:
            Count of newly added tickers
        """
        logger.info(f"Syncing candidates with score >= {min_score} to watchlist")

        # Get all candidates above the threshold
        candidates = get_top_candidates(
            self.session,
            limit=self.MAX_SYNC_CANDIDATES,
            min_score=min_score,
        )

        if not candidates:
            logger.info("No candidates found above threshold")
            return 0

        added_count = 0
        for candidate in candidates:
            if self.add_candidate_to_watchlist(candidate):
                added_count += 1

        logger.info(f"Synced {added_count} new candidates to watchlist")
        return added_count

    def _load_portfolio(self) -> dict:
        """Load portfolio data from YAML file.

        Creates a minimal structure if file doesn't exist.

        Returns:
            Dictionary with portfolio data
        """
        if not self.portfolio_path.exists():
            logger.debug(f"Portfolio file not found at {self.portfolio_path}, creating new")
            return {"watchlist": []}

        try:
            with open(self.portfolio_path) as f:
                data = yaml.safe_load(f)
                return data if data else {"watchlist": []}
        except Exception as e:
            logger.warning(f"Error loading portfolio file: {e}")
            return {"watchlist": []}

    def _save_portfolio(self, data: dict) -> None:
        """Save portfolio data to YAML file.

        Creates parent directories if needed.

        Args:
            data: Portfolio data dictionary to save

        Raises:
            OSError: If unable to write to the portfolio file
        """
        try:
            # Ensure parent directory exists
            self.portfolio_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.portfolio_path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            logger.debug(f"Saved portfolio to {self.portfolio_path}")
        except OSError as e:
            logger.error(f"Failed to save portfolio to {self.portfolio_path}: {e}")
            raise

    def _build_watchlist_entry(
        self,
        candidate: StockCandidate,
        report: ResearchReport | None = None,
    ) -> dict:
        """Build a watchlist entry with metadata.

        Args:
            candidate: StockCandidate to build entry for
            report: Optional ResearchReport for additional context

        Returns:
            Dictionary representing the watchlist entry
        """
        entry = {
            "ticker": candidate.ticker.upper(),
            "score": candidate.composite_score,
            "date_added": datetime.now().strftime("%Y-%m-%d"),
        }

        # Add report summary if available
        if report and report.summary:
            # Truncate summary for brevity
            summary = report.summary
            if len(summary) > self.MAX_SUMMARY_LENGTH:
                summary = summary[: self.MAX_SUMMARY_LENGTH - 3] + "..."
            entry["reason"] = summary

        # Add target price if available from report
        if report and report.target_price:
            entry["target_price"] = report.target_price

        return entry
