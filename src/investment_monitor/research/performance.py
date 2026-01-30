"""Performance analysis for tracking and optimizing investment decisions.

This module provides tools for:
1. Tracking 30/60/90 day performance of researched candidates
2. Analyzing correlations between factor scores and actual returns
3. Suggesting weight adjustments based on historical performance
"""

from datetime import date, timedelta

import yfinance as yf
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import ScoringWeights
from ..storage.research_models import CandidateScore, PerformanceTracker
from ..storage.research_operations import get_records_needing_update


class PerformanceAnalyzer:
    """Analyzes performance of stock candidates to improve scoring.

    Tracks how well scored candidates perform over time and uses
    this data to suggest improvements to scoring weights.
    """

    # Minimum number of candidates required to suggest weight adjustments
    MIN_CANDIDATES_FOR_SUGGESTIONS = 20

    def __init__(self, session: Session):
        """Initialize the performance analyzer.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    async def update_performance_data(self) -> int:
        """Update performance records with current prices and calculated returns.

        Gets all PerformanceTracker records needing update, fetches current
        prices for each ticker, and calculates returns vs entry_price for
        30/60/90 day periods.

        Returns:
            Count of updated records
        """
        records = get_records_needing_update(self.session)

        if not records:
            logger.debug("No performance records need updating")
            return 0

        updated_count = 0

        for record in records:
            try:
                current_price = await self._fetch_current_price(record.ticker)

                if current_price is None:
                    logger.warning(
                        f"Could not fetch price for {record.ticker}, skipping"
                    )
                    continue

                # Update the record with current price and calculated returns
                record.current_price = current_price

                # Calculate returns based on how long since entry
                return_pct = self._calculate_return(record.entry_price, current_price)

                days_since_entry = (date.today() - record.entry_date).days

                # Assign returns to appropriate periods
                if days_since_entry >= 30:
                    record.return_30d = return_pct
                if days_since_entry >= 60:
                    record.return_60d = return_pct
                if days_since_entry >= 90:
                    record.return_90d = return_pct

                updated_count += 1
                logger.debug(
                    f"Updated {record.ticker}: price={current_price:.2f}, "
                    f"return={return_pct:.2f}%"
                )

            except Exception as e:
                logger.error(f"Error updating performance for {record.ticker}: {e}")

        if updated_count > 0:
            self.session.commit()
            logger.info(f"Updated {updated_count} performance records")

        return updated_count

    async def _fetch_current_price(self, ticker: str) -> float | None:
        """Fetch current price for a ticker using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price or None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            # Get the most recent close price
            hist = stock.history(period="1d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch price for {ticker}: {e}")
            return None

    def _calculate_return(self, entry_price: float, current_price: float) -> float:
        """Calculate percentage return.

        Args:
            entry_price: Original entry price
            current_price: Current price

        Returns:
            Percentage return (e.g., 10.0 for 10% gain)
        """
        if entry_price <= 0:
            return 0.0
        return ((current_price - entry_price) / entry_price) * 100

    def analyze_factor_performance(self) -> dict:
        """Analyze correlation between factor scores and actual returns.

        Gets candidates that have both scores (CandidateScore) and performance
        data (PerformanceTracker with return_30d), then calculates correlation
        between each factor score and actual returns.

        Returns:
            Dictionary mapping factor names to correlation coefficients.
            Example: {"value": 0.35, "growth": 0.22, "quality": 0.15, ...}
            Returns empty dict if not enough data.
        """
        # Get candidates with both scores and performance data
        data = self._get_candidates_with_scores_and_performance()

        if not data:
            logger.debug("No candidates with both scores and performance data")
            return {}

        # Extract factor scores and returns
        factors = ["value", "growth", "quality", "momentum", "sentiment"]
        correlations = {}

        for factor in factors:
            factor_scores = [d[f"{factor}_score"] for d in data]
            returns = [d["return_30d"] for d in data]

            correlation = self._calculate_correlation(factor_scores, returns)
            correlations[factor] = correlation

        logger.info(f"Factor correlations: {correlations}")
        return correlations

    def _get_candidates_with_scores_and_performance(self) -> list[dict]:
        """Get candidates that have both scores and performance data.

        Returns:
            List of dicts with score and performance data for each candidate
        """
        # Query to join CandidateScore with PerformanceTracker
        stmt = (
            select(
                CandidateScore.ticker,
                CandidateScore.value_score,
                CandidateScore.growth_score,
                CandidateScore.quality_score,
                CandidateScore.momentum_score,
                CandidateScore.sentiment_score,
                PerformanceTracker.return_30d,
            )
            .join(
                PerformanceTracker,
                CandidateScore.ticker == PerformanceTracker.ticker,
            )
            .where(PerformanceTracker.return_30d.isnot(None))
        )

        results = self.session.execute(stmt).fetchall()

        return [
            {
                "ticker": row.ticker,
                "value_score": row.value_score,
                "growth_score": row.growth_score,
                "quality_score": row.quality_score,
                "momentum_score": row.momentum_score,
                "sentiment_score": row.sentiment_score,
                "return_30d": row.return_30d,
            }
            for row in results
        ]

    def _calculate_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient between two lists.

        Args:
            x: First list of values
            y: Second list of values

        Returns:
            Correlation coefficient between -1 and 1
        """
        n = len(x)
        if n < 2:
            return 0.0

        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate standard deviations and covariance
        variance_x = sum((xi - mean_x) ** 2 for xi in x) / n
        variance_y = sum((yi - mean_y) ** 2 for yi in y) / n
        covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n

        # Avoid division by zero
        std_x = variance_x ** 0.5
        std_y = variance_y ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return covariance / (std_x * std_y)

    def suggest_weight_adjustments(self) -> ScoringWeights | None:
        """Suggest new scoring weights based on factor correlations.

        Based on factor correlations with actual returns, suggests new weights
        that emphasize factors with higher correlation to returns.

        Returns:
            ScoringWeights with suggested weights, or None if not enough data
            (less than MIN_CANDIDATES_FOR_SUGGESTIONS candidates with performance)
        """
        # Check if we have enough data
        data = self._get_candidates_with_scores_and_performance()

        if len(data) < self.MIN_CANDIDATES_FOR_SUGGESTIONS:
            logger.info(
                f"Not enough data for weight suggestions: {len(data)} candidates, "
                f"need {self.MIN_CANDIDATES_FOR_SUGGESTIONS}"
            )
            return None

        # Get correlations
        correlations = self.analyze_factor_performance()

        if not correlations:
            return None

        # Convert correlations to weights
        # Use absolute values and normalize to sum to 1.0
        # Add a small base to avoid zero weights
        factors = ["value", "growth", "quality", "momentum", "sentiment"]

        # Shift correlations to be non-negative (correlation can be -1 to 1)
        # We add 1 to make range 0-2, then normalize
        shifted = {f: max(correlations.get(f, 0) + 1, 0.1) for f in factors}

        # Normalize to sum to 1.0
        total = sum(shifted.values())
        weights = {f: shifted[f] / total for f in factors}

        logger.info(f"Suggested weight adjustments: {weights}")

        return ScoringWeights(
            value=weights["value"],
            growth=weights["growth"],
            quality=weights["quality"],
            momentum=weights["momentum"],
            sentiment=weights["sentiment"],
        )
