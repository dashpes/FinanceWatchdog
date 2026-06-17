"""Research Scorer using Ollama for multi-factor stock scoring.

This module provides AI-powered stock scoring across 5 factors:
- Value: Evaluates valuation metrics (P/E, P/B, P/S, PEG, etc.)
- Growth: Evaluates revenue and EPS growth trajectory
- Quality: Evaluates business quality (ROE, margins, debt levels)
- Momentum: Evaluates price momentum and technical indicators
- Sentiment: Evaluates market sentiment from news, insiders, analysts

Example usage:
    from investment_monitor.analysis import ResearchScorer
    from investment_monitor.collectors import FundamentalsData
    from investment_monitor.models import ScoringWeights

    scorer = ResearchScorer()

    if scorer.is_available():
        value_result = await scorer.score_value(fundamentals)
        print(f"Value Score: {value_result.score} - {value_result.reasoning}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .research_prompts import (
    GROWTH_SCORE_PROMPT,
    MOMENTUM_SCORE_PROMPT,
    QUALITY_SCORE_PROMPT,
    SENTIMENT_SCORE_PROMPT,
    VALUE_SCORE_PROMPT,
)

if TYPE_CHECKING:
    import ollama
    from ..collectors.fundamentals import FundamentalsData
    from ..models.research import ScoringWeights
    from ..storage.research_models import CandidateScore


@dataclass
class ScoreResult:
    """Result from a scoring operation.

    Attributes:
        score: The score from 0-100 (or 50.0 as default if unavailable)
        reasoning: Explanation for the score
    """
    score: float  # 0-100
    reasoning: str


# Default score when LLM is unavailable
DEFAULT_SCORE = 50.0
DEFAULT_REASONING = "LLM unavailable - using neutral default score"


class ResearchScorer:
    """AI-powered stock scorer using Ollama for multi-factor analysis.

    Scores stocks on 5 factors (value, growth, quality, momentum, sentiment)
    using local LLM inference via Ollama. Falls back to neutral default scores
    when Ollama is unavailable.

    Follows the LocalLLM pattern:
    - _generate() is synchronous
    - Public scoring methods are async
    - Handles unavailable Ollama gracefully
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize the ResearchScorer.

        Args:
            model: The Ollama model to use (default: phi3:mini)
            base_url: The Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self._client: ollama.Client | None = None
        self._available: bool | None = None

    @property
    def client(self) -> ollama.Client:
        """Lazily initialize and return the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                logger.warning("ollama package not installed, research scoring features unavailable")
                raise
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available.

        Returns:
            True if Ollama is running and model is loaded, False otherwise.
        """
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            # List models to check if server is running. Handle both the new ollama
            # client (ListResponse.models with a .model attr) and the old dict shape.
            response = client.list()
            model_names: list[str] = []
            if hasattr(response, "models"):
                for m in response.models:
                    model_names.append(getattr(m, "model", "") or "")
            else:
                for m in response.get("models", []):
                    model_names.append(m.get("model") or m.get("name") or "")

            # Check if our model is available (handle both full and short names)
            base_model = self.model.split(":")[0]
            for name in model_names:
                if name == self.model or name.startswith(base_model):
                    self._available = True
                    return True

            # Model not found but server is running
            logger.warning(f"Model {self.model} not found in Ollama. Available: {model_names}")
            self._available = False
            return False

        except ImportError:
            logger.warning("ollama package not installed")
            self._available = False
            return False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False

    def _generate(self, prompt: str) -> str | None:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM response text, or None if unavailable.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent scoring
                    "num_predict": 200,  # Allow enough tokens for JSON + reasoning
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.debug(f"LLM generation failed: {e}")
            return None

    def _parse_json_response(self, response: str | None) -> ScoreResult:
        """Parse a JSON response from the LLM.

        Handles various formats the LLM might produce:
        - Clean JSON: {"score": 75, "reasoning": "..."}
        - JSON with extra text before/after
        - Malformed JSON with recovery attempts

        Args:
            response: The raw LLM response

        Returns:
            ScoreResult with parsed score and reasoning, or defaults if parsing fails
        """
        if not response:
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        # Try to extract JSON from the response
        json_str = self._extract_json(response)
        if not json_str:
            return ScoreResult(score=DEFAULT_SCORE, reasoning=f"Failed to extract JSON from: {response[:100]}")

        try:
            data = json.loads(json_str)

            # Extract score
            score = data.get("score")
            if score is None:
                return ScoreResult(score=DEFAULT_SCORE, reasoning="No score field in response")

            # Validate and clamp score
            try:
                score = float(score)
                score = max(0.0, min(100.0, score))
            except (TypeError, ValueError):
                return ScoreResult(score=DEFAULT_SCORE, reasoning=f"Invalid score value: {score}")

            # Extract reasoning
            reasoning = data.get("reasoning", "No reasoning provided")
            if not isinstance(reasoning, str):
                reasoning = str(reasoning)

            return ScoreResult(score=score, reasoning=reasoning)

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e} for: {json_str[:100]}")
            return ScoreResult(score=DEFAULT_SCORE, reasoning=f"JSON parse error: {str(e)}")

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON object from text that may contain extra content.

        Args:
            text: Text that may contain a JSON object

        Returns:
            Extracted JSON string, or None if not found
        """
        if not text:
            return None

        # Try to find JSON object using regex
        # Look for {...} pattern, being careful with nested braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)

        # Try each match to find valid JSON with score field
        for match in matches:
            try:
                data = json.loads(match)
                if "score" in data:
                    return match
            except json.JSONDecodeError:
                continue

        # If no valid JSON found, try the whole text
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        return None

    def _format_value(self, value: float | None, as_percent: bool = False) -> str:
        """Format a numeric value for display in prompts.

        Args:
            value: The value to format (may be None)
            as_percent: If True, multiply by 100 and add % suffix

        Returns:
            Formatted string representation
        """
        if value is None:
            return "N/A"
        if as_percent:
            return f"{value * 100:.2f}"
        return f"{value:.2f}"

    async def score_value(self, fundamentals: FundamentalsData) -> ScoreResult:
        """Score a stock on value metrics.

        Args:
            fundamentals: FundamentalsData containing valuation metrics

        Returns:
            ScoreResult with value score (0-100) and reasoning
        """
        if not self.is_available():
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        prompt = VALUE_SCORE_PROMPT.format(
            ticker=fundamentals.ticker,
            company_name=fundamentals.ticker,  # Use ticker as fallback
            pe_ratio=self._format_value(fundamentals.pe_ratio),
            pb_ratio=self._format_value(fundamentals.pb_ratio),
            ps_ratio=self._format_value(fundamentals.ps_ratio),
            peg_ratio=self._format_value(fundamentals.peg_ratio),
            dividend_yield=self._format_value(fundamentals.dividend_yield, as_percent=True),
            free_cash_flow=self._format_value(fundamentals.free_cash_flow),
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
        )

        response = self._generate(prompt)
        return self._parse_json_response(response)

    async def score_growth(self, fundamentals: FundamentalsData) -> ScoreResult:
        """Score a stock on growth metrics.

        Args:
            fundamentals: FundamentalsData containing growth metrics

        Returns:
            ScoreResult with growth score (0-100) and reasoning
        """
        if not self.is_available():
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        prompt = GROWTH_SCORE_PROMPT.format(
            ticker=fundamentals.ticker,
            company_name=fundamentals.ticker,
            revenue_growth_yoy=self._format_value(fundamentals.revenue_growth_yoy, as_percent=True),
            revenue_growth_3y=self._format_value(fundamentals.revenue_growth_3y, as_percent=True),
            eps_growth_yoy=self._format_value(fundamentals.eps_growth_yoy, as_percent=True),
            eps_growth_3y=self._format_value(fundamentals.eps_growth_3y, as_percent=True),
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
        )

        response = self._generate(prompt)
        return self._parse_json_response(response)

    async def score_quality(self, fundamentals: FundamentalsData) -> ScoreResult:
        """Score a stock on quality metrics.

        Args:
            fundamentals: FundamentalsData containing quality metrics

        Returns:
            ScoreResult with quality score (0-100) and reasoning
        """
        if not self.is_available():
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        prompt = QUALITY_SCORE_PROMPT.format(
            ticker=fundamentals.ticker,
            company_name=fundamentals.ticker,
            roe=self._format_value(fundamentals.roe, as_percent=True),
            profit_margin=self._format_value(fundamentals.profit_margin, as_percent=True),
            debt_to_equity=self._format_value(fundamentals.debt_to_equity),
            current_ratio=self._format_value(fundamentals.current_ratio),
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
        )

        response = self._generate(prompt)
        return self._parse_json_response(response)

    async def score_momentum(
        self,
        ticker: str,
        company_name: str = "",
        price_change_1m: float | None = None,
        price_change_3m: float | None = None,
        price_change_6m: float | None = None,
        price_change_1y: float | None = None,
        rsi: float | None = None,
        vs_52w_high: float | None = None,
        vs_52w_low: float | None = None,
        sector: str = "Unknown",
        industry: str = "Unknown",
    ) -> ScoreResult:
        """Score a stock on momentum metrics.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name (defaults to ticker)
            price_change_1m: 1-month price change percentage
            price_change_3m: 3-month price change percentage
            price_change_6m: 6-month price change percentage
            price_change_1y: 1-year price change percentage
            rsi: Relative Strength Index (14-day)
            vs_52w_high: Distance from 52-week high (%)
            vs_52w_low: Distance from 52-week low (%)
            sector: Stock sector
            industry: Stock industry

        Returns:
            ScoreResult with momentum score (0-100) and reasoning
        """
        if not self.is_available():
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        prompt = MOMENTUM_SCORE_PROMPT.format(
            ticker=ticker,
            company_name=company_name or ticker,
            price_change_1m=self._format_value(price_change_1m),
            price_change_3m=self._format_value(price_change_3m),
            price_change_6m=self._format_value(price_change_6m),
            price_change_1y=self._format_value(price_change_1y),
            rsi=self._format_value(rsi),
            vs_52w_high=self._format_value(vs_52w_high),
            vs_52w_low=self._format_value(vs_52w_low),
            sector=sector,
            industry=industry,
        )

        response = self._generate(prompt)
        return self._parse_json_response(response)

    async def score_sentiment(
        self,
        ticker: str,
        company_name: str = "",
        recent_news_summary: str = "No recent news available",
        insider_activity: str = "No insider activity data",
        analyst_rating: str = "No analyst data",
        short_interest: float | None = None,
        sector: str = "Unknown",
        industry: str = "Unknown",
    ) -> ScoreResult:
        """Score a stock on sentiment indicators.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name (defaults to ticker)
            recent_news_summary: Summary of recent news
            insider_activity: Description of insider buying/selling
            analyst_rating: Analyst consensus and ratings
            short_interest: Short interest as percentage
            sector: Stock sector
            industry: Stock industry

        Returns:
            ScoreResult with sentiment score (0-100) and reasoning
        """
        if not self.is_available():
            return ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)

        prompt = SENTIMENT_SCORE_PROMPT.format(
            ticker=ticker,
            company_name=company_name or ticker,
            recent_news_summary=recent_news_summary,
            insider_activity=insider_activity,
            analyst_rating=analyst_rating,
            short_interest=self._format_value(short_interest),
            sector=sector,
            industry=industry,
        )

        response = self._generate(prompt)
        return self._parse_json_response(response)

    async def calculate_composite_score(
        self,
        value_result: ScoreResult,
        growth_result: ScoreResult,
        quality_result: ScoreResult,
        momentum_result: ScoreResult,
        sentiment_result: ScoreResult,
        weights: ScoringWeights,
        ticker: str,
    ) -> CandidateScore:
        """Calculate weighted composite score from individual factor scores.

        Args:
            value_result: Value scoring result
            growth_result: Growth scoring result
            quality_result: Quality scoring result
            momentum_result: Momentum scoring result
            sentiment_result: Sentiment scoring result
            weights: ScoringWeights defining factor weights
            ticker: Stock ticker symbol

        Returns:
            CandidateScore object with all scores and reasoning
        """
        # Import here to avoid circular imports
        from ..storage.research_models import CandidateScore

        # Calculate weighted composite
        composite = (
            value_result.score * weights.value +
            growth_result.score * weights.growth +
            quality_result.score * weights.quality +
            momentum_result.score * weights.momentum +
            sentiment_result.score * weights.sentiment
        )

        # Combine reasoning
        combined_reasoning = (
            f"Value ({value_result.score:.0f}): {value_result.reasoning}\n"
            f"Growth ({growth_result.score:.0f}): {growth_result.reasoning}\n"
            f"Quality ({quality_result.score:.0f}): {quality_result.reasoning}\n"
            f"Momentum ({momentum_result.score:.0f}): {momentum_result.reasoning}\n"
            f"Sentiment ({sentiment_result.score:.0f}): {sentiment_result.reasoning}"
        )

        return CandidateScore(
            ticker=ticker,
            value_score=value_result.score,
            growth_score=growth_result.score,
            quality_score=quality_result.score,
            momentum_score=momentum_result.score,
            sentiment_score=sentiment_result.score,
            composite_score=composite,
            reasoning=combined_reasoning,
        )

    async def score_stock(
        self,
        fundamentals: FundamentalsData,
        weights: ScoringWeights,
        price_change_1m: float | None = None,
        price_change_3m: float | None = None,
        price_change_6m: float | None = None,
        price_change_1y: float | None = None,
        rsi: float | None = None,
        vs_52w_high: float | None = None,
        vs_52w_low: float | None = None,
        recent_news_summary: str = "No recent news available",
        insider_activity: str = "No insider activity data",
        analyst_rating: str = "No analyst data",
        short_interest: float | None = None,
    ) -> CandidateScore:
        """Score a stock across all 5 factors and calculate composite.

        Convenience method that runs all scoring methods and combines results.

        Args:
            fundamentals: FundamentalsData for the stock
            weights: ScoringWeights for composite calculation
            price_change_*: Price change percentages for momentum
            rsi: RSI indicator for momentum
            vs_52w_*: Distance from 52-week high/low for momentum
            recent_news_summary: For sentiment scoring
            insider_activity: For sentiment scoring
            analyst_rating: For sentiment scoring
            short_interest: For sentiment scoring

        Returns:
            CandidateScore with all factor scores and composite
        """
        # Score each factor
        value_result = await self.score_value(fundamentals)
        growth_result = await self.score_growth(fundamentals)
        quality_result = await self.score_quality(fundamentals)

        momentum_result = await self.score_momentum(
            ticker=fundamentals.ticker,
            price_change_1m=price_change_1m,
            price_change_3m=price_change_3m,
            price_change_6m=price_change_6m,
            price_change_1y=price_change_1y,
            rsi=rsi,
            vs_52w_high=vs_52w_high,
            vs_52w_low=vs_52w_low,
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
        )

        sentiment_result = await self.score_sentiment(
            ticker=fundamentals.ticker,
            recent_news_summary=recent_news_summary,
            insider_activity=insider_activity,
            analyst_rating=analyst_rating,
            short_interest=short_interest,
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
        )

        # Calculate composite
        return await self.calculate_composite_score(
            value_result=value_result,
            growth_result=growth_result,
            quality_result=quality_result,
            momentum_result=momentum_result,
            sentiment_result=sentiment_result,
            weights=weights,
            ticker=fundamentals.ticker,
        )
