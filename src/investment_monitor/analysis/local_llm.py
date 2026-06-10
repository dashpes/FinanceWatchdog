"""Local LLM integration using Ollama for news analysis."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from .ollama_client import has_model, model_names, response_text
from .prompts import RELEVANCE_PROMPT, SENTIMENT_PROMPT, SUMMARIZE_PROMPT

if TYPE_CHECKING:
    import ollama

    from investment_monitor.models import Portfolio

    from .claude_api import WeeklyData


class LocalLLM:
    """Client wrapper for Ollama local LLM inference.

    Provides methods for news relevance scoring, sentiment analysis,
    and text summarization using local Ollama models.
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize the LocalLLM client.

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
                logger.warning("ollama package not installed, LLM features unavailable")
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
            # List models to check if server is running.
            response = client.list()

            if has_model(response, self.model):
                self._available = True
                return True

            # Model not found but server is running.
            logger.warning(
                f"Model {self.model} not found in Ollama. "
                f"Available: {model_names(response)}"
            )
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
                    "temperature": 0.1,  # Low temperature for more deterministic output
                    "num_predict": 50,   # Limit response length
                },
            )
            return response_text(response)
        except Exception as e:
            logger.debug(f"LLM generation failed: {e}")
            return None

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        num_predict: int = 200,
    ) -> str | None:
        """Generate text with caller-specified options (object-based API).

        Unlike the internal ``_generate`` (tuned for short scoring replies), this
        lets longer-form callers such as research-report generation request a
        larger token budget.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Sampling temperature.
            num_predict: Maximum tokens to generate.

        Returns:
            The generated text, or None if unavailable.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            )
            return response_text(response) or None
        except Exception as e:
            logger.debug(f"LLM generation failed: {e}")
            return None

    async def score_relevance(
        self,
        headline: str,
        ticker: str,
        company_name: str = "",
    ) -> float:
        """Score how relevant a news headline is to a specific stock.

        Args:
            headline: The news headline to evaluate.
            ticker: The stock ticker symbol.
            company_name: The company name (optional, improves accuracy).

        Returns:
            A relevance score from 0-10, or -1 if LLM is unavailable.
        """
        if not self.is_available():
            return -1.0

        prompt = RELEVANCE_PROMPT.format(
            ticker=ticker,
            company_name=company_name or ticker,
            headline=headline,
        )

        response = self._generate(prompt)
        if response is None:
            return -1.0

        # Parse the response to extract a number
        score = self._parse_score(response)
        return score if score is not None else -1.0

    async def classify_sentiment(
        self,
        text: str,
    ) -> str:
        """Classify the sentiment of a financial text.

        Args:
            text: The text to analyze (headline or short article).

        Returns:
            One of 'bullish', 'bearish', 'neutral', or 'unknown' if unavailable.
        """
        if not self.is_available():
            return "unknown"

        prompt = SENTIMENT_PROMPT.format(headline=text)

        response = self._generate(prompt)
        if response is None:
            return "unknown"

        # Parse the response to extract sentiment
        sentiment = self._parse_sentiment(response)
        return sentiment

    async def summarize(
        self,
        text: str,
        max_sentences: int = 2,
    ) -> str:
        """Summarize text to a specified number of sentences.

        Args:
            text: The text to summarize.
            max_sentences: Maximum number of sentences in the summary.

        Returns:
            The summarized text, or empty string if unavailable.
        """
        if not self.is_available():
            return ""

        prompt = SUMMARIZE_PROMPT.format(
            text=text,
            max_sentences=max_sentences,
        )

        response = self._generate(prompt)
        if response is None:
            return ""

        return response

    async def generate_weekly_synthesis(
        self,
        alert_counts: dict[str, int],
        top_movers: list[tuple[str, float]],
        portfolio_change_pct: float | None = None,
    ) -> str:
        """Generate a weekly synthesis narrative.

        Args:
            alert_counts: Dict of alert_type -> count.
            top_movers: List of (ticker, percent_change) tuples.
            portfolio_change_pct: Portfolio change percentage.

        Returns:
            Synthesis text, or empty string if unavailable.
        """
        if not self.is_available():
            return ""

        # Format inputs for prompt
        alert_str = ", ".join(f"{count} {atype}" for atype, count in alert_counts.items())
        movers_str = ", ".join(f"{ticker} {change:+.1f}%" for ticker, change in top_movers[:5])
        portfolio_str = f"{portfolio_change_pct:+.1f}%" if portfolio_change_pct is not None else "N/A"

        from .prompts import LOCAL_LLM_WEEKLY_SYNTHESIS_PROMPT

        prompt = LOCAL_LLM_WEEKLY_SYNTHESIS_PROMPT.format(
            alert_counts=alert_str or "None",
            top_movers=movers_str or "None",
            portfolio_change=portfolio_str,
        )

        # Use longer response for synthesis
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 150,
                },
            )
            return response_text(response)
        except Exception as e:
            logger.debug(f"Weekly synthesis generation failed: {e}")
            return ""

    async def weekly_synthesis(
        self,
        portfolio: Portfolio,
        week_data: WeeklyData,
        max_tokens: int = 400,
    ) -> str:
        """Generate a full weekly portfolio synthesis locally (free, no API key).

        This mirrors ClaudeAnalyzer.weekly_synthesis but runs entirely on the
        local Ollama model, using the same rich prompt (portfolio + price /
        insider / news / earnings summaries). It is the default tier-2 provider.

        Args:
            portfolio: The user's portfolio to analyze.
            week_data: Aggregated data for the week.
            max_tokens: Maximum tokens for the response.

        Returns:
            The synthesis text, or an empty string if the local LLM is unavailable.
        """
        if not self.is_available():
            return ""

        # Imported lazily to keep the module import graph light and avoid any
        # import-order coupling within the analysis package.
        from .claude_api import build_weekly_synthesis_prompt

        prompt = build_weekly_synthesis_prompt(portfolio, week_data)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": max_tokens,
                },
            )
            return response_text(response)
        except Exception as e:
            logger.debug(f"Local weekly synthesis generation failed: {e}")
            return ""

    @staticmethod
    def _parse_score(response: str) -> float | None:
        """Parse a relevance score from LLM response.

        Handles various formats LLMs might use:
        - Just a number: "7"
        - With decimals: "7.5"
        - With text: "Rating: 7" or "7/10"
        - Verbose: "I would rate this a 7 out of 10"

        Args:
            response: The raw LLM response.

        Returns:
            The parsed score (0-10), or None if parsing fails.
        """
        if not response:
            return None

        # Clean up the response
        response = response.strip().lower()

        # Try to find a number in the response
        # First, try to match a decimal or integer at the start or standalone
        patterns = [
            r"^(\d+\.?\d*)",          # Number at start
            r"rating[:\s]*(\d+\.?\d*)",  # "Rating: X" format
            r"(\d+\.?\d*)\s*(?:/\s*10|out of 10)",  # "X/10" or "X out of 10"
            r"\b(\d+\.?\d*)\b",        # Any standalone number
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to valid range
                    return max(0.0, min(10.0, score))
                except ValueError:
                    continue

        return None

    @staticmethod
    def _parse_sentiment(response: str) -> str:
        """Parse sentiment classification from LLM response.

        Handles various formats:
        - Clean: "bullish"
        - With text: "Sentiment: bullish"
        - Verbose: "I would classify this as bullish"

        Args:
            response: The raw LLM response.

        Returns:
            One of 'bullish', 'bearish', 'neutral', or 'unknown'.
        """
        if not response:
            return "unknown"

        response = response.strip().lower()

        # Check for each sentiment keyword
        if "bullish" in response:
            return "bullish"
        elif "bearish" in response:
            return "bearish"
        elif "neutral" in response:
            return "neutral"

        # Try to match positive/negative synonyms
        positive_words = ["positive", "good", "up", "gain"]
        negative_words = ["negative", "bad", "down", "loss", "decline"]

        for word in positive_words:
            if word in response:
                return "bullish"

        for word in negative_words:
            if word in response:
                return "bearish"

        return "unknown"
