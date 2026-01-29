"""Claude API client for weekly portfolio synthesis.

This module provides a wrapper around the Anthropic Claude API for generating
AI-powered weekly portfolio synthesis reports. It includes cost tracking to
prevent overspending on API usage.

Example usage:
    from investment_monitor.analysis import ClaudeAnalyzer, WeeklyData
    from investment_monitor.models import Portfolio

    analyzer = ClaudeAnalyzer(api_key="sk-ant-...", max_monthly_spend=5.00)

    week_data = WeeklyData(
        price_summary="AAPL up 3%, MSFT down 2%",
        insider_summary="CEO purchased 10,000 shares",
        news_summary="Apple announced new product line",
        earnings_summary="MSFT reports earnings next Tuesday",
    )

    if analyzer.is_available():
        result = await analyzer.weekly_synthesis(portfolio, week_data)
        print(result.synthesis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from investment_monitor.models import Portfolio

# Cost estimates per million tokens (as of 2025)
# Claude 3.5 Sonnet: $3/M input, $15/M output
SONNET_INPUT_COST_PER_MILLION = 3.0
SONNET_OUTPUT_COST_PER_MILLION = 15.0

WEEKLY_SYNTHESIS_PROMPT = """You are my personal investment analyst. Review this week's activity for my portfolio and provide actionable insights.

## My Portfolio

{portfolio_yaml}

## This Week's Activity

### Price Movements
{price_summary}

### Insider Transactions
{insider_summary}

### Relevant News
{news_summary}

### Upcoming Earnings
{earnings_summary}

## Your Analysis

Please provide:
1. The 2-3 most important developments this week and why they matter
2. Any developments that contradict my stated investment thesis for a holding
3. Positions that warrant closer attention or research
4. Any notable patterns (sector rotation, insider behavior, etc.)

Keep your response concise and actionable. Focus on signal, not noise."""


class WeeklyData(BaseModel):
    """Aggregated data for weekly synthesis.

    Attributes:
        price_summary: Summary of price movements for the week
        insider_summary: Summary of insider transactions
        news_summary: Summary of relevant news
        earnings_summary: Summary of upcoming/recent earnings
        week_start: Start date of the reporting week
        week_end: End date of the reporting week
    """

    price_summary: str = Field(default="No significant price movements this week.")
    insider_summary: str = Field(default="No insider transactions reported.")
    news_summary: str = Field(default="No significant news this week.")
    earnings_summary: str = Field(default="No upcoming earnings in the next 7 days.")
    week_start: date | None = None
    week_end: date | None = None


@dataclass
class SynthesisResult:
    """Result of a weekly synthesis request.

    Attributes:
        synthesis: The AI-generated synthesis text
        success: Whether the synthesis was successfully generated
        error_message: Error message if synthesis failed
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost: Estimated cost in USD
        timestamp: When the synthesis was generated
    """

    synthesis: str
    success: bool = True
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ClaudeAnalyzer:
    """Claude API client for weekly portfolio synthesis.

    This class wraps the Anthropic API client and provides:
    - Weekly portfolio synthesis generation
    - Cost tracking to prevent overspending
    - Graceful handling of missing API keys and errors

    Attributes:
        max_monthly_spend: Maximum monthly spend limit in USD
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_monthly_spend: float = 5.00,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Claude analyzer.

        Args:
            api_key: Anthropic API key. If None, API features are disabled.
            max_monthly_spend: Maximum monthly spend limit in USD (default: $5.00)
            model: Claude model to use (default: claude-sonnet-4-20250514)
        """
        self._client = None
        self._api_key = api_key
        self.max_monthly_spend = max_monthly_spend
        self.model = model
        self._monthly_spend = 0.0
        self._spend_reset_date: date | None = None

        if api_key:
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=api_key)
            except ImportError:
                # anthropic package not installed
                pass

    def is_available(self) -> bool:
        """Check if Claude API is configured and available.

        Returns:
            True if API key is set and anthropic package is installed.
        """
        return self._client is not None

    def get_monthly_spend(self) -> float:
        """Get current monthly spend.

        Returns:
            Current spend in USD for the current month.
        """
        self._check_spend_reset()
        return self._monthly_spend

    def get_remaining_budget(self) -> float:
        """Get remaining budget for the current month.

        Returns:
            Remaining budget in USD.
        """
        self._check_spend_reset()
        return max(0.0, self.max_monthly_spend - self._monthly_spend)

    def _check_spend_reset(self) -> None:
        """Reset monthly spend counter if a new month has started."""
        today = date.today()
        if self._spend_reset_date is None or self._spend_reset_date.month != today.month:
            self._monthly_spend = 0.0
            self._spend_reset_date = today

    def _within_budget(self) -> bool:
        """Check if we're within monthly spend limit.

        Returns:
            True if current spend is below the limit.
        """
        self._check_spend_reset()
        return self._monthly_spend < self.max_monthly_spend

    def _record_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Track API usage costs.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Returns:
            The cost of this API call in USD.
        """
        input_cost = (input_tokens / 1_000_000) * SONNET_INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * SONNET_OUTPUT_COST_PER_MILLION
        call_cost = input_cost + output_cost
        self._monthly_spend += call_cost
        return call_cost

    def _build_synthesis_prompt(self, portfolio: Portfolio, week_data: WeeklyData) -> str:
        """Build the weekly synthesis prompt.

        Args:
            portfolio: The user's portfolio
            week_data: Aggregated weekly data

        Returns:
            Formatted prompt string for Claude.
        """
        # Convert portfolio to YAML for clear formatting
        portfolio_dict = {
            "holdings": [
                {
                    "ticker": h.ticker,
                    "shares": float(h.shares),
                    "cost_basis": float(h.cost_basis),
                    "thesis": h.thesis or "No thesis specified",
                }
                for h in portfolio.holdings
            ],
            "watchlist": [
                {
                    "ticker": w.ticker,
                    "reason": w.reason or "No reason specified",
                    "target_price": float(w.target_price) if w.target_price else None,
                }
                for w in portfolio.watchlist
            ],
        }
        portfolio_yaml = yaml.dump(portfolio_dict, default_flow_style=False, sort_keys=False)

        return WEEKLY_SYNTHESIS_PROMPT.format(
            portfolio_yaml=portfolio_yaml,
            price_summary=week_data.price_summary,
            insider_summary=week_data.insider_summary,
            news_summary=week_data.news_summary,
            earnings_summary=week_data.earnings_summary,
        )

    async def weekly_synthesis(
        self,
        portfolio: Portfolio,
        week_data: WeeklyData,
        max_tokens: int = 1000,
    ) -> SynthesisResult:
        """Generate weekly portfolio synthesis.

        Args:
            portfolio: The user's portfolio to analyze
            week_data: Aggregated data for the week
            max_tokens: Maximum tokens for the response (default: 1000)

        Returns:
            SynthesisResult containing the synthesis text or error message.
        """
        if not self.is_available():
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message="Weekly AI synthesis unavailable (no API key or anthropic not installed)",
            )

        if not self._within_budget():
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message=f"Weekly AI synthesis skipped (budget limit of ${self.max_monthly_spend:.2f} reached)",
            )

        prompt = self._build_synthesis_prompt(portfolio, week_data)

        try:
            # Use synchronous call wrapped for async compatibility
            # The Anthropic SDK's sync client works fine in async context
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._record_cost(input_tokens, output_tokens)

            synthesis_text = response.content[0].text

            return SynthesisResult(
                synthesis=synthesis_text,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        except Exception as e:
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message=f"API error: {str(e)}",
            )

    def weekly_synthesis_sync(
        self,
        portfolio: Portfolio,
        week_data: WeeklyData,
        max_tokens: int = 1000,
    ) -> SynthesisResult:
        """Synchronous version of weekly_synthesis.

        Args:
            portfolio: The user's portfolio to analyze
            week_data: Aggregated data for the week
            max_tokens: Maximum tokens for the response (default: 1000)

        Returns:
            SynthesisResult containing the synthesis text or error message.
        """
        if not self.is_available():
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message="Weekly AI synthesis unavailable (no API key or anthropic not installed)",
            )

        if not self._within_budget():
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message=f"Weekly AI synthesis skipped (budget limit of ${self.max_monthly_spend:.2f} reached)",
            )

        prompt = self._build_synthesis_prompt(portfolio, week_data)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._record_cost(input_tokens, output_tokens)

            synthesis_text = response.content[0].text

            return SynthesisResult(
                synthesis=synthesis_text,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )

        except Exception as e:
            return SynthesisResult(
                synthesis="",
                success=False,
                error_message=f"API error: {str(e)}",
            )
