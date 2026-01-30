"""Research Report Generator using Claude API.

This module provides a wrapper around the Anthropic Claude API for generating
comprehensive investment research reports. It includes cost tracking to
prevent overspending on API usage.

Example usage:
    from investment_monitor.analysis import ResearchReportGenerator
    from investment_monitor.collectors import FundamentalsData

    generator = ResearchReportGenerator(api_key="sk-ant-...", max_monthly_spend=50.00)

    if generator.is_available():
        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=fundamentals_data,
            score_result=candidate_score,
        )
        if result.success:
            print(result.report.summary)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..collectors.fundamentals import FundamentalsData
    from ..storage.research_models import CandidateScore, ResearchReport

# Cost estimates per million tokens (as of 2025)
# Claude 3.5 Sonnet: $3/M input, $15/M output
SONNET_INPUT_COST_PER_MILLION = 3.0
SONNET_OUTPUT_COST_PER_MILLION = 15.0

# Valid recommendations
VALID_RECOMMENDATIONS = ("strong_buy", "buy", "hold", "sell", "strong_sell")

RESEARCH_REPORT_PROMPT = """You are a senior equity research analyst. Generate a comprehensive investment research report.

## Company Information
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}

## Financial Metrics
{fundamentals_summary}

## Price Performance
{price_summary}

## Factor Scores (0-100)
Value: {value_score} - {value_reasoning}
Growth: {growth_score} - {growth_reasoning}
Quality: {quality_score} - {quality_reasoning}
Momentum: {momentum_score} - {momentum_reasoning}
Sentiment: {sentiment_score} - {sentiment_reasoning}
Composite: {composite_score}

## Congressional Trading Activity
{congress_summary}

## Generate a research report with the following sections:

1. **Executive Summary** (2-3 sentences): Key investment highlights
2. **Investment Thesis**: Core argument for/against investing
3. **Bull Case**: 3 key reasons the stock could outperform
4. **Bear Case**: 3 key risks to consider
5. **Recommendation**: One of: strong_buy, buy, hold, sell, strong_sell
6. **Target Price Range**: Low/Mid/High scenarios with reasoning

Respond with a structured JSON:
{{
    "summary": "...",
    "thesis": "...",
    "bull_case": "...",
    "bear_case": "...",
    "recommendation": "strong_buy|buy|hold|sell|strong_sell",
    "target_price": <number or null>
}}
"""


@dataclass
class ReportResult:
    """Result of a research report generation request.

    Attributes:
        report: The generated ResearchReport (ORM model), or None if failed
        success: Whether the report was successfully generated
        error_message: Error message if generation failed
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        cost: Estimated cost in USD
        timestamp: When the report was generated
    """

    report: ResearchReport | None
    success: bool = True
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ResearchReportGenerator:
    """Claude API client for generating investment research reports.

    This class wraps the Anthropic API client and provides:
    - Deep research report generation using Claude
    - Cost tracking to prevent overspending
    - Graceful handling of missing API keys and errors
    - JSON response parsing with validation

    Follows the ClaudeAnalyzer pattern for consistency.

    Attributes:
        max_monthly_spend: Maximum monthly spend limit in USD
        model: Claude model to use
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_monthly_spend: float = 50.0,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Research Report Generator.

        Args:
            api_key: Anthropic API key. If None, API features are disabled.
            max_monthly_spend: Maximum monthly spend limit in USD (default: $50.00)
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
                logger.warning("anthropic package not installed, research report features unavailable")

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
            return f"{value * 100:.2f}%"
        return f"{value:.2f}"

    def _format_currency(self, value: float | None) -> str:
        """Format a currency value.

        Args:
            value: The value to format (may be None)

        Returns:
            Formatted currency string
        """
        if value is None:
            return "N/A"
        if abs(value) >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value:,.2f}"

    def _build_fundamentals_summary(self, fundamentals: FundamentalsData) -> str:
        """Build a formatted summary of fundamental metrics.

        Args:
            fundamentals: FundamentalsData containing financial metrics

        Returns:
            Formatted string summary of fundamentals
        """
        lines = []

        # Valuation metrics
        lines.append("### Valuation")
        lines.append(f"- P/E Ratio: {self._format_value(fundamentals.pe_ratio)}")
        lines.append(f"- P/B Ratio: {self._format_value(fundamentals.pb_ratio)}")
        lines.append(f"- P/S Ratio: {self._format_value(fundamentals.ps_ratio)}")
        lines.append(f"- PEG Ratio: {self._format_value(fundamentals.peg_ratio)}")

        # Growth metrics
        lines.append("\n### Growth")
        lines.append(f"- Revenue Growth (YoY): {self._format_value(fundamentals.revenue_growth_yoy, as_percent=True)}")
        lines.append(f"- Revenue Growth (3Y CAGR): {self._format_value(fundamentals.revenue_growth_3y, as_percent=True)}")
        lines.append(f"- EPS Growth (YoY): {self._format_value(fundamentals.eps_growth_yoy, as_percent=True)}")
        lines.append(f"- EPS Growth (3Y CAGR): {self._format_value(fundamentals.eps_growth_3y, as_percent=True)}")

        # Quality metrics
        lines.append("\n### Quality")
        lines.append(f"- ROE: {self._format_value(fundamentals.roe, as_percent=True)}")
        lines.append(f"- Profit Margin: {self._format_value(fundamentals.profit_margin, as_percent=True)}")
        lines.append(f"- Debt/Equity: {self._format_value(fundamentals.debt_to_equity)}")
        lines.append(f"- Current Ratio: {self._format_value(fundamentals.current_ratio)}")

        # Income metrics
        lines.append("\n### Income")
        lines.append(f"- Dividend Yield: {self._format_value(fundamentals.dividend_yield, as_percent=True)}")
        lines.append(f"- Free Cash Flow: {self._format_currency(fundamentals.free_cash_flow)}")
        lines.append(f"- Market Cap: {self._format_currency(fundamentals.market_cap)}")

        return "\n".join(lines)

    def _build_report_prompt(
        self,
        ticker: str,
        company_name: str,
        fundamentals: FundamentalsData,
        score_result: CandidateScore,
        price_summary: str = "",
        congress_summary: str = "",
    ) -> str:
        """Build the research report prompt.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            fundamentals: FundamentalsData containing financial metrics
            score_result: CandidateScore with factor scores and reasoning
            price_summary: Optional summary of price performance
            congress_summary: Optional summary of congressional trading activity

        Returns:
            Formatted prompt string for Claude.
        """
        # Extract individual factor reasoning from the combined reasoning
        # The reasoning is formatted as "Factor (score): reasoning\nFactor2..."
        reasoning_parts = self._parse_score_reasoning(score_result.reasoning or "")

        return RESEARCH_REPORT_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            sector=fundamentals.sector or "Unknown",
            industry=fundamentals.industry or "Unknown",
            fundamentals_summary=self._build_fundamentals_summary(fundamentals),
            price_summary=price_summary or "No price data available",
            value_score=f"{score_result.value_score:.0f}" if score_result.value_score else "N/A",
            value_reasoning=reasoning_parts.get("value", "No reasoning available"),
            growth_score=f"{score_result.growth_score:.0f}" if score_result.growth_score else "N/A",
            growth_reasoning=reasoning_parts.get("growth", "No reasoning available"),
            quality_score=f"{score_result.quality_score:.0f}" if score_result.quality_score else "N/A",
            quality_reasoning=reasoning_parts.get("quality", "No reasoning available"),
            momentum_score=f"{score_result.momentum_score:.0f}" if score_result.momentum_score else "N/A",
            momentum_reasoning=reasoning_parts.get("momentum", "No reasoning available"),
            sentiment_score=f"{score_result.sentiment_score:.0f}" if score_result.sentiment_score else "N/A",
            sentiment_reasoning=reasoning_parts.get("sentiment", "No reasoning available"),
            composite_score=f"{score_result.composite_score:.0f}" if score_result.composite_score else "N/A",
            congress_summary=congress_summary or "No congressional trading data available",
        )

    def _parse_score_reasoning(self, combined_reasoning: str) -> dict[str, str]:
        """Parse combined reasoning string into individual factor reasoning.

        The combined reasoning is formatted as:
        "Value (80): reasoning\nGrowth (70): reasoning\n..."

        Args:
            combined_reasoning: Combined reasoning from CandidateScore

        Returns:
            Dictionary mapping factor names to their reasoning
        """
        result = {}
        if not combined_reasoning:
            return result

        # Parse each line looking for "Factor (score): reasoning" pattern
        pattern = r"(Value|Growth|Quality|Momentum|Sentiment)\s*\(\d+\):\s*(.+?)(?=\n[A-Z]|\Z)"
        matches = re.findall(pattern, combined_reasoning, re.IGNORECASE | re.DOTALL)

        for factor, reasoning in matches:
            result[factor.lower()] = reasoning.strip()

        return result

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
        # Look for {...} pattern, handling nested braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)

        # Try each match to find valid JSON with expected fields
        for match in matches:
            try:
                data = json.loads(match)
                # Check for expected fields
                if any(key in data for key in ["summary", "thesis", "recommendation"]):
                    return match
            except json.JSONDecodeError:
                continue

        # If no valid JSON found, try the whole text
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        return None

    def _parse_report_response(self, response: str | None, ticker: str) -> ReportResult:
        """Parse Claude's JSON response into a ResearchReport.

        Args:
            response: The raw JSON response from Claude
            ticker: Stock ticker symbol

        Returns:
            ReportResult with parsed ResearchReport or error message
        """
        # Import here to avoid circular imports
        from ..storage.research_models import ResearchReport

        if not response:
            return ReportResult(
                report=None,
                success=False,
                error_message="Empty response from API",
            )

        # Extract JSON from response
        json_str = self._extract_json(response)
        if not json_str:
            return ReportResult(
                report=None,
                success=False,
                error_message=f"Failed to extract JSON from response: {response[:200]}...",
            )

        try:
            data = json.loads(json_str)

            # Validate recommendation
            recommendation = data.get("recommendation", "hold")
            if recommendation not in VALID_RECOMMENDATIONS:
                logger.warning(f"Invalid recommendation '{recommendation}', defaulting to 'hold'")
                recommendation = "hold"

            # Extract target price (may be null)
            target_price = data.get("target_price")
            if target_price is not None:
                try:
                    target_price = float(target_price)
                except (TypeError, ValueError):
                    target_price = None

            # Create ResearchReport ORM object
            report = ResearchReport(
                ticker=ticker,
                summary=data.get("summary", ""),
                thesis=data.get("thesis", ""),
                bull_case=data.get("bull_case", ""),
                bear_case=data.get("bear_case", ""),
                recommendation=recommendation,
                target_price=target_price,
            )

            return ReportResult(
                report=report,
                success=True,
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return ReportResult(
                report=None,
                success=False,
                error_message=f"JSON parse error: {str(e)}",
            )

    async def generate_report(
        self,
        ticker: str,
        company_name: str,
        fundamentals: FundamentalsData,
        score_result: CandidateScore,
        price_summary: str = "",
        congress_summary: str = "",
        max_tokens: int = 2000,
    ) -> ReportResult:
        """Generate a comprehensive research report for a stock.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            fundamentals: FundamentalsData containing financial metrics
            score_result: CandidateScore with factor scores and reasoning
            price_summary: Optional summary of price performance
            congress_summary: Optional summary of congressional trading activity
            max_tokens: Maximum tokens for the response (default: 2000)

        Returns:
            ReportResult containing the ResearchReport or error message.
        """
        if not self.is_available():
            return ReportResult(
                report=None,
                success=False,
                error_message="Research report generation unavailable (no API key or anthropic not installed)",
            )

        if not self._within_budget():
            return ReportResult(
                report=None,
                success=False,
                error_message=f"Research report generation skipped (budget limit of ${self.max_monthly_spend:.2f} reached)",
            )

        prompt = self._build_report_prompt(
            ticker=ticker,
            company_name=company_name,
            fundamentals=fundamentals,
            score_result=score_result,
            price_summary=price_summary,
            congress_summary=congress_summary,
        )

        try:
            # Use synchronous call wrapped for async compatibility
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._record_cost(input_tokens, output_tokens)

            response_text = response.content[0].text

            # Parse the response
            result = self._parse_report_response(response_text, ticker)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.cost = cost

            return result

        except Exception as e:
            logger.error(f"API error generating report for {ticker}: {e}")
            return ReportResult(
                report=None,
                success=False,
                error_message=f"API error: {str(e)}",
            )

    def generate_report_sync(
        self,
        ticker: str,
        company_name: str,
        fundamentals: FundamentalsData,
        score_result: CandidateScore,
        price_summary: str = "",
        congress_summary: str = "",
        max_tokens: int = 2000,
    ) -> ReportResult:
        """Synchronous version of generate_report.

        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            fundamentals: FundamentalsData containing financial metrics
            score_result: CandidateScore with factor scores and reasoning
            price_summary: Optional summary of price performance
            congress_summary: Optional summary of congressional trading activity
            max_tokens: Maximum tokens for the response (default: 2000)

        Returns:
            ReportResult containing the ResearchReport or error message.
        """
        if not self.is_available():
            return ReportResult(
                report=None,
                success=False,
                error_message="Research report generation unavailable (no API key or anthropic not installed)",
            )

        if not self._within_budget():
            return ReportResult(
                report=None,
                success=False,
                error_message=f"Research report generation skipped (budget limit of ${self.max_monthly_spend:.2f} reached)",
            )

        prompt = self._build_report_prompt(
            ticker=ticker,
            company_name=company_name,
            fundamentals=fundamentals,
            score_result=score_result,
            price_summary=price_summary,
            congress_summary=congress_summary,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._record_cost(input_tokens, output_tokens)

            response_text = response.content[0].text

            # Parse the response
            result = self._parse_report_response(response_text, ticker)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.cost = cost

            return result

        except Exception as e:
            logger.error(f"API error generating report for {ticker}: {e}")
            return ReportResult(
                report=None,
                success=False,
                error_message=f"API error: {str(e)}",
            )
