"""Analysis module for news processing and AI-powered portfolio insights.

This module provides:
- Local LLM integration for news processing and sentiment analysis
- Claude API integration for weekly portfolio synthesis

Example usage:
    from investment_monitor.analysis import ClaudeAnalyzer, WeeklyData

    analyzer = ClaudeAnalyzer(api_key="your-api-key", max_monthly_spend=5.00)

    week_data = WeeklyData(
        price_summary="AAPL up 3%, MSFT down 2%",
        insider_summary="No significant transactions",
        news_summary="Apple announced new product",
        earnings_summary="MSFT reports next week",
    )

    if analyzer.is_available():
        synthesis = await analyzer.weekly_synthesis(portfolio, week_data)
        print(synthesis.synthesis)
"""

from .claude_api import (
    ClaudeAnalyzer,
    SynthesisResult,
    WeeklyData,
    WEEKLY_SYNTHESIS_PROMPT,
)
from .local_llm import LocalLLM
from .prompts import (
    RELEVANCE_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARIZE_PROMPT,
)

__all__ = [
    # Local LLM
    "LocalLLM",
    # Claude API
    "ClaudeAnalyzer",
    "SynthesisResult",
    "WeeklyData",
    "WEEKLY_SYNTHESIS_PROMPT",
    # Prompts
    "RELEVANCE_PROMPT",
    "SENTIMENT_PROMPT",
    "SUMMARIZE_PROMPT",
]
