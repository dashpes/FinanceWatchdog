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
from .news_processor import NewsProcessor
from .prompts import (
    RELEVANCE_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARIZE_PROMPT,
)
from .research_prompts import (
    GROWTH_SCORE_PROMPT,
    MOMENTUM_SCORE_PROMPT,
    PROMPT_PLACEHOLDERS,
    QUALITY_SCORE_PROMPT,
    RESEARCH_PROMPTS,
    SENTIMENT_SCORE_PROMPT,
    VALUE_SCORE_PROMPT,
)
from .research_scorer import ResearchScorer, ScoreResult

__all__ = [
    # Local LLM
    "LocalLLM",
    # News Processing
    "NewsProcessor",
    # Claude API
    "ClaudeAnalyzer",
    "SynthesisResult",
    "WeeklyData",
    "WEEKLY_SYNTHESIS_PROMPT",
    # Prompts
    "RELEVANCE_PROMPT",
    "SENTIMENT_PROMPT",
    "SUMMARIZE_PROMPT",
    # Research Scoring Prompts
    "VALUE_SCORE_PROMPT",
    "GROWTH_SCORE_PROMPT",
    "QUALITY_SCORE_PROMPT",
    "MOMENTUM_SCORE_PROMPT",
    "SENTIMENT_SCORE_PROMPT",
    "RESEARCH_PROMPTS",
    "PROMPT_PLACEHOLDERS",
    # Research Scorer
    "ResearchScorer",
    "ScoreResult",
]
