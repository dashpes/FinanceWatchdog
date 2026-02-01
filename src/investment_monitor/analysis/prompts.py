"""Prompt templates for local LLM analysis tasks."""

RELEVANCE_PROMPT = """You are a financial news filter. Rate how relevant this headline is to the stock.

Ticker: {ticker}
Company: {company_name}
Headline: {headline}

Rate relevance from 1-10:
1-3: Not relevant (different company, unrelated topic)
4-6: Tangentially relevant (same industry, indirect impact)
7-10: Directly relevant (about this company, material impact)

Respond with ONLY a single number 1-10, nothing else.

Rating:"""

SENTIMENT_PROMPT = """Classify the sentiment of this financial news headline.

Headline: {headline}

Is this news bullish (positive for stock price), bearish (negative for stock price), or neutral?

Respond with ONLY one word: bullish, bearish, or neutral

Sentiment:"""

SUMMARIZE_PROMPT = """Summarize the following text in {max_sentences} sentence(s) or less. Be concise and capture the key points.

Text: {text}

Summary:"""

LOCAL_LLM_WEEKLY_SYNTHESIS_PROMPT = """You are an investment analyst summarizing the week's activity.

Week Summary:
- Alert counts: {alert_counts}
- Top movers: {top_movers}
- Portfolio change: {portfolio_change}

Generate a 2-3 sentence synthesis for an investor. Focus on:
1. Key trends or patterns
2. Notable events
3. What to watch next week

Be concise and actionable. No bullet points.

Synthesis:"""
