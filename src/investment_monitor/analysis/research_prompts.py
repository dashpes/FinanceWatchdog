"""Prompt templates for multi-factor research scoring using local LLM.

These prompts are designed to be used with Ollama for AI-powered stock analysis.
Each prompt returns a JSON response with a score (0-100) and reasoning.

Example usage:
    from investment_monitor.analysis.research_prompts import VALUE_SCORE_PROMPT

    prompt = VALUE_SCORE_PROMPT.format(
        ticker="AAPL",
        company_name="Apple Inc.",
        pe_ratio=28.5,
        pb_ratio=45.2,
        ps_ratio=7.8,
        peg_ratio=2.1,
        dividend_yield=0.5,
        free_cash_flow=99000000000,
        sector="Technology",
        industry="Consumer Electronics",
    )
"""

VALUE_SCORE_PROMPT = """You are a value investing analyst. Evaluate the valuation metrics for this stock and provide a score.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Valuation Metrics:
- P/E Ratio: {pe_ratio}
- P/B Ratio: {pb_ratio}
- P/S Ratio: {ps_ratio}
- PEG Ratio: {peg_ratio}
- Dividend Yield: {dividend_yield}%
- Free Cash Flow: ${free_cash_flow}

Scoring Guidelines:
- 80-100: Significantly undervalued (low P/E, P/B relative to peers and historical averages, strong FCF yield)
- 60-79: Modestly undervalued (reasonable valuations, some margin of safety)
- 40-59: Fairly valued (valuations in line with peers and growth)
- 20-39: Overvalued (elevated multiples relative to growth and peers)
- 0-19: Significantly overvalued (extreme valuations, minimal margin of safety)

Consider:
1. How do these metrics compare to typical ranges for the {sector} sector?
2. Is the company generating strong free cash flow relative to its valuation?
3. Does the PEG ratio suggest growth justifies the P/E?
4. Is there a dividend providing additional return?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

GROWTH_SCORE_PROMPT = """You are a growth investing analyst. Evaluate the growth trajectory for this stock and provide a score.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Growth Metrics:
- Revenue Growth (YoY): {revenue_growth_yoy}%
- Revenue Growth (3-Year CAGR): {revenue_growth_3y}%
- EPS Growth (YoY): {eps_growth_yoy}%
- EPS Growth (3-Year CAGR): {eps_growth_3y}%

Scoring Guidelines:
- 80-100: Exceptional growth (20%+ revenue/EPS growth, sustainable trajectory, market expansion)
- 60-79: Strong growth (10-20% growth, clear growth catalysts)
- 40-59: Moderate growth (0-10% growth, stable but not accelerating)
- 20-39: Declining growth (negative growth trends, slowing momentum)
- 0-19: Significant decline (severe revenue/earnings contraction)

Consider:
1. Is growth accelerating or decelerating year over year?
2. How sustainable is this growth rate for the {industry} industry?
3. Is EPS growth keeping pace with revenue growth (operating leverage)?
4. Are there signs of market saturation or competitive pressure?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

QUALITY_SCORE_PROMPT = """You are a quality investing analyst. Evaluate the business quality metrics for this stock and provide a score.

Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}

Quality Metrics:
- Return on Equity (ROE): {roe}%
- Profit Margin: {profit_margin}%
- Debt-to-Equity Ratio: {debt_to_equity}
- Current Ratio: {current_ratio}

Scoring Guidelines:
- 80-100: Exceptional quality (ROE > 20%, high margins, low debt, strong liquidity)
- 60-79: Good quality (ROE 15-20%, healthy margins, manageable debt)
- 40-59: Average quality (ROE 10-15%, moderate margins, moderate debt)
- 20-39: Below average quality (ROE < 10%, thin margins, elevated debt)
- 0-19: Poor quality (negative ROE, losses, high debt burden, liquidity concerns)

Consider:
1. Is the ROE sustainable or driven by excessive leverage?
2. How do profit margins compare to {industry} peers?
3. Can the company service its debt comfortably?
4. Is liquidity sufficient to weather economic downturns?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

MOMENTUM_SCORE_PROMPT = """You are a technical analyst. Evaluate the price momentum for this stock and provide a score.

Company: {company_name} ({ticker})

Price Performance:
- 1-Month Change: {price_change_1m}%
- 3-Month Change: {price_change_3m}%
- 6-Month Change: {price_change_6m}%
- 1-Year Change: {price_change_1y}%
- RSI (14-day): {rsi}
- Distance from 52-Week High: {vs_52w_high}%
- Distance from 52-Week Low: {vs_52w_low}%

Scoring Guidelines:
- 80-100: Strong uptrend (positive returns across timeframes, healthy consolidation, RSI 50-70)
- 60-79: Positive momentum (recent gains, upward trend forming)
- 40-59: Neutral/sideways (mixed performance, consolidation phase)
- 20-39: Negative momentum (recent losses, downward pressure)
- 0-19: Strong downtrend (significant losses, capitulation, RSI < 30)

Consider:
1. Is momentum consistent across multiple timeframes?
2. Is the RSI indicating overbought (>70) or oversold (<30) conditions?
3. Is the stock near its 52-week high (strength) or low (weakness)?
4. Are short-term trends aligned with longer-term trends?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

SENTIMENT_SCORE_PROMPT = """You are a market sentiment analyst. Evaluate the overall market sentiment for this stock and provide a score.

Company: {company_name} ({ticker})

Sentiment Indicators:
- Recent News Summary: {recent_news_summary}
- Insider Activity: {insider_activity}
- Analyst Rating: {analyst_rating}
- Short Interest: {short_interest}%

Scoring Guidelines:
- 80-100: Very bullish (positive news flow, insider buying, strong analyst support, low short interest)
- 60-79: Moderately bullish (mostly positive news, some insider buying, favorable ratings)
- 40-59: Neutral (mixed news, minimal insider activity, hold ratings)
- 20-39: Moderately bearish (negative news, insider selling, downgrades, elevated short interest)
- 0-19: Very bearish (severe negative news, significant insider selling, analyst warnings, high short interest)

Consider:
1. What is the overall tone of recent news coverage?
2. Are insiders buying (confidence) or selling (concern)?
3. What is the analyst consensus and recent rating changes?
4. Does short interest suggest widespread bearish bets?

Respond with ONLY valid JSON in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation>"}}"""

# List of all research prompts for iteration
RESEARCH_PROMPTS = {
    "value": VALUE_SCORE_PROMPT,
    "growth": GROWTH_SCORE_PROMPT,
    "quality": QUALITY_SCORE_PROMPT,
    "momentum": MOMENTUM_SCORE_PROMPT,
    "sentiment": SENTIMENT_SCORE_PROMPT,
}

# Expected placeholders for each prompt (for validation)
PROMPT_PLACEHOLDERS = {
    "value": [
        "ticker",
        "company_name",
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "peg_ratio",
        "dividend_yield",
        "free_cash_flow",
        "sector",
        "industry",
    ],
    "growth": [
        "ticker",
        "company_name",
        "revenue_growth_yoy",
        "revenue_growth_3y",
        "eps_growth_yoy",
        "eps_growth_3y",
        "sector",
        "industry",
    ],
    "quality": [
        "ticker",
        "company_name",
        "roe",
        "profit_margin",
        "debt_to_equity",
        "current_ratio",
        "sector",
        "industry",
    ],
    "momentum": [
        "ticker",
        "company_name",
        "price_change_1m",
        "price_change_3m",
        "price_change_6m",
        "price_change_1y",
        "rsi",
        "vs_52w_high",
        "vs_52w_low",
    ],
    "sentiment": [
        "ticker",
        "company_name",
        "recent_news_summary",
        "insider_activity",
        "analyst_rating",
        "short_interest",
    ],
}
