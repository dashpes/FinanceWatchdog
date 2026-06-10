# Analysis Module

AI-powered analysis that runs **completely free on local Ollama by default**,
for both tiers (news scoring *and* weekly synthesis / research reports). Claude
is an optional, opt-in provider for higher-quality tier-2 output.

## Overview

```
analysis/
├── hardware.py        # Cross-platform RAM detection + recommend_models()
├── ollama_client.py   # Object-based Ollama response helpers (single source)
├── prompts.py         # Prompt templates
├── local_llm.py       # LocalLLM - Ollama client (scoring, sentiment, synthesis)
├── news_processor.py  # NewsProcessor - relevance scoring
├── research_scorer.py # ResearchScorer - multi-factor stock scoring (Ollama)
├── research_report.py # ResearchReportGenerator - reports (Ollama or Claude)
└── claude_api.py      # ClaudeAnalyzer - optional weekly synthesis via Claude
```

### Provider & model selection

- **Tier-1 (fast):** news relevance, sentiment, factor scoring — always local.
- **Tier-2 (heavy):** weekly synthesis, research reports — local by default;
  Claude when `LLM_PROVIDER=anthropic`, or `auto` with an `ANTHROPIC_API_KEY`.
- **Model agnostic:** `OLLAMA_MODEL` / `OLLAMA_SYNTHESIS_MODEL` default to `auto`,
  which picks models by detected RAM (`hardware.recommend_models`). Pin an
  explicit tag to override. Run `investment-monitor --doctor` to see the
  detected RAM, chosen models, and live Ollama status.

### Modern Ollama API

This module targets the object-based Ollama Python API (>= 0.4): `client.list()`
returns a `ListResponse` (`.models[i].model`) and `client.generate()` returns a
`GenerateResponse` (`.response`). All response parsing goes through
`ollama_client.py` so there is exactly one place that knows the response shape.

## Local LLM (Ollama)

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi3:mini

# Start server (if not using systemd)
ollama serve
```

### Usage

```python
from investment_monitor.analysis import LocalLLM

llm = LocalLLM(
    model="phi3:mini",
    base_url="http://localhost:11434"
)

# Check availability
if llm.is_available():
    print("Ollama ready")

# Score news relevance (0-10)
score = await llm.score_relevance(
    headline="Apple announces new iPhone",
    ticker="AAPL",
    company_name="Apple Inc."
)
# Returns: 8.0

# Classify sentiment
sentiment = await llm.classify_sentiment(
    "Apple stock surges on strong earnings"
)
# Returns: "bullish", "bearish", or "neutral"

# Summarize text
summary = await llm.summarize(
    "Long article text...",
    max_sentences=2
)
```

### Graceful Degradation

All methods handle Ollama being unavailable:

```python
# If Ollama is down:
score = await llm.score_relevance(...)  # Returns -1.0
sentiment = await llm.classify_sentiment(...)  # Returns "unknown"
summary = await llm.summarize(...)  # Returns ""
```

### Prompt Templates

```python
from investment_monitor.analysis import RELEVANCE_PROMPT, SENTIMENT_PROMPT

# Customize prompts in prompts.py
RELEVANCE_PROMPT = """You are a financial news filter...
Ticker: {ticker}
Company: {company_name}
Headline: {headline}
Rating:"""
```

## News Processor

Batch processes news items for relevance scoring.

```python
from investment_monitor.analysis import NewsProcessor, LocalLLM

llm = LocalLLM()
processor = NewsProcessor(
    session=session,
    llm=llm,
    portfolio=portfolio,
    min_relevance=5.0  # Filter threshold
)

# Score all unscored news
processed_count = await processor.process_unscored_news(batch_size=100)
print(f"Scored {processed_count} items")

# Get relevant news (score >= min_relevance)
relevant = await processor.get_relevant_news(
    ticker="AAPL",  # Optional filter
    hours=24
)

# Get news sorted by relevance
top_news = await processor.get_news_by_priority(
    ticker=None,
    hours=48,
    min_score=7.0
)

# Check queue
pending = processor.get_unscored_count()
```

### Scoring Logic

1. Fetches unscored items from database
2. For each item, calls `llm.score_relevance()`
3. Updates `relevance_score` in database
4. Skips if LLM unavailable (score = -1)

### Investment Thesis Context

If a holding has a thesis, it's included in scoring:

```python
# portfolio.yaml
holdings:
  - ticker: AAPL
    thesis: "Services growth driving margin expansion"

# News about Apple Services will score higher
# because it relates to the thesis
```

## Claude API

Weekly portfolio synthesis using Claude.

```python
from investment_monitor.analysis import ClaudeAnalyzer, WeeklyData

analyzer = ClaudeAnalyzer(
    api_key="sk-ant-xxx",
    max_monthly_spend=5.00  # Budget limit
)

# Check availability
if analyzer.is_available():
    print("Claude API ready")

# Generate weekly synthesis
result = await analyzer.weekly_synthesis(
    portfolio=portfolio,
    week_data=WeeklyData(
        price_summary="AAPL -3%, MSFT +2%",
        insider_summary="Tim Cook sold 50k shares",
        news_summary="Apple AI announcement...",
        earnings_summary="AAPL reports Feb 2"
    )
)

if result.success:
    print(result.synthesis)
else:
    print(f"Error: {result.error_message}")

# Check spend
print(f"Month-to-date: ${analyzer.get_monthly_spend():.2f}")
```

### WeeklyData Model

```python
from investment_monitor.analysis import WeeklyData

data = WeeklyData(
    price_summary="...",      # Required
    insider_summary="...",    # Required
    news_summary="...",       # Required
    earnings_summary="..."    # Required
)
```

### Budget Management

```python
# Cost tracking
analyzer = ClaudeAnalyzer(api_key="...", max_monthly_spend=5.00)

# Automatic budget check before API call
result = await analyzer.weekly_synthesis(...)
# If over budget: result.success=False, result.error_message="budget limit reached"

# Check remaining budget
remaining = analyzer.max_monthly_spend - analyzer.get_monthly_spend()

# Reset happens automatically on new month
```

### Cost Estimation

Approximate costs (Claude Sonnet):
- Input: ~$3 per million tokens
- Output: ~$15 per million tokens
- Weekly synthesis: ~$0.02-0.05 per call

### Prompt Template

The synthesis prompt requests:
1. 2-3 most important developments
2. Thesis contradictions
3. Positions needing attention
4. Notable patterns

Customize in `claude_api.py`:

```python
WEEKLY_SYNTHESIS_PROMPT = """You are my personal investment analyst...
## My Portfolio
{portfolio_yaml}
## This Week's Activity
...
"""
```

## Troubleshooting

### Ollama Not Responding

```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart service
sudo systemctl restart ollama

# Check logs
journalctl -u ollama -f
```

### Model Not Loaded

```bash
# List models
ollama list

# Pull model
ollama pull phi3:mini

# Try different model
llm = LocalLLM(model="llama3:8b")
```

### Slow Scoring

1. Use smaller model (`phi3:mini` vs `llama3:70b`)
2. Reduce batch size:
   ```python
   await processor.process_unscored_news(batch_size=20)
   ```
3. Check GPU availability for Ollama

### Claude API Errors

```python
# Rate limit
result.error_message = "API error: rate_limit_exceeded"
# Solution: Wait and retry

# Invalid key
result.error_message = "API error: authentication_error"
# Solution: Check ANTHROPIC_API_KEY

# Over budget
result.error_message = "budget limit reached"
# Solution: Increase max_monthly_spend or wait for reset
```

### Memory Issues with Ollama

```bash
# Check memory usage
ollama ps

# Unload unused models
ollama stop phi3:mini

# Use smaller model
ollama pull phi3:mini  # 2GB vs llama3:70b at 40GB
```

## Extending

### Add New Analysis Function

```python
# In local_llm.py
async def extract_entities(self, text: str) -> list[str]:
    """Extract company names from text."""
    if not self.is_available():
        return []

    prompt = f"""Extract company names from this text:
    {text}

    List only company names, one per line:"""

    response = self.client.generate(
        model=self.model,
        prompt=prompt,
        options={"temperature": 0.1}
    )

    # Object-based API: read .response (helpers in ollama_client.py).
    from .ollama_client import response_text
    return [line.strip() for line in response_text(response).split("\n") if line.strip()]
```

### Add Alternative LLM Provider

```python
# analysis/openai_llm.py
from openai import OpenAI

class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def score_relevance(self, headline: str, ticker: str, company_name: str) -> float:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": RELEVANCE_PROMPT.format(...)}],
            max_tokens=10
        )
        return self._parse_score(response.choices[0].message.content)
```

Then use in `NewsProcessor`:

```python
# Can swap LLM implementations
processor = NewsProcessor(session, openai_llm, portfolio)
```
