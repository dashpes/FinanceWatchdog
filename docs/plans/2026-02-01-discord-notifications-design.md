# Discord Notifications Design

**Date:** 2026-02-01
**Status:** Approved

## Overview

Add Discord as a notification channel for the Investment Monitor system, enabling daily and weekly reports to be delivered via Discord webhook with PDF attachments.

## Goals

- Send investment reports to Discord for mobile-friendly consumption
- Provide quick-glance embed summaries in Discord
- Attach comprehensive PDF reports for detailed reading
- Use Ollama (local LLM) for weekly synthesis to avoid API costs

## Message Flow

```
AlertMessage(s) → NotificationManager → DiscordChannel
                                              ↓
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                    Embed Summary                     PDF Attachment
                    (in Discord)                      (downloadable)
```

## Report Types

| Type | Embed Shows | PDF Contains |
|------|-------------|--------------|
| Daily | HIGH priority alerts + portfolio value change | Curated (MEDIUM+ priority only) |
| Weekly | AI synthesis (via Ollama) | Comprehensive (everything) |

## Components

### 1. DiscordChannel

**File:** `src/investment_monitor/notifications/discord.py`

Implements `NotificationChannel` base class.

```python
class DiscordChannel(NotificationChannel):
    name = "discord"

    def __init__(self, webhook_url: str) -> None: ...
    async def send(self, message: AlertMessage) -> bool: ...
    async def send_digest(self, messages: list[AlertMessage]) -> bool: ...
    async def _post_webhook(self, payload: dict, files: list | None = None) -> bool: ...
```

**Behavior:**
- Uses `httpx` to POST to Discord webhook
- Single alerts: sends embed with alert details (color-coded by type)
- Digests: sends embed summary + PDF attachment
- Rate limit safe (Discord allows 30 req/min, we'll use far fewer)

**Embed for single HIGH priority alert:**
```python
{
    "embeds": [{
        "title": "[AAPL] Price dropped 5.2%",
        "description": "Fell below 50-day moving average...",
        "color": 0xFF0000,  # Red for price drops
        "timestamp": "2026-01-15T14:30:00Z"
    }]
}
```

### 2. PDFReportGenerator

**File:** `src/investment_monitor/notifications/pdf_report.py`

Generates formatted PDF reports using `fpdf2`.

```python
class PDFReportGenerator:
    def generate_daily_report(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        date_value: date | None = None,
    ) -> bytes: ...

    def generate_weekly_report(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        week_start: date | None = None,
        week_end: date | None = None,
        ai_synthesis: str | None = None,
    ) -> bytes: ...
```

**Daily PDF Structure (Curated - MEDIUM+ only):**
```
┌─────────────────────────────────────┐
│  DAILY INVESTMENT REPORT            │
│  January 31, 2026                   │
├─────────────────────────────────────┤
│  PORTFOLIO SNAPSHOT                 │
│  Total Value: $XXX,XXX              │
│  Daily Change: +$X,XXX (+X.X%)      │
├─────────────────────────────────────┤
│  HIGH PRIORITY ALERTS               │
│  • [AAPL] Price dropped 5.2%        │
│  • [TSLA] Insider sale: $2.3M       │
├─────────────────────────────────────┤
│  MEDIUM PRIORITY ALERTS             │
│  • [GOOGL] Earnings in 3 days       │
│  • [MSFT] Volume spike 2.5x avg     │
└─────────────────────────────────────┘
```

**Weekly PDF Structure (Comprehensive - everything):**
```
┌─────────────────────────────────────┐
│  WEEKLY INVESTMENT REPORT           │
│  January 25 - 31, 2026              │
├─────────────────────────────────────┤
│  AI SYNTHESIS                       │
│  [Ollama-generated summary...]      │
├─────────────────────────────────────┤
│  PORTFOLIO PERFORMANCE              │
│  Weekly change, top/bottom movers   │
├─────────────────────────────────────┤
│  ALL ALERTS BY CATEGORY             │
│  Price | Volume | Insider | News... │
├─────────────────────────────────────┤
│  NEWS HIGHLIGHTS                    │
│  Relevant articles with scores      │
├─────────────────────────────────────┤
│  UPCOMING EARNINGS                  │
│  Next week's calendar               │
└─────────────────────────────────────┘
```

### 3. Ollama Weekly Synthesis

**File:** `src/investment_monitor/analysis/local_llm.py` (modify existing)

Add method for generating weekly narrative summaries.

```python
class LocalLLM:
    # ... existing methods ...

    async def generate_weekly_synthesis(
        self,
        alert_summary: dict,
        top_movers: list[tuple[str, float]],
        insider_activity: dict,
        upcoming_earnings: list[str],
    ) -> str: ...
```

**Prompt template:**
```
Week Summary:
- {n} price alerts ({high} HIGH, {medium} MEDIUM)
- {n} insider transactions ({buys} buys totaling ${buy_total}, {sells} sells totaling ${sell_total})
- {n} relevant news items
- Portfolio changed {pct}% (${amount})

Top movers: {movers}
Upcoming earnings: {earnings}

Generate a 2-3 sentence synthesis of this week's activity for an investor.
Focus on actionable insights and key trends.
```

**Graceful degradation:** If Ollama unavailable, return stats-only summary without AI narrative.

### 4. Discord Embed Formatter

**File:** `src/investment_monitor/notifications/discord.py` (helper functions)

```python
def format_daily_embed(
    high_priority_alerts: list[AlertMessage],
    portfolio_change: float | None,
    portfolio_change_pct: float | None,
) -> dict: ...

def format_weekly_embed(
    ai_synthesis: str,
) -> dict: ...

def format_alert_embed(
    message: AlertMessage,
) -> dict: ...
```

## Configuration

### Environment Variable

```bash
# .env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy
```

### Settings Class Update

**File:** `src/investment_monitor/config.py`

```python
class Settings(BaseSettings):
    # ... existing ...
    discord_webhook_url: str = ""  # Empty means disabled
```

### Notifications Config (New)

**File:** `config/notifications.yaml`

```yaml
discord:
  enabled: true

  # Daily digest settings
  daily_embed:
    show_high_priority_alerts: true
    show_portfolio_change: true

  # Weekly digest settings
  weekly_embed:
    show_ai_synthesis: true
```

## File Changes Summary

### New Files
- `src/investment_monitor/notifications/discord.py` - DiscordChannel implementation
- `src/investment_monitor/notifications/pdf_report.py` - PDF generation
- `config/notifications.yaml` - Notification settings

### Modified Files
- `src/investment_monitor/config.py` - Add discord_webhook_url setting
- `src/investment_monitor/analysis/local_llm.py` - Add generate_weekly_synthesis()
- `src/investment_monitor/main.py` - Wire up Discord channel
- `pyproject.toml` - Add fpdf2 dependency

## Dependencies

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "fpdf2>=2.7.0",  # PDF generation - lightweight, pure Python
]
```

## Discord Webhook Limits

- 30 requests per minute per webhook (we'll use far fewer)
- 2000 characters per message content
- 6000 characters total across embeds
- 10 embeds per message
- 8MB file upload limit (PDFs will be ~50-200KB)

All limits are well within our expected usage.

## Testing Plan

1. Unit tests for PDF generation with sample alert data
2. Unit tests for embed formatting
3. Integration test with mock webhook endpoint
4. Manual test with real Discord server

## Future Enhancements (Out of Scope)

- Interactive buttons (would require Discord bot, not just webhook)
- Message threading for related alerts
- User preferences per Discord user
