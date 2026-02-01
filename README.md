# Investment Monitor

A personal investment monitoring system that tracks your portfolio, collects market data, generates alerts based on configurable rules, and delivers daily/weekly digests.

## Features

- **Price Monitoring**: Track daily price movements, detect significant drops/rises
- **Volume Alerts**: Detect unusual trading volume spikes
- **Insider Tracking**: Monitor SEC Form 4 filings for insider buying/selling
- **News Aggregation**: Collect news from RSS feeds, filter by relevance
- **Earnings Calendar**: Get notified before earnings announcements
- **ETF Holdings**: Track changes in ETF compositions
- **AI Analysis**: Local LLM for news relevance scoring, Claude API for weekly synthesis
- **Flexible Notifications**: Console logging (Slack/email ready to add)

## Quick Start

### 1. Install

```bash
# Clone
git clone https://github.com/your-repo/investment-monitor.git
cd investment-monitor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all extras
pip install -e ".[dev,ai,notifications]"
```

### 2. Configure

```bash
# Copy example configs
cp config/portfolio.yaml.example config/portfolio.yaml
cp config/alerts.yaml.example config/alerts.yaml

# Edit your portfolio
nano config/portfolio.yaml
```

**portfolio.yaml example:**
```yaml
holdings:
  - ticker: AAPL
    shares: 50
    cost_basis: 165.00
    thesis: "Services growth driving margin expansion"

  - ticker: MSFT
    shares: 30
    cost_basis: 380.00
    thesis: "Azure cloud growth, AI integration"

watchlist:
  - ticker: GOOGL
    reason: "Waiting for better entry"
    target_price: 140.00
```

### 3. Run

```bash
# Regular monitoring (collect data + immediate alerts)
investment-monitor --type regular

# Daily digest
investment-monitor --type digest

# Weekly AI synthesis (requires ANTHROPIC_API_KEY)
investment-monitor --type weekly

# Dry run (show what would happen)
investment-monitor --dry-run
```

## API Requirements & Estimated Costs

### Required APIs (Free)

| API/Service | Purpose | Auth Required | Rate Limits |
|-------------|---------|---------------|-------------|
| **Yahoo Finance (yfinance)** | Stock prices, earnings calendar, fundamentals | No | ~30 req/min |
| **SEC EDGAR** | Insider transactions (Form 4 filings) | No | 10 req/sec |
| **Congressional Trading** | Congressional stock disclosures | No | Public S3 data |
| **RSS Feeds** | News aggregation (Yahoo Finance, Seeking Alpha) | No | ~30 req/min |

### Optional APIs

| API/Service | Purpose | Free Tier | Paid Pricing |
|-------------|---------|-----------|--------------|
| **Anthropic Claude** | Weekly AI synthesis & research reports | None | See below |
| **Ollama (Local)** | News relevance scoring, sentiment analysis | Unlimited (self-hosted) | Free |
| **SendGrid** | Email notifications | 100 emails/day | $19.95/mo for 50K emails |
| **Slack Webhooks** | Slack notifications | Unlimited | Free |
| **Finnhub** | Financial data (configured but unused) | 60 req/min | Varies |

### Claude API Pricing (Anthropic)

The weekly synthesis feature uses Claude Sonnet. Current pricing (2026):

| Model | Input (per M tokens) | Output (per M tokens) |
|-------|---------------------|----------------------|
| Claude Haiku 4.5 | $1.00 | $5.00 |
| **Claude Sonnet 4.5** | $3.00 | $15.00 |
| Claude Opus 4.5 | $5.00 | $25.00 |

**Estimated monthly cost**: $1-5/month for typical usage (1-2 weekly syntheses). The codebase includes a configurable monthly spending limit (default: $5.00).

### Cost Optimization Tips

1. **Use Ollama for local AI** - Free news scoring and sentiment analysis
2. **Batch API requests** - 50% discount on Claude API for async processing
3. **Prompt caching** - 90% savings on repeated context with Claude
4. **Console notifications only** - Skip SendGrid/Slack for zero notification costs

### Total Estimated Monthly Costs

| Usage Level | Description | Est. Cost |
|-------------|-------------|-----------|
| **Free** | yfinance + SEC + console logging + Ollama | $0/mo |
| **Basic AI** | Free + Claude weekly synthesis | $1-5/mo |
| **Full Featured** | Basic AI + SendGrid email notifications | $20-25/mo |

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Optional: Claude API for weekly synthesis
ANTHROPIC_API_KEY=sk-ant-xxx

# Optional: Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/xxx

# Optional: Email notifications
SENDGRID_API_KEY=SG.xxx

# Ollama endpoint (default: localhost)
OLLAMA_HOST=http://localhost:11434
```

### Alert Thresholds

Edit `config/alerts.yaml`:

```yaml
price:
  enabled: true
  daily_drop_pct: 3.0      # Alert if drops > 3%
  daily_rise_pct: 5.0      # Alert if rises > 5%
  weekly_drop_pct: 7.0     # Alert if weekly drop > 7%
  below_cost_basis: true   # Alert if below your cost basis

volume:
  enabled: true
  lookback_days: 20
  multiplier: 2.5          # Alert if volume > 2.5x average

insider:
  enabled: true
  min_buy_value: 100000    # Minimum insider buy to alert
  min_sell_value: 500000   # Minimum insider sell to alert
  alert_ceo_cfo_any: true  # Alert any CEO/CFO transaction

earnings:
  enabled: true
  lookahead_days: 7        # Alert N days before earnings

news:
  enabled: true
  min_relevance_score: 5.0 # 0-10 scale (requires Ollama)
```

## Deployment

### Option A: Systemd (Linux)

```bash
# Copy to /opt
sudo cp -r . /opt/investment-monitor
cd /opt/investment-monitor

# Setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ai,notifications]"

# Install systemd services
sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/

# Edit services to set your username
sudo nano /etc/systemd/system/investment-monitor.service

# Enable timers
sudo systemctl daemon-reload
sudo systemctl enable --now investment-monitor.timer
sudo systemctl enable --now investment-digest.timer
sudo systemctl enable --now investment-weekly.timer

# Check status
systemctl list-timers | grep investment
journalctl -u investment-monitor -f
```

### Option B: Docker

```bash
# Start all services (monitor + ollama + cron)
docker-compose up -d

# View logs
docker-compose logs -f

# Run manually
docker-compose exec monitor investment-monitor --type regular
```

### Schedule Overview

| Schedule | Command | Purpose |
|----------|---------|---------|
| Every 4h (weekdays) | `--type regular` | Collect data, send urgent alerts |
| Daily 7am | `--type digest` | Daily summary email |
| Sunday 6pm | `--type weekly` | AI-powered weekly synthesis |

## Local LLM Setup (Ollama)

For AI-powered news relevance scoring:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model
ollama pull phi3:mini

# Start service
sudo systemctl enable --now ollama
```

The system gracefully degrades if Ollama isn't available.

## Project Structure

```
investment-monitor/
├── src/investment_monitor/
│   ├── config.py           # Configuration loading
│   ├── main.py              # Main orchestrator
│   ├── cli.py               # Command-line interface
│   ├── models/              # Pydantic models
│   │   ├── portfolio.py     # Portfolio/Holding models
│   │   └── alerts.py        # Alert config models
│   ├── storage/             # Database layer
│   │   ├── database.py      # SQLite/SQLAlchemy setup
│   │   ├── models.py        # ORM models
│   │   └── operations.py    # CRUD operations
│   ├── collectors/          # Data collectors
│   │   ├── base.py          # Base with rate limiting
│   │   ├── prices.py        # yfinance prices
│   │   ├── insider.py       # SEC Form 4
│   │   ├── news.py          # RSS feeds
│   │   ├── earnings.py      # Earnings calendar
│   │   └── etf_holdings.py  # ETF compositions
│   ├── alerts/              # Alert engine
│   │   ├── engine.py        # Main alert processor
│   │   ├── rules.py         # Alert rule implementations
│   │   ├── priority.py      # Priority classification
│   │   └── dedup.py         # Deduplication
│   ├── notifications/       # Notification system
│   │   ├── base.py          # Abstract channel
│   │   ├── console.py       # Console/log output
│   │   ├── manager.py       # Routing by priority
│   │   └── digest.py        # Digest formatting
│   └── analysis/            # AI integration
│       ├── local_llm.py     # Ollama client
│       ├── news_processor.py # News relevance scoring
│       └── claude_api.py    # Weekly synthesis
├── config/                  # Configuration files
├── data/                    # SQLite database
├── logs/                    # Log files
├── tests/                   # Test suite (534 tests)
├── systemd/                 # Systemd service files
├── Dockerfile
└── docker-compose.yaml
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test file
pytest tests/test_alerts.py -v

# Run with coverage
pytest --cov=investment_monitor tests/
```

## Adding Slack/Email Notifications

The notification system is designed for easy extension. To add Slack:

1. Create `src/investment_monitor/notifications/slack.py`
2. Implement the `NotificationChannel` interface
3. Add to `NotificationManager` in `main.py`

Example:
```python
from slack_sdk.webhook import WebhookClient
from .base import NotificationChannel, AlertMessage

class SlackChannel(NotificationChannel):
    name = "slack"

    def __init__(self, webhook_url: str):
        self.client = WebhookClient(webhook_url)

    async def send(self, message: AlertMessage) -> bool:
        response = self.client.send(text=f"*{message.title}*\n{message.body}")
        return response.status_code == 200
```

## License

MIT
