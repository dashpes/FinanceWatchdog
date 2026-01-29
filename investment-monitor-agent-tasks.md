# Investment Monitor: Agent Task Breakdown

## How to Use This Document

This document contains discrete, self-contained tasks for building a personal investment monitoring system. Each task is designed to be completed independently by an AI coding agent.

**Task Format:**
- **Context**: Background needed to understand the task
- **Requirements**: What must be built
- **Inputs/Outputs**: Data flow expectations
- **Files to Create**: Expected deliverables
- **Acceptance Criteria**: How to verify completion
- **Dependencies**: Other tasks that must be completed first

**Conventions:**
- Python 3.11+
- Use type hints throughout
- Use `loguru` for logging
- Use `pydantic` for configuration and data validation
- SQLite for database
- All paths relative to project root

---

## Task Dependency Graph

```
PHASE 1: Foundation
[T01] ─── [T02] ─┬─ [T03]
                 ├─ [T04]
                 └─ [T05] ─── [T06]

PHASE 2: Data Collectors
[T07] ─┬─ [T08]
       ├─ [T09]
       ├─ [T10]
       ├─ [T11]
       └─ [T12]

PHASE 3: Alert Engine
[T08-T12] ─── [T13] ─── [T14] ─── [T15]

PHASE 4: AI Integration
[T15] ─── [T16] ─── [T17] ─── [T18]

PHASE 5: Orchestration
[T18] ─── [T19] ─── [T20]
```

---

# PHASE 1: Foundation

---

## Task T01: Project Scaffolding

### Context
Create the initial project structure with proper Python packaging, configuration management, and development tooling.

### Requirements
1. Create directory structure matching the specification below
2. Set up `pyproject.toml` with all dependencies
3. Create base configuration loading from YAML files
4. Set up logging configuration

### Files to Create

```
investment-monitor/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── portfolio.yaml.example
│   ├── alerts.yaml.example
│   ├── sources.yaml.example
│   └── notifications.yaml.example
├── src/
│   └── investment_monitor/
│       ├── __init__.py
│       ├── config.py
│       └── logging_config.py
├── tests/
│   └── __init__.py
└── data/
    └── .gitkeep
```

### pyproject.toml Dependencies

```toml
[project]
name = "investment-monitor"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "yfinance>=0.2.0",
    "feedparser>=6.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
    "pandas>=2.0.0",
    "loguru>=0.7.0",
    "httpx>=0.25.0",
]

[project.optional-dependencies]
notifications = [
    "sendgrid>=6.10.0",
    "slack-sdk>=3.21.0",
]
ai = [
    "ollama>=0.1.0",
    "anthropic>=0.18.0",
]
dashboard = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "jinja2>=3.1.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
```

### config.py Requirements

```python
# Must implement:
# - Pydantic models for all configuration types
# - Load from YAML files in config/ directory
# - Support environment variable overrides
# - Validate on load

class PortfolioConfig(BaseModel):
    holdings: list[Holding]
    watchlist: list[WatchlistItem]

class AlertsConfig(BaseModel):
    price_alerts: PriceAlertSettings
    volume_alerts: VolumeAlertSettings
    insider_alerts: InsiderAlertSettings
    earnings_alerts: EarningsAlertSettings
    news_alerts: NewsAlertSettings

class Settings(BaseSettings):
    portfolio: PortfolioConfig
    alerts: AlertsConfig
    # ... etc
```

### Acceptance Criteria
- [ ] `pip install -e .` succeeds
- [ ] `from investment_monitor.config import Settings` works
- [ ] Configuration loads from YAML files
- [ ] Missing required config raises clear error
- [ ] Logging writes to both console and `logs/monitor.log`

### Dependencies
None (first task)

---

## Task T02: Database Schema and Models

### Context
Create SQLite database schema for storing historical data, alert history, and application state. Use SQLAlchemy ORM.

### Requirements
1. Define SQLAlchemy models for all data types
2. Create database initialization function
3. Implement basic CRUD operations
4. Support migrations via Alembic (optional but preferred)

### Database Tables

**prices**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol |
| date | DATE | Trading date |
| open | REAL | Opening price |
| high | REAL | Daily high |
| low | REAL | Daily low |
| close | REAL | Closing price |
| volume | INTEGER | Trading volume |
| created_at | TIMESTAMP | Record creation time |

**insider_transactions**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol |
| filing_date | DATE | SEC filing date |
| trade_date | DATE | Actual trade date |
| owner_name | TEXT | Insider name |
| owner_title | TEXT | CEO, CFO, Director, etc. |
| transaction_type | TEXT | P (purchase), S (sale), etc. |
| shares | INTEGER | Number of shares |
| price_per_share | REAL | Transaction price |
| total_value | REAL | shares * price |
| sec_url | TEXT | Link to SEC filing |
| created_at | TIMESTAMP | Record creation time |

**news_items**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Related stock (nullable) |
| headline | TEXT | Article headline |
| source | TEXT | News source |
| url | TEXT | Article URL (unique) |
| published_at | TIMESTAMP | Publication time |
| relevance_score | REAL | AI-assigned score (nullable) |
| sentiment | TEXT | bullish/bearish/neutral (nullable) |
| created_at | TIMESTAMP | Record creation time |

**alerts_sent**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| alert_type | TEXT | price/insider/news/earnings |
| ticker | TEXT | Related stock |
| message | TEXT | Alert content |
| priority | TEXT | high/medium/low |
| sent_at | TIMESTAMP | When alert was sent |
| channel | TEXT | email/slack/pushover |

**etf_holdings**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| etf_ticker | TEXT | ETF symbol |
| holding_ticker | TEXT | Held stock symbol |
| shares | REAL | Number of shares held |
| weight_pct | REAL | Percentage of ETF |
| as_of_date | DATE | Holdings date |
| created_at | TIMESTAMP | Record creation time |

### Files to Create

```
src/investment_monitor/
├── storage/
│   ├── __init__.py
│   ├── database.py      # Engine, session management
│   ├── models.py        # SQLAlchemy ORM models
│   └── operations.py    # CRUD functions
```

### Key Functions

```python
# database.py
def init_db(db_path: str = "data/portfolio.db") -> Engine: ...
def get_session() -> Generator[Session, None, None]: ...

# operations.py
def save_prices(session: Session, prices: list[PriceRecord]) -> int: ...
def get_latest_price(session: Session, ticker: str) -> PriceRecord | None: ...
def save_insider_transaction(session: Session, txn: InsiderTransaction) -> int: ...
def get_insider_transactions(session: Session, ticker: str, days: int = 30) -> list[InsiderTransaction]: ...
def save_news_item(session: Session, item: NewsItem) -> int: ...
def news_exists(session: Session, url: str) -> bool: ...  # For deduplication
def save_alert(session: Session, alert: AlertRecord) -> int: ...
def get_recent_alerts(session: Session, hours: int = 24) -> list[AlertRecord]: ...
```

### Acceptance Criteria
- [ ] `init_db()` creates database file with all tables
- [ ] All CRUD operations work correctly
- [ ] Unique constraints prevent duplicate news items
- [ ] Foreign keys and indexes are properly defined
- [ ] Test file demonstrates all operations

### Dependencies
- T01 (Project Scaffolding)

---

## Task T03: Portfolio Configuration Models

### Context
Define the data structures for portfolio holdings, watchlist, and investment theses. These will be used throughout the system.

### Requirements
1. Pydantic models for portfolio configuration
2. Support loading from YAML
3. Computed properties for portfolio analysis
4. Validation rules

### Files to Create

```
src/investment_monitor/
├── models/
│   ├── __init__.py
│   └── portfolio.py
config/
└── portfolio.yaml.example
```

### Pydantic Models

```python
from pydantic import BaseModel, Field, computed_field
from decimal import Decimal

class Holding(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    shares: Decimal = Field(..., gt=0)
    cost_basis: Decimal = Field(..., gt=0)
    thesis: str = Field(default="", max_length=500)
    
    @computed_field
    @property
    def total_cost(self) -> Decimal:
        return self.shares * self.cost_basis

class WatchlistItem(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    reason: str = Field(default="")
    target_price: Decimal | None = None

class Portfolio(BaseModel):
    holdings: list[Holding] = Field(default_factory=list)
    watchlist: list[WatchlistItem] = Field(default_factory=list)
    
    @computed_field
    @property
    def all_tickers(self) -> list[str]:
        """All tickers to monitor (holdings + watchlist)"""
        return list(set(
            [h.ticker for h in self.holdings] + 
            [w.ticker for w in self.watchlist]
        ))
    
    @computed_field
    @property
    def holding_tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]
    
    def get_holding(self, ticker: str) -> Holding | None:
        """Get holding by ticker"""
        ...
    
    def get_thesis(self, ticker: str) -> str | None:
        """Get investment thesis for a ticker"""
        ...
```

### Example portfolio.yaml

```yaml
holdings:
  - ticker: AAPL
    shares: 50
    cost_basis: 165.00
    thesis: "Services growth driving margin expansion, aggressive buybacks reducing share count"
    
  - ticker: MSFT
    shares: 30
    cost_basis: 380.00
    thesis: "Azure cloud growth, AI integration across Office suite, gaming division"
    
  - ticker: VTI
    shares: 100
    cost_basis: 220.00
    thesis: "Core US total market exposure, low expense ratio"

watchlist:
  - ticker: GOOGL
    reason: "Waiting for better entry, concerns about AI competition"
    target_price: 140.00
    
  - ticker: ASML
    reason: "Semiconductor equipment monopoly, waiting for cyclical bottom"
```

### Acceptance Criteria
- [ ] Portfolio loads from YAML without errors
- [ ] Invalid tickers (lowercase, too long) raise validation errors
- [ ] `all_tickers` returns deduplicated list
- [ ] `get_thesis()` returns thesis or None for watchlist items
- [ ] Computed properties work correctly

### Dependencies
- T01 (Project Scaffolding)

---

## Task T04: Alert Configuration Models

### Context
Define configuration structures for all alert types with sensible defaults and validation.

### Requirements
1. Pydantic models for each alert category
2. Sensible defaults for all thresholds
3. Support enabling/disabling individual alert types
4. Load from YAML

### Files to Create

```
src/investment_monitor/
├── models/
│   └── alerts.py
config/
└── alerts.yaml.example
```

### Models

```python
class PriceAlertSettings(BaseModel):
    enabled: bool = True
    daily_drop_pct: float = Field(default=3.0, ge=0, le=100)
    daily_rise_pct: float = Field(default=5.0, ge=0, le=100)
    weekly_drop_pct: float = Field(default=7.0, ge=0, le=100)
    below_cost_basis: bool = True

class VolumeAlertSettings(BaseModel):
    enabled: bool = True
    lookback_days: int = Field(default=20, ge=5, le=60)
    multiplier: float = Field(default=2.5, ge=1.0)

class InsiderAlertSettings(BaseModel):
    enabled: bool = True
    min_buy_value: int = Field(default=100_000, ge=0)
    min_sell_value: int = Field(default=500_000, ge=0)
    alert_ceo_cfo_any: bool = True
    cluster_threshold: int = Field(default=3, ge=2)
    cluster_days: int = Field(default=7, ge=1)

class EarningsAlertSettings(BaseModel):
    enabled: bool = True
    lookahead_days: int = Field(default=7, ge=1, le=30)

class NewsAlertSettings(BaseModel):
    enabled: bool = True
    keywords: list[str] = Field(default_factory=lambda: [
        "lawsuit", "SEC", "investigation", "guidance", 
        "acquisition", "merger", "layoffs", "dividend", "buyback"
    ])
    min_relevance_score: float = Field(default=5.0, ge=0, le=10)

class ETFAlertSettings(BaseModel):
    enabled: bool = True
    holdings_change: bool = True
    expense_ratio_change: bool = True

class AlertsConfig(BaseModel):
    price: PriceAlertSettings = Field(default_factory=PriceAlertSettings)
    volume: VolumeAlertSettings = Field(default_factory=VolumeAlertSettings)
    insider: InsiderAlertSettings = Field(default_factory=InsiderAlertSettings)
    earnings: EarningsAlertSettings = Field(default_factory=EarningsAlertSettings)
    news: NewsAlertSettings = Field(default_factory=NewsAlertSettings)
    etf: ETFAlertSettings = Field(default_factory=ETFAlertSettings)
```

### Acceptance Criteria
- [ ] All models have sensible defaults
- [ ] Invalid values (negative percentages, etc.) raise validation errors
- [ ] Can selectively disable alert types via `enabled: false`
- [ ] Loads from YAML and merges with defaults

### Dependencies
- T01 (Project Scaffolding)

---

## Task T05: Notification System Base

### Context
Create the notification infrastructure that alert handlers will use to send messages. Support multiple channels.

### Requirements
1. Abstract base class for notification channels
2. Email implementation (SMTP or SendGrid)
3. Slack webhook implementation
4. Message priority levels (high = immediate, medium = digest, low = log only)
5. Configuration from environment variables

### Files to Create

```
src/investment_monitor/
├── notifications/
│   ├── __init__.py
│   ├── base.py          # Abstract base, message models
│   ├── email.py         # Email sender
│   ├── slack.py         # Slack webhook
│   └── manager.py       # Routes messages to appropriate channels
config/
└── notifications.yaml.example
```

### Base Classes

```python
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel

class Priority(str, Enum):
    HIGH = "high"      # Send immediately
    MEDIUM = "medium"  # Include in next digest
    LOW = "low"        # Log only

class AlertMessage(BaseModel):
    title: str
    body: str
    ticker: str | None = None
    alert_type: str
    priority: Priority = Priority.MEDIUM
    url: str | None = None  # Link for more info

class NotificationChannel(ABC):
    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """Send a message. Returns True if successful."""
        ...
    
    @abstractmethod
    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Send a batch of messages as a digest."""
        ...

class NotificationManager:
    """Routes messages to appropriate channels based on priority and config"""
    
    def __init__(self, channels: list[NotificationChannel]):
        ...
    
    async def notify(self, message: AlertMessage) -> None:
        """Send notification via configured channels"""
        ...
    
    async def send_daily_digest(self, messages: list[AlertMessage]) -> None:
        """Compile and send daily digest"""
        ...
```

### Email Implementation Notes

Support both raw SMTP and SendGrid:

```python
class EmailChannel(NotificationChannel):
    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        sendgrid_api_key: str | None = None,
        from_address: str = "",
        to_address: str = "",
    ):
        # Use SendGrid if API key provided, else SMTP
        ...
```

### Slack Implementation Notes

```python
class SlackChannel(NotificationChannel):
    def __init__(self, webhook_url: str):
        ...
    
    def _format_message(self, message: AlertMessage) -> dict:
        """Format as Slack Block Kit message"""
        ...
```

### Acceptance Criteria
- [ ] Email sends via SMTP successfully
- [ ] Email sends via SendGrid successfully (if API key provided)
- [ ] Slack webhook sends formatted messages
- [ ] NotificationManager routes based on priority
- [ ] Digest formatter creates readable summary
- [ ] Graceful handling of send failures (log, don't crash)

### Dependencies
- T01 (Project Scaffolding)

---

## Task T06: Digest Formatter

### Context
Create formatted digests for daily/weekly email summaries. Should be readable and scannable.

### Requirements
1. Plain text format (for email compatibility)
2. Optional HTML format
3. Group alerts by type and ticker
4. Include summary statistics

### Files to Create

```
src/investment_monitor/
├── notifications/
│   └── digest.py
```

### Functions

```python
def format_daily_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio,
    date: date,
) -> tuple[str, str]:
    """
    Format messages into a daily digest.
    
    Returns:
        tuple of (plain_text, html)
    """
    ...

def format_weekly_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio,
    week_start: date,
    week_end: date,
    ai_synthesis: str | None = None,
) -> tuple[str, str]:
    """
    Format messages into weekly digest with optional AI synthesis.
    """
    ...
```

### Example Output (Plain Text)

```
═══════════════════════════════════════════════════════════════
INVESTMENT MONITOR DAILY DIGEST
January 28, 2026
═══════════════════════════════════════════════════════════════

SUMMARY
───────
• 2 price alerts
• 1 insider transaction
• 3 relevant news items
• 1 earnings upcoming

PRICE MOVEMENTS
───────────────
▼ AAPL: -3.2% ($182.50 → $176.66)
  Below your cost basis of $165.00: No
  
▲ MSFT: +2.1% ($415.00 → $423.72)

INSIDER ACTIVITY
────────────────
AAPL: Tim Cook (CEO) sold 50,000 shares @ $180.00
      Total value: $9,000,000
      Note: Likely pre-planned 10b5-1 sale
      Filing: https://sec.gov/...

NEWS
────
[AAPL] "Apple Announces New AI Features for iPhone 17"
       Source: Reuters | Relevance: 8/10
       
[MSFT] "Microsoft Azure Revenue Grows 29% YoY"
       Source: Bloomberg | Relevance: 9/10

UPCOMING EARNINGS
─────────────────
• AAPL reports in 5 days (Feb 2)
• MSFT reports in 12 days (Feb 9)

───────────────────────────────────────────────────────────────
Generated by Investment Monitor | Manage settings: [link]
```

### Acceptance Criteria
- [ ] Plain text digest is readable in any email client
- [ ] HTML digest renders correctly in Gmail/Outlook
- [ ] Alerts grouped logically by type
- [ ] Price changes show direction arrows
- [ ] Includes links where available
- [ ] Handles empty sections gracefully

### Dependencies
- T05 (Notification System Base)

---

# PHASE 2: Data Collectors

---

## Task T07: Collector Base Class

### Context
Create a standardized interface for all data collectors to ensure consistency and enable easy orchestration.

### Requirements
1. Abstract base class with standard interface
2. Built-in rate limiting
3. Error handling and retry logic
4. Logging integration

### Files to Create

```
src/investment_monitor/
├── collectors/
│   ├── __init__.py
│   └── base.py
```

### Base Class

```python
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

class CollectorResult(BaseModel):
    collector_name: str
    success: bool
    records_collected: int
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

class BaseCollector(ABC):
    name: str
    rate_limit_calls: int = 60      # Calls per minute
    rate_limit_period: int = 60     # Period in seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __init__(self, session: Session, config: Settings):
        self.session = session
        self.config = config
        self._call_times: list[float] = []
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting"""
        ...
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry failed requests with exponential backoff"""
        ...
    
    @abstractmethod
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Collect data for given tickers.
        Must be implemented by subclasses.
        """
        ...
    
    @abstractmethod
    async def collect_single(self, ticker: str) -> int:
        """
        Collect data for a single ticker.
        Returns number of records saved.
        """
        ...
```

### Acceptance Criteria
- [ ] Rate limiting prevents exceeding API limits
- [ ] Retries work with exponential backoff
- [ ] Results include success/failure status and timing
- [ ] Errors are logged but don't crash the collector
- [ ] Can be subclassed easily

### Dependencies
- T01 (Project Scaffolding)
- T02 (Database Schema)

---

## Task T08: Price Collector

### Context
Collect daily price and volume data using yfinance. This is the foundation for price-based alerts.

### Requirements
1. Fetch daily OHLCV data
2. Store in database, avoid duplicates
3. Support batch fetching for efficiency
4. Calculate basic derived metrics (daily change %)

### Files to Create

```
src/investment_monitor/
├── collectors/
│   └── prices.py
```

### Implementation Notes

```python
import yfinance as yf

class PriceCollector(BaseCollector):
    name = "prices"
    
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Fetch prices for all tickers.
        yfinance supports batch requests efficiently.
        """
        # Use yf.download() for batch fetching
        # Parse results and save to database
        ...
    
    async def collect_single(self, ticker: str) -> int:
        """Fetch price for single ticker"""
        ...
    
    def get_price_with_change(self, ticker: str) -> dict | None:
        """
        Get latest price with daily/weekly change calculations.
        Returns:
            {
                "ticker": "AAPL",
                "price": 176.66,
                "daily_change_pct": -3.2,
                "weekly_change_pct": -5.1,
                "volume": 82_000_000,
                "avg_volume_20d": 65_000_000,
            }
        """
        ...
```

### Acceptance Criteria
- [ ] Fetches prices for all portfolio tickers
- [ ] Stores in database without duplicates
- [ ] Handles market holidays (no data) gracefully
- [ ] Calculates daily and weekly percent changes
- [ ] Calculates volume vs 20-day average
- [ ] Works with both stocks and ETFs

### Dependencies
- T07 (Collector Base Class)

---

## Task T09: Insider Transaction Collector

### Context
Fetch Form 4 filings from SEC EDGAR to track insider buying and selling. This is valuable signal for individual stocks.

### Requirements
1. Fetch recent Form 4 filings for tracked tickers
2. Parse XML to extract transaction details
3. Store in database with deduplication
4. Handle multiple transactions per filing

### Files to Create

```
src/investment_monitor/
├── collectors/
│   └── insider.py
```

### SEC EDGAR Details

**RSS Feed URL:**
```
https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK={cik}&type=4&company=&dateb=&owner=only&count=40&output=atom
```

**CIK Lookup:**
```
https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type=4&owner=include&count=1&action=getcompany
```

**Form 4 XML Structure (key fields):**
```xml
<ownershipDocument>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>Cook Timothy D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>true</isDirector>
            <isOfficer>true</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-01-15</value></transactionDate>
            <transactionCoding>
                <transactionCode>S</transactionCode> <!-- S=Sale, P=Purchase -->
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>180.00</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
```

### Implementation Notes

```python
class InsiderCollector(BaseCollector):
    name = "insider"
    rate_limit_calls = 10  # SEC asks for max 10 requests/second
    rate_limit_period = 1
    
    async def collect(self, tickers: list[str]) -> CollectorResult:
        ...
    
    async def collect_single(self, ticker: str) -> int:
        """
        1. Get CIK for ticker
        2. Fetch RSS feed of recent Form 4 filings
        3. For each new filing, fetch and parse XML
        4. Extract transactions and save to database
        """
        ...
    
    def _parse_form4_xml(self, xml_content: str) -> list[InsiderTransaction]:
        """Parse Form 4 XML into transaction records"""
        ...
    
    def _get_cik_for_ticker(self, ticker: str) -> str | None:
        """Look up SEC CIK for a ticker symbol"""
        ...
```

### Acceptance Criteria
- [ ] Fetches Form 4 filings for all portfolio tickers
- [ ] Correctly parses transaction type (buy/sell)
- [ ] Extracts owner name and title
- [ ] Calculates total transaction value
- [ ] Stores SEC filing URL for reference
- [ ] Deduplicates based on filing URL
- [ ] Respects SEC rate limits

### Dependencies
- T07 (Collector Base Class)

---

## Task T10: News Collector

### Context
Aggregate news from RSS feeds related to portfolio tickers. Focus on headline and source capture; AI relevance scoring comes later.

### Requirements
1. Fetch from multiple RSS sources
2. Match headlines to portfolio tickers
3. Deduplicate by URL
4. Store with source and timestamp

### Files to Create

```
src/investment_monitor/
├── collectors/
│   └── news.py
config/
└── sources.yaml.example
```

### RSS Sources

```yaml
# sources.yaml
news_feeds:
  - name: "Yahoo Finance"
    url: "https://finance.yahoo.com/rss/headline?s={ticker}"
    per_ticker: true
    
  - name: "Reuters Business"
    url: "https://www.reutersagency.com/feed/?best-topics=business-finance"
    per_ticker: false
    
  - name: "SEC Filings"
    url: "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&count=40&output=atom"
    per_ticker: false
    
  - name: "Seeking Alpha"
    url: "https://seekingalpha.com/market_currents.xml"
    per_ticker: false
```

### Implementation Notes

```python
import feedparser

class NewsCollector(BaseCollector):
    name = "news"
    
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        1. Fetch all configured RSS feeds
        2. For per-ticker feeds, substitute ticker symbol
        3. Parse entries and match to portfolio tickers
        4. Deduplicate and save new items
        """
        ...
    
    def _ticker_mentioned(self, text: str, tickers: list[str]) -> list[str]:
        """
        Check if any portfolio tickers are mentioned in text.
        Handle variations: $AAPL, AAPL, Apple Inc, etc.
        Returns list of matched tickers.
        """
        ...
    
    def _parse_feed(self, feed_url: str) -> list[dict]:
        """Parse RSS feed and return entries"""
        ...
```

### Acceptance Criteria
- [ ] Fetches from all configured RSS sources
- [ ] Correctly parses feed entries
- [ ] Matches headlines to tickers (including company names)
- [ ] Deduplicates by URL
- [ ] Handles feed fetch failures gracefully
- [ ] Stores source name and publication time

### Dependencies
- T07 (Collector Base Class)

---

## Task T11: Earnings Calendar Collector

### Context
Track upcoming earnings dates for portfolio holdings so users can be alerted before announcements.

### Requirements
1. Fetch earnings dates for all portfolio tickers
2. Store upcoming dates in database
3. Support alerting X days before earnings

### Files to Create

```
src/investment_monitor/
├── collectors/
│   └── earnings.py
```

### Data Source

yfinance provides earnings dates:

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
calendar = ticker.calendar
# Returns dict with 'Earnings Date' (can be a range)
```

Alternatively, Finnhub free tier:
```
GET https://finnhub.io/api/v1/calendar/earnings?from=2026-01-01&to=2026-02-01&token={api_key}
```

### Implementation Notes

```python
class EarningsCollector(BaseCollector):
    name = "earnings"
    
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """Fetch earnings dates for all tickers"""
        ...
    
    def get_upcoming_earnings(
        self, 
        tickers: list[str], 
        days_ahead: int = 14
    ) -> list[dict]:
        """
        Get earnings within the next N days.
        Returns:
            [
                {"ticker": "AAPL", "date": "2026-02-02", "days_until": 5},
                ...
            ]
        """
        ...
```

### Acceptance Criteria
- [ ] Fetches earnings dates for all portfolio tickers
- [ ] Stores in database with update timestamps
- [ ] Handles missing/unknown dates gracefully
- [ ] `get_upcoming_earnings()` returns sorted list
- [ ] Works with both stocks and ETFs (ETFs won't have earnings)

### Dependencies
- T07 (Collector Base Class)

---

## Task T12: ETF Holdings Collector

### Context
Track ETF holdings to detect when funds add or drop positions. ETF providers publish daily holdings as CSVs.

### Requirements
1. Download holdings CSVs from ETF provider sites
2. Parse and store current holdings
3. Detect changes vs previous day
4. Handle different CSV formats by provider

### Files to Create

```
src/investment_monitor/
├── collectors/
│   └── etf_holdings.py
```

### ETF Holdings Sources

| Provider | URL Pattern | Format |
|----------|-------------|--------|
| Vanguard | `https://investor.vanguard.com/investment-products/etfs/profile/api/{ticker}/portfolio-holding/stock` | JSON |
| iShares | `https://www.ishares.com/us/products/{fund_id}/...` | CSV |
| SPDR | `https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx` | Excel |

This is complex because each provider has different formats. Start with one provider (Vanguard is easiest with JSON).

### Implementation Notes

```python
class ETFHoldingsCollector(BaseCollector):
    name = "etf_holdings"
    
    async def collect(self, tickers: list[str]) -> CollectorResult:
        """
        Fetch holdings for ETF tickers only.
        Skip non-ETF tickers.
        """
        ...
    
    async def collect_single(self, etf_ticker: str) -> int:
        """
        1. Determine provider from ticker
        2. Fetch holdings in provider-specific format
        3. Parse into standard format
        4. Compare to previous holdings
        5. Save current holdings
        """
        ...
    
    def get_holdings_changes(
        self, 
        etf_ticker: str
    ) -> dict:
        """
        Compare current holdings to previous day.
        Returns:
            {
                "added": [{"ticker": "NVDA", "weight": 0.5}],
                "removed": [{"ticker": "INTC", "weight": 0.3}],
                "weight_changes": [{"ticker": "AAPL", "old": 5.2, "new": 5.8}]
            }
        """
        ...
```

### Acceptance Criteria
- [ ] Fetches holdings for at least one ETF provider (Vanguard)
- [ ] Parses holdings into standard format
- [ ] Detects added/removed positions
- [ ] Detects significant weight changes
- [ ] Handles non-ETF tickers gracefully (skip)
- [ ] Stores historical holdings for comparison

### Dependencies
- T07 (Collector Base Class)

---

# PHASE 3: Alert Engine

---

## Task T13: Rule-Based Alert Engine

### Context
Process collected data against configured thresholds to generate alerts. No AI required for this task.

### Requirements
1. Check all alert rules against latest data
2. Generate AlertMessage objects for triggered rules
3. Support enabling/disabling individual rules
4. Avoid duplicate alerts (don't re-alert same event)

### Files to Create

```
src/investment_monitor/
├── alerts/
│   ├── __init__.py
│   ├── engine.py        # Main alert processor
│   └── rules.py         # Individual rule implementations
```

### Alert Rules to Implement

```python
# rules.py

def check_price_alerts(
    session: Session,
    portfolio: Portfolio,
    config: PriceAlertSettings,
) -> list[AlertMessage]:
    """
    Check:
    - Daily price drop > threshold
    - Daily price rise > threshold
    - Weekly price drop > threshold
    - Price below cost basis
    """
    ...

def check_volume_alerts(
    session: Session,
    portfolio: Portfolio,
    config: VolumeAlertSettings,
) -> list[AlertMessage]:
    """
    Check:
    - Volume > X times 20-day average
    """
    ...

def check_insider_alerts(
    session: Session,
    portfolio: Portfolio,
    config: InsiderAlertSettings,
) -> list[AlertMessage]:
    """
    Check:
    - Insider buys > min value
    - Insider sells > min value (different threshold)
    - CEO/CFO transactions (any size)
    - Cluster buying/selling (multiple insiders)
    """
    ...

def check_earnings_alerts(
    session: Session,
    portfolio: Portfolio,
    config: EarningsAlertSettings,
) -> list[AlertMessage]:
    """
    Check:
    - Earnings within lookahead window
    """
    ...

def check_news_keyword_alerts(
    session: Session,
    portfolio: Portfolio,
    config: NewsAlertSettings,
) -> list[AlertMessage]:
    """
    Check:
    - Headlines containing configured keywords
    (AI relevance scoring is separate task)
    """
    ...
```

### Alert Engine

```python
# engine.py

class AlertEngine:
    def __init__(
        self,
        session: Session,
        portfolio: Portfolio,
        alerts_config: AlertsConfig,
    ):
        ...
    
    def run_all_checks(self) -> list[AlertMessage]:
        """Run all enabled alert checks and return triggered alerts"""
        alerts = []
        
        if self.alerts_config.price.enabled:
            alerts.extend(check_price_alerts(...))
        
        if self.alerts_config.volume.enabled:
            alerts.extend(check_volume_alerts(...))
        
        # ... etc
        
        return self._deduplicate(alerts)
    
    def _deduplicate(self, alerts: list[AlertMessage]) -> list[AlertMessage]:
        """Remove alerts that were already sent recently"""
        ...
```

### Acceptance Criteria
- [ ] All rule types implemented and working
- [ ] Alerts include meaningful messages with context
- [ ] Deduplication prevents repeat alerts
- [ ] Rules can be individually disabled
- [ ] Handles missing data gracefully (no crash)
- [ ] Test coverage for each rule type

### Dependencies
- T02 (Database Schema)
- T03 (Portfolio Configuration)
- T04 (Alert Configuration)
- T08-T12 (All Collectors)

---

## Task T14: Alert Priority Classification

### Context
Assign priority levels to alerts so high-priority alerts get sent immediately while lower-priority ones go into digests.

### Requirements
1. Define priority rules for each alert type
2. Allow user override of default priorities
3. Priority affects notification routing

### Files to Create

```
src/investment_monitor/
├── alerts/
│   └── priority.py
```

### Priority Logic

```python
def classify_priority(alert: AlertMessage, config: AlertsConfig) -> Priority:
    """
    Assign priority based on alert type and severity.
    
    HIGH (immediate):
    - Price drop > 2x threshold
    - CEO/CFO insider sale
    - Keywords: "SEC investigation", "lawsuit", "fraud"
    
    MEDIUM (daily digest):
    - Normal threshold breaches
    - Insider transactions meeting criteria
    - Upcoming earnings
    
    LOW (weekly digest or log only):
    - Minor price movements
    - Low-relevance news
    """
    ...
```

### Acceptance Criteria
- [ ] Priority assigned to all alerts
- [ ] Severe events get HIGH priority
- [ ] Routine events get MEDIUM priority
- [ ] Low-signal events get LOW priority
- [ ] User can override default priorities in config

### Dependencies
- T13 (Rule-Based Alert Engine)

---

## Task T15: Alert Deduplication

### Context
Prevent sending the same alert multiple times. Track what's been sent and implement smart deduplication.

### Requirements
1. Track sent alerts in database
2. Define deduplication windows per alert type
3. Allow re-alerting after sufficient time passes

### Files to Create

```
src/investment_monitor/
├── alerts/
│   └── dedup.py
```

### Deduplication Rules

```python
DEDUP_WINDOWS = {
    "price_drop": timedelta(hours=24),      # Don't re-alert same drop for 24h
    "price_rise": timedelta(hours=24),
    "volume_spike": timedelta(hours=12),
    "insider_transaction": timedelta(days=7),  # Same filing
    "earnings_upcoming": timedelta(days=3),    # Remind again if still upcoming
    "news_keyword": timedelta(days=1),         # Same article
}

class AlertDeduplicator:
    def __init__(self, session: Session):
        ...
    
    def is_duplicate(self, alert: AlertMessage) -> bool:
        """Check if similar alert was sent within dedup window"""
        ...
    
    def mark_sent(self, alert: AlertMessage) -> None:
        """Record that alert was sent"""
        ...
    
    def filter_duplicates(
        self, 
        alerts: list[AlertMessage]
    ) -> list[AlertMessage]:
        """Remove duplicates from alert list"""
        ...
```

### Acceptance Criteria
- [ ] Duplicate alerts are filtered out
- [ ] Different dedup windows per alert type
- [ ] Alerts are re-sent after window expires
- [ ] Sent alerts tracked in database
- [ ] Can manually clear dedup cache if needed

### Dependencies
- T02 (Database Schema)
- T13 (Rule-Based Alert Engine)

---

# PHASE 4: AI Integration

---

## Task T16: Local LLM Integration (Ollama)

### Context
Integrate with Ollama for local LLM inference. Use for news relevance scoring and basic sentiment analysis.

### Requirements
1. Ollama client wrapper
2. Prompt templates for common tasks
3. Response parsing and validation
4. Fallback handling if Ollama unavailable

### Files to Create

```
src/investment_monitor/
├── analysis/
│   ├── __init__.py
│   ├── local_llm.py
│   └── prompts.py
```

### Implementation Notes

```python
import ollama

class LocalLLM:
    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
    ):
        self.client = ollama.Client(host=base_url)
        self.model = model
    
    async def score_relevance(
        self, 
        headline: str, 
        ticker: str,
        company_name: str,
    ) -> float:
        """
        Score news relevance 0-10.
        Returns -1 if LLM unavailable.
        """
        prompt = RELEVANCE_PROMPT.format(
            headline=headline,
            ticker=ticker,
            company_name=company_name,
        )
        ...
    
    async def classify_sentiment(
        self,
        text: str,
    ) -> str:
        """
        Classify as 'bullish', 'bearish', or 'neutral'.
        Returns 'unknown' if LLM unavailable.
        """
        ...
    
    async def summarize(
        self,
        text: str,
        max_sentences: int = 2,
    ) -> str:
        """Summarize text to N sentences"""
        ...
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is loaded"""
        ...
```

### Prompt Templates

```python
# prompts.py

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
```

### Acceptance Criteria
- [ ] Connects to Ollama successfully
- [ ] Relevance scoring returns 0-10 float
- [ ] Sentiment classification returns valid category
- [ ] Handles Ollama being unavailable gracefully
- [ ] Responses are parsed correctly (handles LLM verbosity)
- [ ] Reasonable latency (<2s per call for small models)

### Dependencies
- T01 (Project Scaffolding)

---

## Task T17: AI-Enhanced News Processing

### Context
Use local LLM to score news relevance and filter out noise before including in alerts/digests.

### Requirements
1. Score all new news items for relevance
2. Update database with scores
3. Filter alerts based on relevance threshold
4. Handle high volume efficiently (batch if possible)

### Files to Create

```
src/investment_monitor/
├── analysis/
│   └── news_processor.py
```

### Implementation

```python
class NewsProcessor:
    def __init__(
        self,
        session: Session,
        llm: LocalLLM,
        portfolio: Portfolio,
        min_relevance: float = 5.0,
    ):
        ...
    
    async def process_unscored_news(self) -> int:
        """
        Find news items without relevance scores and score them.
        Returns number of items processed.
        """
        ...
    
    async def get_relevant_news(
        self,
        ticker: str | None = None,
        hours: int = 24,
    ) -> list[NewsItem]:
        """
        Get news items above relevance threshold.
        Optionally filter by ticker.
        """
        ...
    
    async def _score_item(self, item: NewsItem) -> float:
        """Score a single news item"""
        # Get company name from ticker
        # Call LLM for relevance score
        # Update database
        ...
```

### Acceptance Criteria
- [ ] All news items get relevance scores
- [ ] Scores persist in database
- [ ] Low-relevance items filtered from alerts
- [ ] Handles LLM unavailability (skip scoring, don't crash)
- [ ] Reasonable throughput (process 50+ items in <2 minutes)

### Dependencies
- T10 (News Collector)
- T16 (Local LLM Integration)

---

## Task T18: Claude API Weekly Synthesis

### Context
Use Claude API for weekly portfolio synthesis. Batch pre-filtered data and request high-level insights.

### Requirements
1. Claude API client wrapper
2. Weekly synthesis prompt template
3. Cost tracking and limiting
4. Store synthesis results

### Files to Create

```
src/investment_monitor/
├── analysis/
│   └── claude_api.py
```

### Implementation

```python
from anthropic import Anthropic

class ClaudeAnalyzer:
    def __init__(
        self,
        api_key: str,
        max_monthly_spend: float = 5.00,
    ):
        self.client = Anthropic(api_key=api_key)
        self.max_monthly_spend = max_monthly_spend
    
    async def weekly_synthesis(
        self,
        portfolio: Portfolio,
        week_data: WeeklyData,
    ) -> str:
        """
        Generate weekly portfolio synthesis.
        
        Args:
            portfolio: User's holdings and theses
            week_data: Aggregated data from the week
                - Price movements
                - Insider transactions
                - Relevant news (pre-filtered)
                - Upcoming earnings
        
        Returns:
            Synthesis text for inclusion in weekly digest
        """
        prompt = self._build_synthesis_prompt(portfolio, week_data)
        
        # Check cost budget before calling
        if not self._within_budget():
            return "Weekly AI synthesis skipped (budget limit reached)"
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self._record_cost(response.usage)
        return response.content[0].text
    
    async def analyze_event(
        self,
        portfolio: Portfolio,
        event_description: str,
        ticker: str,
    ) -> str:
        """
        Deep analysis of specific event (earnings surprise, major news).
        Called ad-hoc when significant events occur.
        """
        ...
    
    def _within_budget(self) -> bool:
        """Check if we're within monthly spend limit"""
        ...
    
    def _record_cost(self, usage) -> None:
        """Track API usage costs"""
        ...
```

### Weekly Synthesis Prompt

```python
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

Keep your response concise and actionable. I want signal, not noise. Write in prose, not bullet points."""
```

### Acceptance Criteria
- [ ] Successfully calls Claude API
- [ ] Synthesis is relevant and actionable
- [ ] Cost tracking prevents overspend
- [ ] Handles API errors gracefully
- [ ] Stores synthesis for later reference
- [ ] Respects rate limits

### Dependencies
- T13 (Rule-Based Alert Engine)
- T06 (Digest Formatter)

---

# PHASE 5: Orchestration

---

## Task T19: Main Orchestrator

### Context
Tie everything together into a single entry point that can be scheduled via cron.

### Requirements
1. Run all collectors
2. Run alert engine
3. Send notifications appropriately
4. Handle errors without crashing
5. Log execution summary

### Files to Create

```
src/investment_monitor/
├── main.py
```

### Implementation

```python
import asyncio
from datetime import datetime

async def run_monitor(
    config_path: str = "config",
    run_type: str = "regular",  # "regular", "digest", "weekly"
) -> None:
    """
    Main entry point for the investment monitor.
    
    Run types:
    - regular: Collect data, check alerts, send immediate notifications
    - digest: Compile and send daily digest
    - weekly: Run weekly synthesis with Claude API
    """
    logger.info(f"Starting investment monitor run: {run_type}")
    start_time = datetime.now()
    
    # Load configuration
    settings = load_settings(config_path)
    
    # Initialize database
    session = get_session()
    
    try:
        if run_type in ("regular", "digest"):
            # Run collectors
            await run_collectors(session, settings)
            
            # Process news with AI (if available)
            await process_news_ai(session, settings)
            
            # Run alert engine
            alerts = run_alert_checks(session, settings)
            
            # Send immediate alerts (HIGH priority)
            await send_immediate_alerts(alerts, settings)
        
        if run_type == "digest":
            # Compile and send daily digest
            await send_daily_digest(session, settings)
        
        if run_type == "weekly":
            # Run Claude synthesis and send weekly digest
            await send_weekly_digest(session, settings)
    
    except Exception as e:
        logger.error(f"Monitor run failed: {e}")
        # Notify admin of failure
        await send_error_notification(e, settings)
    
    finally:
        elapsed = datetime.now() - start_time
        logger.info(f"Monitor run complete in {elapsed.total_seconds():.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["regular", "digest", "weekly"], default="regular")
    parser.add_argument("--config", default="config")
    args = parser.parse_args()
    
    asyncio.run(run_monitor(config_path=args.config, run_type=args.type))
```

### Acceptance Criteria
- [ ] All components run in correct order
- [ ] Errors in one component don't crash others
- [ ] Execution time logged
- [ ] Can run different modes (regular, digest, weekly)
- [ ] Works from command line

### Dependencies
- All previous tasks

---

## Task T20: Cron Configuration and Docker Setup

### Context
Package everything for deployment on the home lab NUC with proper scheduling.

### Requirements
1. Docker Compose configuration
2. Cron schedule definitions
3. Environment variable handling
4. Health check endpoint (optional)

### Files to Create

```
investment-monitor/
├── docker-compose.yaml
├── Dockerfile
├── cron/
│   └── crontab
├── scripts/
│   ├── run_regular.sh
│   ├── run_digest.sh
│   └── run_weekly.sh
```

### docker-compose.yaml

```yaml
version: "3.8"

services:
  monitor:
    build: .
    container_name: investment-monitor
    restart: unless-stopped
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - SENDGRID_API_KEY=${SENDGRID_API_KEY}
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TZ=America/Los_Angeles
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    container_name: ollama
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama
    # Uncomment for GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  cron:
    build: .
    container_name: investment-cron
    restart: unless-stopped
    entrypoint: ["cron", "-f"]
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ./cron/crontab:/etc/cron.d/monitor
    environment:
      - SENDGRID_API_KEY=${SENDGRID_API_KEY}
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TZ=America/Los_Angeles

volumes:
  ollama_data:
```

### Crontab

```cron
# Investment Monitor Schedules
# Timezone: America/Los_Angeles (set in docker-compose)

# Regular data collection and immediate alerts
# Run every 4 hours during market days
0 6,10,14,18 * * 1-5 /app/scripts/run_regular.sh >> /app/logs/cron.log 2>&1

# Daily digest at 7am
0 7 * * * /app/scripts/run_digest.sh >> /app/logs/cron.log 2>&1

# Weekly synthesis on Sunday at 6pm
0 18 * * 0 /app/scripts/run_weekly.sh >> /app/logs/cron.log 2>&1
```

### Acceptance Criteria
- [ ] `docker-compose up -d` starts all services
- [ ] Cron jobs execute on schedule
- [ ] Ollama accessible from main container
- [ ] Data persists across restarts
- [ ] Logs accessible on host
- [ ] Environment variables properly passed

### Dependencies
- T19 (Main Orchestrator)

---

# Appendix: Quick Reference

## Environment Variables

```bash
# .env
SENDGRID_API_KEY=SG.xxx          # Optional: for email
SLACK_WEBHOOK_URL=https://...    # Optional: for Slack
ANTHROPIC_API_KEY=sk-ant-xxx     # Optional: for Claude
FINNHUB_API_KEY=xxx              # Optional: for Finnhub data
OLLAMA_HOST=http://ollama:11434  # Local LLM endpoint
```

## Common Commands

```bash
# Development
pip install -e ".[dev,ai,notifications]"
pytest tests/

# Run manually
python -m investment_monitor.main --type regular
python -m investment_monitor.main --type digest
python -m investment_monitor.main --type weekly

# Docker
docker-compose up -d
docker-compose logs -f monitor
docker exec -it investment-monitor python -m investment_monitor.main --type regular

# Ollama
docker exec -it ollama ollama pull phi3:mini
docker exec -it ollama ollama list
```

## Testing Individual Components

```bash
# Test single collector
python -c "from investment_monitor.collectors.prices import PriceCollector; ..."

# Test notifications
python -c "from investment_monitor.notifications.email import EmailChannel; ..."

# Test local LLM
python -c "from investment_monitor.analysis.local_llm import LocalLLM; ..."
```

---

*Document created for AI agent task delegation*
*Last updated: January 2026*
