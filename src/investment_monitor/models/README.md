# Models Module

Pydantic models for configuration and data validation.

## Overview

```
models/
├── portfolio.py   # Portfolio, Holding, WatchlistItem
└── alerts.py      # Alert configuration settings
```

## Portfolio Models

### Holding

Represents a stock position you own.

```python
from investment_monitor.models import Holding
from decimal import Decimal

holding = Holding(
    ticker="AAPL",           # Required, 1-5 uppercase letters
    shares=Decimal("50"),    # Required, must be > 0
    cost_basis=Decimal("165.00"),  # Required, must be > 0
    thesis="Services growth driving margin expansion"  # Optional, max 500 chars
)

# Computed property
print(holding.total_cost)  # Decimal("8250.00")
```

**Validation:**
- `ticker`: Must match pattern `^[A-Z]{1,5}$`
- `shares`: Must be > 0
- `cost_basis`: Must be > 0

### WatchlistItem

Stocks you're monitoring but don't own.

```python
from investment_monitor.models import WatchlistItem

item = WatchlistItem(
    ticker="GOOGL",          # Required
    reason="Waiting for better entry",  # Optional
    target_price=Decimal("140.00")      # Optional
)
```

### Portfolio

Container for holdings and watchlist.

```python
from investment_monitor.models import Portfolio
from pathlib import Path

# Load from YAML
portfolio = Portfolio.from_yaml(Path("config/portfolio.yaml"))

# Access properties
portfolio.all_tickers      # ["AAPL", "GOOGL", "MSFT"] - deduplicated, sorted
portfolio.holding_tickers  # ["AAPL", "MSFT"] - just holdings

# Lookup methods
holding = portfolio.get_holding("AAPL")  # Returns Holding or None
thesis = portfolio.get_thesis("AAPL")    # Returns str or None
cost = portfolio.get_cost_basis("AAPL")  # Returns Decimal or None
```

## Alert Configuration Models

All settings have sensible defaults and can be selectively disabled.

### PriceAlertSettings

```python
from investment_monitor.models import PriceAlertSettings

settings = PriceAlertSettings(
    enabled=True,           # Enable/disable this alert type
    daily_drop_pct=3.0,     # Alert if daily drop > 3%
    daily_rise_pct=5.0,     # Alert if daily rise > 5%
    weekly_drop_pct=7.0,    # Alert if weekly drop > 7%
    below_cost_basis=True   # Alert if price < your cost basis
)
```

**Validation:** All percentages must be 0-100.

### VolumeAlertSettings

```python
settings = VolumeAlertSettings(
    enabled=True,
    lookback_days=20,   # Days to calculate average (5-60)
    multiplier=2.5      # Alert if volume > 2.5x average (min 1.0)
)
```

### InsiderAlertSettings

```python
settings = InsiderAlertSettings(
    enabled=True,
    min_buy_value=100_000,    # Minimum $ for buy alerts
    min_sell_value=500_000,   # Minimum $ for sell alerts
    alert_ceo_cfo_any=True,   # Alert any CEO/CFO transaction
    cluster_threshold=3,      # N insiders in cluster (min 2)
    cluster_days=7            # Days for cluster window (min 1)
)
```

### EarningsAlertSettings

```python
settings = EarningsAlertSettings(
    enabled=True,
    lookahead_days=7  # Alert N days before earnings (1-30)
)
```

### NewsAlertSettings

```python
settings = NewsAlertSettings(
    enabled=True,
    keywords=[
        "lawsuit", "SEC", "investigation", "guidance",
        "acquisition", "merger", "layoffs", "dividend", "buyback"
    ],
    min_relevance_score=5.0  # 0-10, requires Ollama for scoring
)
```

### ETFAlertSettings

```python
settings = ETFAlertSettings(
    enabled=True,
    holdings_change=True,     # Alert on added/removed positions
    weight_change_pct=1.0     # Alert if weight changes > 1%
)
```

### AlertsConfig

Container for all alert settings.

```python
from investment_monitor.models import AlertsConfig
from pathlib import Path

# Load from YAML (merges with defaults)
config = AlertsConfig.from_yaml(Path("config/alerts.yaml"))

# Access individual settings
if config.price.enabled:
    threshold = config.price.daily_drop_pct

# Disable a category
config.insider.enabled = False
```

## YAML Format

### portfolio.yaml

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

### alerts.yaml

```yaml
price:
  enabled: true
  daily_drop_pct: 3.0
  daily_rise_pct: 5.0
  weekly_drop_pct: 7.0
  below_cost_basis: true

volume:
  enabled: true
  lookback_days: 20
  multiplier: 2.5

insider:
  enabled: true
  min_buy_value: 100000
  min_sell_value: 500000
  alert_ceo_cfo_any: true

earnings:
  enabled: true
  lookahead_days: 7

news:
  enabled: true
  keywords:
    - lawsuit
    - SEC
    - investigation
  min_relevance_score: 5.0

etf:
  enabled: false  # Disable ETF alerts
```

## Troubleshooting

### Validation Errors

```python
# Invalid ticker (lowercase)
Holding(ticker="aapl", ...)
# ValidationError: String should match pattern '^[A-Z]{1,5}$'

# Invalid shares (zero)
Holding(ticker="AAPL", shares=0, ...)
# ValidationError: Input should be greater than 0

# Invalid percentage
PriceAlertSettings(daily_drop_pct=150)
# ValidationError: Input should be less than or equal to 100
```

### YAML Loading Errors

```python
# File not found - returns empty Portfolio/default AlertsConfig
portfolio = Portfolio.from_yaml(Path("nonexistent.yaml"))
# Returns Portfolio(holdings=[], watchlist=[])

# Invalid YAML syntax
# yaml.YAMLError: mapping values are not allowed here
```

### Type Coercion

Pydantic automatically coerces compatible types:
```python
# String to Decimal
Holding(ticker="AAPL", shares="50", cost_basis="165.00")  # Works

# Int to Decimal
Holding(ticker="AAPL", shares=50, cost_basis=165)  # Works
```

## Extending

### Add New Field to Holding

```python
# In portfolio.py
class Holding(BaseModel):
    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    shares: Decimal = Field(..., gt=0)
    cost_basis: Decimal = Field(..., gt=0)
    thesis: str = Field(default="", max_length=500)

    # New field with default (backward compatible)
    sector: str = Field(default="")
```

### Add New Alert Type

```python
# In alerts.py
class DividendAlertSettings(BaseModel):
    enabled: bool = True
    min_yield_pct: float = Field(default=2.0, ge=0)

class AlertsConfig(BaseModel):
    # ... existing
    dividend: DividendAlertSettings = Field(default_factory=DividendAlertSettings)
```

Then implement the corresponding rule in `alerts/rules.py`.
