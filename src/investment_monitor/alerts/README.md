# Alerts Module

Rule-based alert engine with priority classification and deduplication.

## Overview

```
alerts/
├── engine.py     # AlertEngine - orchestrates all checks
├── rules.py      # Individual rule implementations
├── priority.py   # Priority classification (HIGH/MEDIUM/LOW)
└── dedup.py      # Deduplication to prevent repeat alerts
```

## Alert Engine

The `AlertEngine` runs all enabled alert checks and returns triggered alerts.

```python
from investment_monitor.alerts import AlertEngine
from investment_monitor.models import Portfolio, AlertsConfig

engine = AlertEngine(session, portfolio, alerts_config)

# Run all checks
alerts = engine.run_all_checks()

# Run individual checks
price_alerts = engine.check_prices()
insider_alerts = engine.check_insider()
volume_alerts = engine.check_volume()
earnings_alerts = engine.check_earnings()
news_alerts = engine.check_news()
```

## Alert Rules

### Price Alerts (`rules.py`)

```python
from investment_monitor.alerts.rules import check_price_alerts

alerts = check_price_alerts(session, portfolio, config.price)
```

**Triggers:**
- Daily drop > `daily_drop_pct` (default 3%)
- Daily rise > `daily_rise_pct` (default 5%)
- Weekly drop > `weekly_drop_pct` (default 7%)
- Price below cost basis (if `below_cost_basis=True`)

**Example alert:**
```
Title: AAPL dropped 4.2% today
Body: AAPL fell from $185.00 to $177.23 (-4.2%). Weekly change: -6.1%
Priority: MEDIUM (or HIGH if > 2x threshold)
```

### Volume Alerts

```python
from investment_monitor.alerts.rules import check_volume_alerts

alerts = check_volume_alerts(session, portfolio, config.volume)
```

**Triggers:**
- Volume > `multiplier` × 20-day average (default 2.5x)

**Example alert:**
```
Title: AAPL unusual volume: 3.2x average
Body: Volume 156M vs 20-day avg 49M (3.2x)
```

### Insider Alerts

```python
from investment_monitor.alerts.rules import check_insider_alerts

alerts = check_insider_alerts(session, portfolio, config.insider)
```

**Triggers:**
- Buy transaction > `min_buy_value` (default $100k)
- Sell transaction > `min_sell_value` (default $500k)
- Any CEO/CFO transaction (if `alert_ceo_cfo_any=True`)
- Cluster: >= `cluster_threshold` insiders in `cluster_days`

**Example alert:**
```
Title: AAPL insider sale: Tim Cook (CEO)
Body: Sold 50,000 shares at $180.00 ($9,000,000 total)
URL: https://sec.gov/...
Priority: HIGH (CEO sale)
```

### Earnings Alerts

```python
from investment_monitor.alerts.rules import check_earnings_alerts

alerts = check_earnings_alerts(session, portfolio, config.earnings)
```

**Triggers:**
- Earnings date within `lookahead_days` (default 7)

**Example alert:**
```
Title: AAPL earnings in 5 days
Body: Earnings expected on Feb 2, 2026
```

### News Keyword Alerts

```python
from investment_monitor.alerts.rules import check_news_keyword_alerts

alerts = check_news_keyword_alerts(session, portfolio, config.news)
```

**Triggers:**
- Headline contains any keyword from `keywords` list
- Relevance score >= `min_relevance_score` (if scored by AI)

**Default keywords:** lawsuit, SEC, investigation, guidance, acquisition, merger, layoffs, dividend, buyback

**Example alert:**
```
Title: [AAPL] SEC investigation mentioned
Body: "Apple faces SEC investigation over..."
URL: https://...
Priority: HIGH (contains "SEC investigation")
```

## Priority Classification

```python
from investment_monitor.alerts import classify_priority
from investment_monitor.notifications import Priority

priority = classify_priority(alert, alerts_config)
# Returns: Priority.HIGH, Priority.MEDIUM, or Priority.LOW
```

### Priority Levels

| Priority | Routing | Examples |
|----------|---------|----------|
| HIGH | Immediate notification | SEC investigation, fraud, CEO sale, 2x threshold drops |
| MEDIUM | Daily digest | Normal threshold breaches, insider trades, upcoming earnings |
| LOW | Weekly digest / log only | Minor movements, routine news |

### HIGH Priority Triggers

- **Keywords:** SEC investigation, fraud, lawsuit, bankruptcy, delisted, halt, criminal, indictment, restatement, default
- **Price:** Drop exceeds 2x configured threshold
- **Insider:** CEO or CFO sale

### LOW Priority Triggers

- **Keywords:** minor, routine, scheduled, expected, unchanged
- **Price:** Movement < 50% of threshold

### Batch Classification

```python
from investment_monitor.alerts import classify_priority_batch, get_alerts_by_priority

# Classify multiple alerts
alerts_with_priority = classify_priority_batch(alerts, config)

# Group by priority
by_priority = get_alerts_by_priority(alerts, config)
# {
#     Priority.HIGH: [...],
#     Priority.MEDIUM: [...],
#     Priority.LOW: [...]
# }
```

## Deduplication

Prevents sending the same alert multiple times.

```python
from investment_monitor.alerts import AlertDeduplicator

dedup = AlertDeduplicator(session)

# Check if duplicate
if dedup.is_duplicate(alert):
    print("Already sent recently")

# Filter list
unique_alerts = dedup.filter_duplicates(alerts)

# Mark as sent
dedup.mark_sent(alert, channel="console")
```

### Dedup Windows

| Alert Type | Window |
|------------|--------|
| price_drop, price_rise | 24 hours |
| volume_spike | 12 hours |
| insider_transaction | 7 days |
| earnings_upcoming | 3 days |
| news_keyword | 1 day |
| default | 24 hours |

### Dedup Key Format

Keys are generated as: `{alert_type}:{ticker}:{title_hash}`

```python
key = dedup.generate_dedup_key(alert)
# "price_drop:AAPL:a1b2c3d4"
```

## Complete Workflow

```python
from investment_monitor.alerts import AlertEngine, AlertDeduplicator, classify_priority
from investment_monitor.notifications import NotificationManager, ConsoleChannel, Priority

# 1. Run alert checks
engine = AlertEngine(session, portfolio, alerts_config)
alerts = engine.run_all_checks()

# 2. Deduplicate
dedup = AlertDeduplicator(session)
alerts = dedup.filter_duplicates(alerts)

# 3. Classify priorities
for alert in alerts:
    alert.priority = classify_priority(alert, alerts_config)

# 4. Route by priority
manager = NotificationManager([ConsoleChannel()])
for alert in alerts:
    if alert.priority == Priority.HIGH:
        await manager.notify(alert)  # Immediate
        dedup.mark_sent(alert, "console")
    else:
        # Queue for digest
        pass
```

## Troubleshooting

### No Alerts Generated

1. Check if alerts are enabled in config:
   ```python
   print(config.price.enabled)  # Should be True
   ```

2. Check if data exists:
   ```python
   prices = get_prices(session, "AAPL", days=7)
   print(len(prices))  # Should have data
   ```

3. Check thresholds:
   ```python
   # Maybe threshold is too high
   print(config.price.daily_drop_pct)  # 3.0 = 3%
   ```

### Duplicate Alerts

1. Check dedup window:
   ```python
   from investment_monitor.alerts.dedup import DEDUP_WINDOWS
   print(DEDUP_WINDOWS["price_drop"])  # 24 hours
   ```

2. Clear dedup history (if needed):
   ```sql
   DELETE FROM alerts_sent WHERE alert_type = 'price_drop';
   ```

### Wrong Priority

Check the classification logic:
```python
from investment_monitor.alerts.priority import HIGH_PRIORITY_KEYWORDS
print(HIGH_PRIORITY_KEYWORDS)
```

## Extending

### Add New Rule

```python
# In rules.py
def check_dividend_alerts(
    session: Session,
    portfolio: Portfolio,
    config: DividendAlertSettings,
) -> list[AlertMessage]:
    alerts = []
    for holding in portfolio.holdings:
        # Check dividend data
        # Generate alert if criteria met
        if should_alert:
            alerts.append(AlertMessage(
                title=f"{holding.ticker} dividend announced",
                body=f"${amount} per share, ex-date {date}",
                ticker=holding.ticker,
                alert_type="dividend",
                priority=Priority.MEDIUM,
            ))
    return alerts
```

Then add to `AlertEngine.run_all_checks()`.

### Customize Priority

```python
# In priority.py, add to classify_priority:
if alert.alert_type == "dividend":
    if "special" in alert.body.lower():
        return Priority.HIGH
    return Priority.MEDIUM
```

### Adjust Dedup Window

```python
# In dedup.py
DEDUP_WINDOWS["dividend"] = timedelta(days=7)
```
