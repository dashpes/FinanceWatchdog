# Storage Module

Database layer using SQLAlchemy ORM with SQLite.

## Overview

```
storage/
├── database.py    # Engine, session management
├── models.py      # ORM table definitions
└── operations.py  # CRUD functions
```

## Database Schema

### prices
Daily OHLCV data from yfinance.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol (indexed) |
| date | DATE | Trading date |
| open | REAL | Opening price |
| high | REAL | Daily high |
| low | REAL | Daily low |
| close | REAL | Closing price (required) |
| volume | INTEGER | Trading volume |
| created_at | TIMESTAMP | Record creation time |

**Unique constraint:** `(ticker, date)`

### insider_transactions
SEC Form 4 filings.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol (indexed) |
| filing_date | DATE | SEC filing date |
| trade_date | DATE | Actual trade date |
| owner_name | TEXT | Insider name |
| owner_title | TEXT | CEO, CFO, Director, etc. |
| transaction_type | TEXT | P (purchase), S (sale) |
| shares | INTEGER | Number of shares |
| price_per_share | REAL | Transaction price |
| total_value | REAL | shares * price |
| sec_url | TEXT | Link to SEC filing (unique) |
| created_at | TIMESTAMP | Record creation time |

### news_items
Aggregated news from RSS feeds.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Related stock (nullable, indexed) |
| headline | TEXT | Article headline |
| source | TEXT | News source name |
| url | TEXT | Article URL (unique) |
| published_at | TIMESTAMP | Publication time |
| relevance_score | REAL | AI-assigned 0-10 (nullable) |
| sentiment | TEXT | bullish/bearish/neutral (nullable) |
| created_at | TIMESTAMP | Record creation time |

### alerts_sent
Deduplication tracking for sent alerts.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| alert_type | TEXT | price/insider/news/earnings (indexed) |
| ticker | TEXT | Related stock (indexed) |
| message | TEXT | Alert content |
| priority | TEXT | high/medium/low |
| sent_at | TIMESTAMP | When alert was sent |
| channel | TEXT | console/slack/email |
| dedup_key | TEXT | Deduplication key (indexed) |

### earnings_dates
Upcoming earnings calendar.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol (indexed) |
| earnings_date | DATE | Expected earnings date |
| confirmed | BOOLEAN | Whether date is confirmed |
| created_at | TIMESTAMP | Record creation |
| updated_at | TIMESTAMP | Last update |

**Unique constraint:** `(ticker, earnings_date)`

### etf_holdings
ETF composition tracking.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| etf_ticker | TEXT | ETF symbol (indexed) |
| holding_ticker | TEXT | Held stock symbol |
| shares | REAL | Number of shares held |
| weight_pct | REAL | Percentage of ETF |
| as_of_date | DATE | Holdings date |
| created_at | TIMESTAMP | Record creation |

**Unique constraint:** `(etf_ticker, holding_ticker, as_of_date)`

## Usage

### Initialize Database

```python
from investment_monitor.storage import init_db, get_session

# Creates database file and all tables
init_db("data/portfolio.db")

# Use session as context manager
with get_session() as session:
    # Your operations here
    pass  # Auto-commits on success, rollbacks on exception
```

### Common Operations

```python
from investment_monitor.storage import (
    # Price operations
    save_price, get_latest_price, get_prices, price_exists,

    # Insider operations
    save_insider_transaction, get_insider_transactions, insider_transaction_exists,

    # News operations
    save_news_item, news_exists, get_unscored_news, get_recent_news,

    # Alert operations
    save_alert, get_recent_alerts, alert_exists_by_dedup_key,

    # Earnings operations
    save_earnings_date, get_upcoming_earnings,

    # ETF operations
    save_etf_holdings, get_etf_holdings,
)

# Example: Check if price exists before saving
with get_session() as session:
    if not price_exists(session, "AAPL", date(2026, 1, 28)):
        price = Price(ticker="AAPL", date=date(2026, 1, 28), close=180.0)
        save_price(session, price)
```

## Troubleshooting

### Database Locked
SQLite locks the entire database during writes. If you see "database is locked":
- Ensure only one process writes at a time
- Use shorter transactions
- Consider WAL mode for concurrent reads:
  ```python
  engine = create_engine("sqlite:///data/portfolio.db?mode=wal")
  ```

### Missing Tables
Tables are created on `init_db()`. If tables are missing:
```python
from investment_monitor.storage.models import Base
from investment_monitor.storage.database import _engine

Base.metadata.create_all(bind=_engine)
```

### Query Performance
Add indexes for frequently queried columns:
```python
# Already indexed: ticker, date, alert_type, dedup_key
# Add custom index:
from sqlalchemy import Index
Index('ix_custom', Model.column).create(bind=engine)
```

### Viewing Data
```bash
# SQLite CLI
sqlite3 data/portfolio.db
.tables
.schema prices
SELECT * FROM prices WHERE ticker='AAPL' ORDER BY date DESC LIMIT 5;

# Or use DB Browser for SQLite (GUI)
```

## Extending

### Add New Table

1. Add model to `models.py`:
```python
class NewTable(Base):
    __tablename__ = "new_table"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # ... columns
```

2. Add CRUD functions to `operations.py`:
```python
def save_new_record(session: Session, record: NewTable) -> int:
    session.add(record)
    session.flush()
    return record.id
```

3. Export in `__init__.py`

4. Run `init_db()` to create the table (or use Alembic for migrations)

### Add Migration (Alembic)
```bash
pip install alembic
alembic init alembic
alembic revision --autogenerate -m "add new table"
alembic upgrade head
```
