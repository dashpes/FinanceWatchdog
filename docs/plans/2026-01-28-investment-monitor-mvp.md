# Investment Monitor MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a personal investment monitoring system that tracks portfolio holdings, collects market data, generates alerts based on configurable rules, and delivers daily/weekly digests.

**Architecture:** Python async application with SQLite storage, modular collectors for different data sources (prices, insider transactions, news, earnings), rule-based alert engine, optional AI enhancement via Ollama/Claude, and Docker deployment with cron scheduling.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.0, Pydantic 2.0, yfinance, feedparser, httpx, loguru, Ollama, Anthropic API, Docker

---

## Phase 1: Foundation

### Task 1: Project Scaffolding (T01)

**Files:**
- Create: `investment-monitor/pyproject.toml`
- Create: `investment-monitor/README.md`
- Create: `investment-monitor/.env.example`
- Create: `investment-monitor/.gitignore`
- Create: `investment-monitor/src/investment_monitor/__init__.py`
- Create: `investment-monitor/src/investment_monitor/config.py`
- Create: `investment-monitor/src/investment_monitor/logging_config.py`
- Create: `investment-monitor/tests/__init__.py`
- Create: `investment-monitor/tests/test_config.py`
- Create: `investment-monitor/config/portfolio.yaml.example`
- Create: `investment-monitor/config/alerts.yaml.example`
- Create: `investment-monitor/config/sources.yaml.example`
- Create: `investment-monitor/config/notifications.yaml.example`
- Create: `investment-monitor/data/.gitkeep`
- Create: `investment-monitor/logs/.gitkeep`

**Step 1: Create directory structure and pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "investment-monitor"
version = "0.1.0"
description = "Personal investment monitoring system"
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

[tool.hatch.build.targets.wheel]
packages = ["src/investment_monitor"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

**Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
.env
config/*.yaml
!config/*.yaml.example
data/*.db
logs/*.log

# OS
.DS_Store
Thumbs.db
```

**Step 3: Create .env.example**

```bash
# Optional: Email notifications via SendGrid
SENDGRID_API_KEY=

# Optional: Slack notifications
SLACK_WEBHOOK_URL=

# Optional: Claude API for weekly synthesis
ANTHROPIC_API_KEY=

# Optional: Finnhub for additional data
FINNHUB_API_KEY=

# Ollama endpoint (defaults to localhost)
OLLAMA_HOST=http://localhost:11434
```

**Step 4: Create logging_config.py**

```python
# src/investment_monitor/logging_config.py
"""Logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """Configure loguru for console and file output."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler - concise format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler - detailed format with rotation
    logger.add(
        log_path / "monitor.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info("Logging configured", log_dir=str(log_path), level=log_level)
```

**Step 5: Create config.py with base settings**

```python
# src/investment_monitor/config.py
"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (optional)
    sendgrid_api_key: str = ""
    slack_webhook_url: str = ""
    anthropic_api_key: str = ""
    finnhub_api_key: str = ""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # Database
    db_path: Path = Path("data/portfolio.db")


def load_yaml_config(config_dir: Path, filename: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = config_dir / filename
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
```

**Step 6: Create __init__.py**

```python
# src/investment_monitor/__init__.py
"""Investment Monitor - Personal portfolio monitoring system."""

__version__ = "0.1.0"
```

**Step 7: Create example config files**

portfolio.yaml.example:
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

alerts.yaml.example:
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

earnings:
  enabled: true
  lookahead_days: 7

news:
  enabled: true
  min_relevance_score: 5.0
```

sources.yaml.example:
```yaml
news_feeds:
  - name: "Yahoo Finance"
    url: "https://finance.yahoo.com/rss/headline?s={ticker}"
    per_ticker: true

  - name: "Seeking Alpha"
    url: "https://seekingalpha.com/market_currents.xml"
    per_ticker: false
```

notifications.yaml.example:
```yaml
channels:
  console:
    enabled: true

  slack:
    enabled: false
    webhook_url: ""

  email:
    enabled: false
    provider: "sendgrid"
    from_address: ""
    to_address: ""
```

**Step 8: Create test file**

```python
# tests/test_config.py
"""Tests for configuration loading."""

import pytest
from investment_monitor.config import Settings, get_settings


def test_settings_loads_defaults():
    """Settings should load with default values."""
    settings = Settings()
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "phi3:mini"


def test_get_settings_returns_settings():
    """get_settings should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
```

**Step 9: Create empty test __init__.py and .gitkeep files**

```python
# tests/__init__.py
```

**Step 10: Initialize git and make first commit**

```bash
cd investment-monitor
git init
git add .
git commit -m "feat: initial project scaffolding (T01)"
```

---

### Task 2: Database Schema and Models (T02)

**Files:**
- Create: `src/investment_monitor/storage/__init__.py`
- Create: `src/investment_monitor/storage/database.py`
- Create: `src/investment_monitor/storage/models.py`
- Create: `src/investment_monitor/storage/operations.py`
- Create: `tests/test_storage.py`

**Step 1: Create database.py with engine and session management**

```python
# src/investment_monitor/storage/database.py
"""Database engine and session management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine = None
_SessionLocal = None


def init_db(db_path: str | Path = "data/portfolio.db") -> None:
    """Initialize database engine and create tables."""
    global _engine, _SessionLocal

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

**Step 2: Create models.py with SQLAlchemy ORM models**

```python
# src/investment_monitor/storage/models.py
"""SQLAlchemy ORM models for all data types."""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Price(Base):
    """Daily price and volume data."""

    __tablename__ = "prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=True)
    high: Mapped[float] = mapped_column(Float, nullable=True)
    low: Mapped[float] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
        Index("ix_price_ticker_date", "ticker", "date"),
    )


class InsiderTransaction(Base):
    """SEC Form 4 insider transactions."""

    __tablename__ = "insider_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    owner_name: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_title: Mapped[str] = mapped_column(String(100), nullable=True)
    transaction_type: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    price_per_share: Mapped[float] = mapped_column(Float, nullable=True)
    total_value: Mapped[float] = mapped_column(Float, nullable=True)
    sec_url: Mapped[str] = mapped_column(String(500), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class NewsItem(Base):
    """News articles and headlines."""

    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class AlertSent(Base):
    """Record of alerts that have been sent."""

    __tablename__ = "alerts_sent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    dedup_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)


class EarningsDate(Base):
    """Upcoming earnings dates."""

    __tablename__ = "earnings_dates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    earnings_date: Mapped[date] = mapped_column(Date, nullable=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("ticker", "earnings_date", name="uq_earnings_ticker_date"),)


class ETFHolding(Base):
    """ETF holdings data."""

    __tablename__ = "etf_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    etf_ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    holding_ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=True)
    weight_pct: Mapped[float] = mapped_column(Float, nullable=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "etf_ticker", "holding_ticker", "as_of_date", name="uq_etf_holding_date"
        ),
    )
```

**Step 3: Create operations.py with CRUD functions**

```python
# src/investment_monitor/storage/operations.py
"""CRUD operations for database models."""

from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AlertSent, EarningsDate, ETFHolding, InsiderTransaction, NewsItem, Price


# Price operations
def save_price(session: Session, price: Price) -> int:
    """Save a price record, returning its ID."""
    session.add(price)
    session.flush()
    return price.id


def save_prices(session: Session, prices: list[Price]) -> int:
    """Save multiple price records, returning count saved."""
    session.add_all(prices)
    session.flush()
    return len(prices)


def get_latest_price(session: Session, ticker: str) -> Price | None:
    """Get the most recent price for a ticker."""
    stmt = (
        select(Price)
        .where(Price.ticker == ticker)
        .order_by(Price.date.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def get_prices(
    session: Session, ticker: str, days: int = 30
) -> list[Price]:
    """Get price history for a ticker."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(Price)
        .where(Price.ticker == ticker, Price.date >= cutoff)
        .order_by(Price.date.desc())
    )
    return list(session.scalars(stmt))


def price_exists(session: Session, ticker: str, price_date: date) -> bool:
    """Check if a price record exists for ticker/date."""
    stmt = select(Price.id).where(Price.ticker == ticker, Price.date == price_date)
    return session.scalar(stmt) is not None


# Insider transaction operations
def save_insider_transaction(session: Session, txn: InsiderTransaction) -> int:
    """Save an insider transaction, returning its ID."""
    session.add(txn)
    session.flush()
    return txn.id


def get_insider_transactions(
    session: Session, ticker: str, days: int = 30
) -> list[InsiderTransaction]:
    """Get recent insider transactions for a ticker."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(InsiderTransaction)
        .where(InsiderTransaction.ticker == ticker, InsiderTransaction.trade_date >= cutoff)
        .order_by(InsiderTransaction.trade_date.desc())
    )
    return list(session.scalars(stmt))


def insider_transaction_exists(session: Session, sec_url: str) -> bool:
    """Check if an insider transaction already exists by SEC URL."""
    stmt = select(InsiderTransaction.id).where(InsiderTransaction.sec_url == sec_url)
    return session.scalar(stmt) is not None


# News operations
def save_news_item(session: Session, item: NewsItem) -> int:
    """Save a news item, returning its ID."""
    session.add(item)
    session.flush()
    return item.id


def news_exists(session: Session, url: str) -> bool:
    """Check if a news item exists by URL."""
    stmt = select(NewsItem.id).where(NewsItem.url == url)
    return session.scalar(stmt) is not None


def get_unscored_news(session: Session, limit: int = 100) -> list[NewsItem]:
    """Get news items without relevance scores."""
    stmt = (
        select(NewsItem)
        .where(NewsItem.relevance_score.is_(None))
        .order_by(NewsItem.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_recent_news(
    session: Session, ticker: str | None = None, hours: int = 24
) -> list[NewsItem]:
    """Get recent news items, optionally filtered by ticker."""
    cutoff = datetime.now() - timedelta(hours=hours)
    stmt = select(NewsItem).where(NewsItem.created_at >= cutoff)
    if ticker:
        stmt = stmt.where(NewsItem.ticker == ticker)
    stmt = stmt.order_by(NewsItem.created_at.desc())
    return list(session.scalars(stmt))


# Alert operations
def save_alert(session: Session, alert: AlertSent) -> int:
    """Save an alert record, returning its ID."""
    session.add(alert)
    session.flush()
    return alert.id


def get_recent_alerts(
    session: Session, hours: int = 24, alert_type: str | None = None
) -> list[AlertSent]:
    """Get recently sent alerts."""
    cutoff = datetime.now() - timedelta(hours=hours)
    stmt = select(AlertSent).where(AlertSent.sent_at >= cutoff)
    if alert_type:
        stmt = stmt.where(AlertSent.alert_type == alert_type)
    stmt = stmt.order_by(AlertSent.sent_at.desc())
    return list(session.scalars(stmt))


def alert_exists_by_dedup_key(
    session: Session, dedup_key: str, hours: int = 24
) -> bool:
    """Check if an alert with this dedup key was sent recently."""
    cutoff = datetime.now() - timedelta(hours=hours)
    stmt = select(AlertSent.id).where(
        AlertSent.dedup_key == dedup_key, AlertSent.sent_at >= cutoff
    )
    return session.scalar(stmt) is not None


# Earnings operations
def save_earnings_date(session: Session, earnings: EarningsDate) -> int:
    """Save an earnings date, returning its ID."""
    session.add(earnings)
    session.flush()
    return earnings.id


def get_upcoming_earnings(
    session: Session, tickers: list[str], days_ahead: int = 14
) -> list[EarningsDate]:
    """Get upcoming earnings dates for given tickers."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    stmt = (
        select(EarningsDate)
        .where(
            EarningsDate.ticker.in_(tickers),
            EarningsDate.earnings_date >= today,
            EarningsDate.earnings_date <= cutoff,
        )
        .order_by(EarningsDate.earnings_date)
    )
    return list(session.scalars(stmt))


# ETF operations
def save_etf_holdings(session: Session, holdings: list[ETFHolding]) -> int:
    """Save ETF holdings, returning count saved."""
    session.add_all(holdings)
    session.flush()
    return len(holdings)


def get_etf_holdings(
    session: Session, etf_ticker: str, as_of_date: date | None = None
) -> list[ETFHolding]:
    """Get ETF holdings for a given date (or latest)."""
    if as_of_date is None:
        # Get most recent date
        stmt = (
            select(ETFHolding.as_of_date)
            .where(ETFHolding.etf_ticker == etf_ticker)
            .order_by(ETFHolding.as_of_date.desc())
            .limit(1)
        )
        as_of_date = session.scalar(stmt)
        if as_of_date is None:
            return []

    stmt = (
        select(ETFHolding)
        .where(ETFHolding.etf_ticker == etf_ticker, ETFHolding.as_of_date == as_of_date)
        .order_by(ETFHolding.weight_pct.desc())
    )
    return list(session.scalars(stmt))
```

**Step 4: Create storage __init__.py**

```python
# src/investment_monitor/storage/__init__.py
"""Database storage module."""

from .database import get_session, init_db
from .models import (
    AlertSent,
    Base,
    EarningsDate,
    ETFHolding,
    InsiderTransaction,
    NewsItem,
    Price,
)
from .operations import (
    alert_exists_by_dedup_key,
    get_etf_holdings,
    get_insider_transactions,
    get_latest_price,
    get_prices,
    get_recent_alerts,
    get_recent_news,
    get_upcoming_earnings,
    get_unscored_news,
    insider_transaction_exists,
    news_exists,
    price_exists,
    save_alert,
    save_earnings_date,
    save_etf_holdings,
    save_insider_transaction,
    save_news_item,
    save_price,
    save_prices,
)

__all__ = [
    "init_db",
    "get_session",
    "Base",
    "Price",
    "InsiderTransaction",
    "NewsItem",
    "AlertSent",
    "EarningsDate",
    "ETFHolding",
    "save_price",
    "save_prices",
    "get_latest_price",
    "get_prices",
    "price_exists",
    "save_insider_transaction",
    "get_insider_transactions",
    "insider_transaction_exists",
    "save_news_item",
    "news_exists",
    "get_unscored_news",
    "get_recent_news",
    "save_alert",
    "get_recent_alerts",
    "alert_exists_by_dedup_key",
    "save_earnings_date",
    "get_upcoming_earnings",
    "save_etf_holdings",
    "get_etf_holdings",
]
```

**Step 5: Create tests**

```python
# tests/test_storage.py
"""Tests for database storage."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from investment_monitor.storage import (
    EarningsDate,
    InsiderTransaction,
    NewsItem,
    Price,
    get_latest_price,
    get_prices,
    get_session,
    init_db,
    news_exists,
    price_exists,
    save_earnings_date,
    save_insider_transaction,
    save_news_item,
    save_price,
)


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


def test_save_and_get_price(db_session):
    """Test saving and retrieving prices."""
    price = Price(
        ticker="AAPL",
        date=date(2026, 1, 28),
        open=180.0,
        high=182.0,
        low=179.0,
        close=181.0,
        volume=50000000,
    )
    save_price(db_session, price)

    retrieved = get_latest_price(db_session, "AAPL")
    assert retrieved is not None
    assert retrieved.ticker == "AAPL"
    assert retrieved.close == 181.0


def test_price_exists(db_session):
    """Test price existence check."""
    price = Price(ticker="MSFT", date=date(2026, 1, 28), close=420.0)
    save_price(db_session, price)

    assert price_exists(db_session, "MSFT", date(2026, 1, 28))
    assert not price_exists(db_session, "MSFT", date(2026, 1, 27))


def test_get_prices_history(db_session):
    """Test getting price history."""
    for i in range(5):
        price = Price(
            ticker="AAPL",
            date=date(2026, 1, 28 - i),
            close=180.0 + i,
        )
        save_price(db_session, price)

    prices = get_prices(db_session, "AAPL", days=10)
    assert len(prices) == 5


def test_save_news_item(db_session):
    """Test saving news items."""
    item = NewsItem(
        ticker="AAPL",
        headline="Apple announces new product",
        source="Reuters",
        url="https://example.com/news/1",
        published_at=datetime.now(),
    )
    save_news_item(db_session, item)

    assert news_exists(db_session, "https://example.com/news/1")
    assert not news_exists(db_session, "https://example.com/news/2")


def test_save_insider_transaction(db_session):
    """Test saving insider transactions."""
    txn = InsiderTransaction(
        ticker="AAPL",
        filing_date=date(2026, 1, 28),
        trade_date=date(2026, 1, 27),
        owner_name="Tim Cook",
        owner_title="CEO",
        transaction_type="S",
        shares=50000,
        price_per_share=180.0,
        total_value=9000000.0,
        sec_url="https://sec.gov/filing/123",
    )
    save_insider_transaction(db_session, txn)
    # Should not raise


def test_save_earnings_date(db_session):
    """Test saving earnings dates."""
    earnings = EarningsDate(
        ticker="AAPL",
        earnings_date=date(2026, 2, 2),
        confirmed=True,
    )
    save_earnings_date(db_session, earnings)
    # Should not raise
```

**Step 6: Commit**

```bash
git add src/investment_monitor/storage tests/test_storage.py
git commit -m "feat: add database schema and CRUD operations (T02)"
```

---

### Task 3: Portfolio Configuration Models (T03)

**Files:**
- Create: `src/investment_monitor/models/__init__.py`
- Create: `src/investment_monitor/models/portfolio.py`
- Create: `tests/test_models.py`

**Step 1: Create portfolio.py**

```python
# src/investment_monitor/models/portfolio.py
"""Portfolio and holding models."""

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, computed_field


class Holding(BaseModel):
    """A single stock holding."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    shares: Decimal = Field(..., gt=0)
    cost_basis: Decimal = Field(..., gt=0)
    thesis: str = Field(default="", max_length=500)

    @computed_field
    @property
    def total_cost(self) -> Decimal:
        """Total cost basis for this holding."""
        return self.shares * self.cost_basis


class WatchlistItem(BaseModel):
    """A stock on the watchlist."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}$")
    reason: str = Field(default="")
    target_price: Decimal | None = None


class Portfolio(BaseModel):
    """Portfolio configuration with holdings and watchlist."""

    holdings: list[Holding] = Field(default_factory=list)
    watchlist: list[WatchlistItem] = Field(default_factory=list)

    @computed_field
    @property
    def all_tickers(self) -> list[str]:
        """All tickers to monitor (holdings + watchlist, deduplicated)."""
        tickers = set()
        tickers.update(h.ticker for h in self.holdings)
        tickers.update(w.ticker for w in self.watchlist)
        return sorted(tickers)

    @computed_field
    @property
    def holding_tickers(self) -> list[str]:
        """Just the holding tickers."""
        return [h.ticker for h in self.holdings]

    def get_holding(self, ticker: str) -> Holding | None:
        """Get a holding by ticker."""
        for h in self.holdings:
            if h.ticker == ticker:
                return h
        return None

    def get_thesis(self, ticker: str) -> str | None:
        """Get investment thesis for a ticker."""
        holding = self.get_holding(ticker)
        if holding:
            return holding.thesis if holding.thesis else None
        return None

    def get_cost_basis(self, ticker: str) -> Decimal | None:
        """Get cost basis for a ticker."""
        holding = self.get_holding(ticker)
        return holding.cost_basis if holding else None

    @classmethod
    def from_yaml(cls, path: Path) -> "Portfolio":
        """Load portfolio from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
```

**Step 2: Create models __init__.py**

```python
# src/investment_monitor/models/__init__.py
"""Data models module."""

from .portfolio import Holding, Portfolio, WatchlistItem

__all__ = ["Holding", "WatchlistItem", "Portfolio"]
```

**Step 3: Create tests**

```python
# tests/test_models.py
"""Tests for data models."""

from decimal import Decimal
from pathlib import Path
import tempfile

import pytest

from investment_monitor.models import Holding, Portfolio, WatchlistItem


def test_holding_validation():
    """Test holding validation."""
    holding = Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00"))
    assert holding.ticker == "AAPL"
    assert holding.total_cost == Decimal("8250.00")


def test_holding_invalid_ticker():
    """Test that invalid tickers are rejected."""
    with pytest.raises(ValueError):
        Holding(ticker="aapl", shares=Decimal("50"), cost_basis=Decimal("165.00"))

    with pytest.raises(ValueError):
        Holding(ticker="TOOLONG", shares=Decimal("50"), cost_basis=Decimal("165.00"))


def test_holding_invalid_shares():
    """Test that invalid shares are rejected."""
    with pytest.raises(ValueError):
        Holding(ticker="AAPL", shares=Decimal("0"), cost_basis=Decimal("165.00"))

    with pytest.raises(ValueError):
        Holding(ticker="AAPL", shares=Decimal("-10"), cost_basis=Decimal("165.00"))


def test_watchlist_item():
    """Test watchlist item."""
    item = WatchlistItem(
        ticker="GOOGL", reason="Waiting for entry", target_price=Decimal("140.00")
    )
    assert item.ticker == "GOOGL"
    assert item.target_price == Decimal("140.00")


def test_portfolio_all_tickers():
    """Test all_tickers computed property."""
    portfolio = Portfolio(
        holdings=[
            Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00")),
            Holding(ticker="MSFT", shares=Decimal("30"), cost_basis=Decimal("380.00")),
        ],
        watchlist=[
            WatchlistItem(ticker="GOOGL", reason="Watching"),
            WatchlistItem(ticker="AAPL", reason="Already own"),  # Duplicate
        ],
    )

    assert portfolio.all_tickers == ["AAPL", "GOOGL", "MSFT"]
    assert portfolio.holding_tickers == ["AAPL", "MSFT"]


def test_portfolio_get_holding():
    """Test get_holding method."""
    portfolio = Portfolio(
        holdings=[
            Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00")),
        ]
    )

    holding = portfolio.get_holding("AAPL")
    assert holding is not None
    assert holding.shares == Decimal("50")

    assert portfolio.get_holding("MSFT") is None


def test_portfolio_get_thesis():
    """Test get_thesis method."""
    portfolio = Portfolio(
        holdings=[
            Holding(
                ticker="AAPL",
                shares=Decimal("50"),
                cost_basis=Decimal("165.00"),
                thesis="Services growth",
            ),
            Holding(ticker="MSFT", shares=Decimal("30"), cost_basis=Decimal("380.00")),
        ]
    )

    assert portfolio.get_thesis("AAPL") == "Services growth"
    assert portfolio.get_thesis("MSFT") is None  # Empty thesis
    assert portfolio.get_thesis("GOOGL") is None  # Not in portfolio


def test_portfolio_from_yaml():
    """Test loading portfolio from YAML."""
    yaml_content = """
holdings:
  - ticker: AAPL
    shares: 50
    cost_basis: 165.00
    thesis: "Services growth"

watchlist:
  - ticker: GOOGL
    reason: "Waiting for entry"
    target_price: 140.00
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        portfolio = Portfolio.from_yaml(Path(f.name))
        assert len(portfolio.holdings) == 1
        assert len(portfolio.watchlist) == 1
        assert portfolio.holdings[0].ticker == "AAPL"
```

**Step 4: Commit**

```bash
git add src/investment_monitor/models tests/test_models.py
git commit -m "feat: add portfolio configuration models (T03)"
```

---

### Task 4: Alert Configuration Models (T04)

**Files:**
- Modify: `src/investment_monitor/models/__init__.py`
- Create: `src/investment_monitor/models/alerts.py`
- Modify: `tests/test_models.py`

**Step 1: Create alerts.py**

```python
# src/investment_monitor/models/alerts.py
"""Alert configuration models."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PriceAlertSettings(BaseModel):
    """Price movement alert settings."""

    enabled: bool = True
    daily_drop_pct: float = Field(default=3.0, ge=0, le=100)
    daily_rise_pct: float = Field(default=5.0, ge=0, le=100)
    weekly_drop_pct: float = Field(default=7.0, ge=0, le=100)
    below_cost_basis: bool = True


class VolumeAlertSettings(BaseModel):
    """Volume spike alert settings."""

    enabled: bool = True
    lookback_days: int = Field(default=20, ge=5, le=60)
    multiplier: float = Field(default=2.5, ge=1.0)


class InsiderAlertSettings(BaseModel):
    """Insider transaction alert settings."""

    enabled: bool = True
    min_buy_value: int = Field(default=100_000, ge=0)
    min_sell_value: int = Field(default=500_000, ge=0)
    alert_ceo_cfo_any: bool = True
    cluster_threshold: int = Field(default=3, ge=2)
    cluster_days: int = Field(default=7, ge=1)


class EarningsAlertSettings(BaseModel):
    """Earnings calendar alert settings."""

    enabled: bool = True
    lookahead_days: int = Field(default=7, ge=1, le=30)


class NewsAlertSettings(BaseModel):
    """News alert settings."""

    enabled: bool = True
    keywords: list[str] = Field(
        default_factory=lambda: [
            "lawsuit",
            "SEC",
            "investigation",
            "guidance",
            "acquisition",
            "merger",
            "layoffs",
            "dividend",
            "buyback",
        ]
    )
    min_relevance_score: float = Field(default=5.0, ge=0, le=10)


class ETFAlertSettings(BaseModel):
    """ETF holdings alert settings."""

    enabled: bool = True
    holdings_change: bool = True
    weight_change_pct: float = Field(default=1.0, ge=0)


class AlertsConfig(BaseModel):
    """All alert configuration settings."""

    price: PriceAlertSettings = Field(default_factory=PriceAlertSettings)
    volume: VolumeAlertSettings = Field(default_factory=VolumeAlertSettings)
    insider: InsiderAlertSettings = Field(default_factory=InsiderAlertSettings)
    earnings: EarningsAlertSettings = Field(default_factory=EarningsAlertSettings)
    news: NewsAlertSettings = Field(default_factory=NewsAlertSettings)
    etf: ETFAlertSettings = Field(default_factory=ETFAlertSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "AlertsConfig":
        """Load alerts config from YAML file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
```

**Step 2: Update models __init__.py**

```python
# src/investment_monitor/models/__init__.py
"""Data models module."""

from .alerts import (
    AlertsConfig,
    EarningsAlertSettings,
    ETFAlertSettings,
    InsiderAlertSettings,
    NewsAlertSettings,
    PriceAlertSettings,
    VolumeAlertSettings,
)
from .portfolio import Holding, Portfolio, WatchlistItem

__all__ = [
    "Holding",
    "WatchlistItem",
    "Portfolio",
    "PriceAlertSettings",
    "VolumeAlertSettings",
    "InsiderAlertSettings",
    "EarningsAlertSettings",
    "NewsAlertSettings",
    "ETFAlertSettings",
    "AlertsConfig",
]
```

**Step 3: Add tests to test_models.py**

```python
# Add to tests/test_models.py

from investment_monitor.models import (
    AlertsConfig,
    PriceAlertSettings,
    VolumeAlertSettings,
)


def test_price_alert_settings_defaults():
    """Test price alert settings defaults."""
    settings = PriceAlertSettings()
    assert settings.enabled is True
    assert settings.daily_drop_pct == 3.0
    assert settings.daily_rise_pct == 5.0


def test_price_alert_settings_validation():
    """Test price alert settings validation."""
    with pytest.raises(ValueError):
        PriceAlertSettings(daily_drop_pct=-1.0)

    with pytest.raises(ValueError):
        PriceAlertSettings(daily_drop_pct=101.0)


def test_volume_alert_settings():
    """Test volume alert settings."""
    settings = VolumeAlertSettings(lookback_days=30, multiplier=3.0)
    assert settings.lookback_days == 30
    assert settings.multiplier == 3.0


def test_alerts_config_defaults():
    """Test alerts config with all defaults."""
    config = AlertsConfig()
    assert config.price.enabled is True
    assert config.volume.enabled is True
    assert config.insider.enabled is True


def test_alerts_config_from_yaml():
    """Test loading alerts config from YAML."""
    yaml_content = """
price:
  enabled: true
  daily_drop_pct: 5.0

volume:
  enabled: false
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = AlertsConfig.from_yaml(Path(f.name))
        assert config.price.daily_drop_pct == 5.0
        assert config.volume.enabled is False
        # Defaults still apply for unset fields
        assert config.insider.enabled is True
```

**Step 4: Commit**

```bash
git add src/investment_monitor/models tests/test_models.py
git commit -m "feat: add alert configuration models (T04)"
```

---

### Task 5: Notification System Base (T05)

**Files:**
- Create: `src/investment_monitor/notifications/__init__.py`
- Create: `src/investment_monitor/notifications/base.py`
- Create: `src/investment_monitor/notifications/console.py`
- Create: `src/investment_monitor/notifications/manager.py`
- Create: `tests/test_notifications.py`

**Step 1: Create base.py with abstract classes**

```python
# src/investment_monitor/notifications/base.py
"""Base classes for notification system."""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class Priority(str, Enum):
    """Alert priority levels."""

    HIGH = "high"  # Send immediately
    MEDIUM = "medium"  # Include in next digest
    LOW = "low"  # Log only


class AlertMessage(BaseModel):
    """A message to be sent via notification channels."""

    title: str
    body: str
    ticker: str | None = None
    alert_type: str
    priority: Priority = Priority.MEDIUM
    url: str | None = None

    def __str__(self) -> str:
        """String representation for logging."""
        prefix = f"[{self.ticker}] " if self.ticker else ""
        return f"{prefix}{self.title}: {self.body}"


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    name: str = "base"

    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """
        Send a single message.

        Returns True if successful.
        """
        ...

    @abstractmethod
    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """
        Send a batch of messages as a digest.

        Returns True if successful.
        """
        ...
```

**Step 2: Create console.py**

```python
# src/investment_monitor/notifications/console.py
"""Console notification channel using loguru."""

from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority


class ConsoleChannel(NotificationChannel):
    """Notification channel that logs to console."""

    name = "console"

    async def send(self, message: AlertMessage) -> bool:
        """Log a message to console."""
        level = self._get_log_level(message.priority)
        ticker_prefix = f"[{message.ticker}] " if message.ticker else ""

        logger.log(
            level,
            f"ALERT ({message.alert_type}) {ticker_prefix}{message.title}",
        )
        logger.log(level, f"  {message.body}")
        if message.url:
            logger.log(level, f"  Link: {message.url}")

        return True

    async def send_digest(self, messages: list[AlertMessage]) -> bool:
        """Log a digest to console."""
        if not messages:
            logger.info("DIGEST: No alerts to report")
            return True

        logger.info("=" * 60)
        logger.info("INVESTMENT MONITOR DIGEST")
        logger.info("=" * 60)

        # Group by alert type
        by_type: dict[str, list[AlertMessage]] = {}
        for msg in messages:
            by_type.setdefault(msg.alert_type, []).append(msg)

        for alert_type, type_messages in by_type.items():
            logger.info(f"\n{alert_type.upper()} ({len(type_messages)} alerts)")
            logger.info("-" * 40)
            for msg in type_messages:
                ticker_prefix = f"[{msg.ticker}] " if msg.ticker else ""
                logger.info(f"  {ticker_prefix}{msg.title}")
                logger.info(f"    {msg.body}")

        logger.info("=" * 60)
        return True

    def _get_log_level(self, priority: Priority) -> str:
        """Map priority to log level."""
        return {
            Priority.HIGH: "WARNING",
            Priority.MEDIUM: "INFO",
            Priority.LOW: "DEBUG",
        }.get(priority, "INFO")
```

**Step 3: Create manager.py**

```python
# src/investment_monitor/notifications/manager.py
"""Notification manager for routing messages."""

from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority


class NotificationManager:
    """Routes messages to appropriate channels based on priority."""

    def __init__(self, channels: list[NotificationChannel]):
        """Initialize with list of channels."""
        self.channels = channels
        self._pending_digest: list[AlertMessage] = []

    async def notify(self, message: AlertMessage) -> None:
        """
        Send notification via configured channels.

        HIGH priority: Send immediately
        MEDIUM priority: Queue for digest
        LOW priority: Log only
        """
        if message.priority == Priority.HIGH:
            await self._send_immediate(message)
        elif message.priority == Priority.MEDIUM:
            self._pending_digest.append(message)
            logger.debug(f"Queued for digest: {message.title}")
        else:
            logger.debug(f"Low priority alert: {message}")

    async def _send_immediate(self, message: AlertMessage) -> None:
        """Send message immediately to all channels."""
        for channel in self.channels:
            try:
                success = await channel.send(message)
                if success:
                    logger.debug(f"Sent via {channel.name}: {message.title}")
                else:
                    logger.warning(f"Failed to send via {channel.name}: {message.title}")
            except Exception as e:
                logger.error(f"Error sending via {channel.name}: {e}")

    async def send_digest(self) -> None:
        """Send accumulated messages as a digest."""
        if not self._pending_digest:
            logger.info("No messages pending for digest")
            return

        for channel in self.channels:
            try:
                success = await channel.send_digest(self._pending_digest)
                if success:
                    logger.info(
                        f"Sent digest via {channel.name} "
                        f"({len(self._pending_digest)} messages)"
                    )
                else:
                    logger.warning(f"Failed to send digest via {channel.name}")
            except Exception as e:
                logger.error(f"Error sending digest via {channel.name}: {e}")

        self._pending_digest.clear()

    def get_pending_count(self) -> int:
        """Get number of messages pending for digest."""
        return len(self._pending_digest)
```

**Step 4: Create notifications __init__.py**

```python
# src/investment_monitor/notifications/__init__.py
"""Notification system module."""

from .base import AlertMessage, NotificationChannel, Priority
from .console import ConsoleChannel
from .manager import NotificationManager

__all__ = [
    "Priority",
    "AlertMessage",
    "NotificationChannel",
    "ConsoleChannel",
    "NotificationManager",
]
```

**Step 5: Create tests**

```python
# tests/test_notifications.py
"""Tests for notification system."""

import pytest

from investment_monitor.notifications import (
    AlertMessage,
    ConsoleChannel,
    NotificationManager,
    Priority,
)


@pytest.fixture
def console_channel():
    """Create a console channel."""
    return ConsoleChannel()


@pytest.fixture
def notification_manager(console_channel):
    """Create a notification manager with console channel."""
    return NotificationManager([console_channel])


def test_alert_message_creation():
    """Test creating an alert message."""
    msg = AlertMessage(
        title="Price Alert",
        body="AAPL dropped 5%",
        ticker="AAPL",
        alert_type="price_drop",
        priority=Priority.HIGH,
    )
    assert msg.title == "Price Alert"
    assert msg.priority == Priority.HIGH


def test_alert_message_str():
    """Test alert message string representation."""
    msg = AlertMessage(
        title="Test",
        body="Test body",
        ticker="AAPL",
        alert_type="test",
    )
    assert "[AAPL]" in str(msg)
    assert "Test" in str(msg)


@pytest.mark.asyncio
async def test_console_channel_send(console_channel):
    """Test console channel sends successfully."""
    msg = AlertMessage(
        title="Test Alert",
        body="Test body",
        ticker="AAPL",
        alert_type="test",
        priority=Priority.HIGH,
    )
    result = await console_channel.send(msg)
    assert result is True


@pytest.mark.asyncio
async def test_console_channel_digest(console_channel):
    """Test console channel sends digest."""
    messages = [
        AlertMessage(
            title=f"Alert {i}",
            body=f"Body {i}",
            alert_type="test",
        )
        for i in range(3)
    ]
    result = await console_channel.send_digest(messages)
    assert result is True


@pytest.mark.asyncio
async def test_notification_manager_high_priority(notification_manager):
    """Test high priority messages are sent immediately."""
    msg = AlertMessage(
        title="Urgent",
        body="Urgent body",
        alert_type="test",
        priority=Priority.HIGH,
    )
    await notification_manager.notify(msg)
    # Should be sent immediately, not queued
    assert notification_manager.get_pending_count() == 0


@pytest.mark.asyncio
async def test_notification_manager_medium_priority(notification_manager):
    """Test medium priority messages are queued."""
    msg = AlertMessage(
        title="Normal",
        body="Normal body",
        alert_type="test",
        priority=Priority.MEDIUM,
    )
    await notification_manager.notify(msg)
    assert notification_manager.get_pending_count() == 1


@pytest.mark.asyncio
async def test_notification_manager_digest(notification_manager):
    """Test sending digest clears queue."""
    for i in range(3):
        msg = AlertMessage(
            title=f"Alert {i}",
            body=f"Body {i}",
            alert_type="test",
            priority=Priority.MEDIUM,
        )
        await notification_manager.notify(msg)

    assert notification_manager.get_pending_count() == 3
    await notification_manager.send_digest()
    assert notification_manager.get_pending_count() == 0
```

**Step 6: Commit**

```bash
git add src/investment_monitor/notifications tests/test_notifications.py
git commit -m "feat: add notification system with console channel (T05)"
```

---

### Task 6: Digest Formatter (T06)

**Files:**
- Create: `src/investment_monitor/notifications/digest.py`
- Modify: `src/investment_monitor/notifications/__init__.py`
- Modify: `tests/test_notifications.py`

**Step 1: Create digest.py**

```python
# src/investment_monitor/notifications/digest.py
"""Digest formatting for daily and weekly summaries."""

from datetime import date

from investment_monitor.models import Portfolio

from .base import AlertMessage


def format_daily_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio | None = None,
    digest_date: date | None = None,
) -> tuple[str, str]:
    """
    Format messages into a daily digest.

    Returns:
        tuple of (plain_text, html)
    """
    if digest_date is None:
        digest_date = date.today()

    # Group messages by type
    by_type: dict[str, list[AlertMessage]] = {}
    for msg in messages:
        by_type.setdefault(msg.alert_type, []).append(msg)

    # Build plain text version
    lines = [
        "=" * 60,
        "INVESTMENT MONITOR DAILY DIGEST",
        digest_date.strftime("%B %d, %Y"),
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 7,
    ]

    # Summary counts
    for alert_type, type_messages in sorted(by_type.items()):
        lines.append(f"* {len(type_messages)} {alert_type.replace('_', ' ')} alerts")

    if not messages:
        lines.append("* No alerts today")

    # Detail sections
    for alert_type, type_messages in sorted(by_type.items()):
        lines.extend(["", alert_type.upper().replace("_", " "), "-" * len(alert_type)])

        for msg in type_messages:
            ticker_prefix = f"[{msg.ticker}] " if msg.ticker else ""
            lines.append(f"{ticker_prefix}{msg.title}")
            lines.append(f"  {msg.body}")
            if msg.url:
                lines.append(f"  Link: {msg.url}")
            lines.append("")

    lines.extend([
        "-" * 60,
        "Generated by Investment Monitor",
    ])

    plain_text = "\n".join(lines)

    # Build HTML version
    html = _format_html_digest(messages, by_type, digest_date)

    return plain_text, html


def format_weekly_digest(
    messages: list[AlertMessage],
    portfolio: Portfolio | None = None,
    week_start: date | None = None,
    week_end: date | None = None,
    ai_synthesis: str | None = None,
) -> tuple[str, str]:
    """
    Format messages into weekly digest with optional AI synthesis.

    Returns:
        tuple of (plain_text, html)
    """
    if week_end is None:
        week_end = date.today()
    if week_start is None:
        from datetime import timedelta
        week_start = week_end - timedelta(days=7)

    # Group messages by type
    by_type: dict[str, list[AlertMessage]] = {}
    for msg in messages:
        by_type.setdefault(msg.alert_type, []).append(msg)

    lines = [
        "=" * 60,
        "INVESTMENT MONITOR WEEKLY DIGEST",
        f"{week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}",
        "=" * 60,
    ]

    # AI synthesis if provided
    if ai_synthesis:
        lines.extend([
            "",
            "AI ANALYSIS",
            "-" * 11,
            ai_synthesis,
        ])

    # Summary
    lines.extend(["", "WEEKLY SUMMARY", "-" * 14])

    for alert_type, type_messages in sorted(by_type.items()):
        lines.append(f"* {len(type_messages)} {alert_type.replace('_', ' ')} alerts")

    if not messages:
        lines.append("* No significant events this week")

    # Condensed details (just titles)
    for alert_type, type_messages in sorted(by_type.items()):
        lines.extend(["", alert_type.upper().replace("_", " ")])
        for msg in type_messages[:5]:  # Limit to 5 per type for weekly
            ticker_prefix = f"[{msg.ticker}] " if msg.ticker else ""
            lines.append(f"  - {ticker_prefix}{msg.title}")
        if len(type_messages) > 5:
            lines.append(f"  ... and {len(type_messages) - 5} more")

    lines.extend([
        "",
        "-" * 60,
        "Generated by Investment Monitor",
    ])

    plain_text = "\n".join(lines)
    html = _format_html_weekly(messages, by_type, week_start, week_end, ai_synthesis)

    return plain_text, html


def _format_html_digest(
    messages: list[AlertMessage],
    by_type: dict[str, list[AlertMessage]],
    digest_date: date,
) -> str:
    """Format HTML version of daily digest."""
    sections = []

    for alert_type, type_messages in sorted(by_type.items()):
        items = []
        for msg in type_messages:
            ticker = f"<strong>[{msg.ticker}]</strong> " if msg.ticker else ""
            link = f' <a href="{msg.url}">[link]</a>' if msg.url else ""
            items.append(f"<li>{ticker}{msg.title}{link}<br><small>{msg.body}</small></li>")

        sections.append(f"""
        <h3>{alert_type.replace('_', ' ').title()}</h3>
        <ul>{''.join(items)}</ul>
        """)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            h3 {{ color: #666; border-bottom: 1px solid #ddd; }}
            ul {{ list-style-type: none; padding-left: 0; }}
            li {{ margin-bottom: 10px; padding: 10px; background: #f9f9f9; }}
            small {{ color: #666; }}
        </style>
    </head>
    <body>
        <h1>Investment Monitor Daily Digest</h1>
        <p>{digest_date.strftime('%B %d, %Y')}</p>
        {''.join(sections) if sections else '<p>No alerts today.</p>'}
        <hr>
        <small>Generated by Investment Monitor</small>
    </body>
    </html>
    """


def _format_html_weekly(
    messages: list[AlertMessage],
    by_type: dict[str, list[AlertMessage]],
    week_start: date,
    week_end: date,
    ai_synthesis: str | None,
) -> str:
    """Format HTML version of weekly digest."""
    synthesis_html = ""
    if ai_synthesis:
        synthesis_html = f"""
        <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3>AI Analysis</h3>
            <p>{ai_synthesis}</p>
        </div>
        """

    sections = []
    for alert_type, type_messages in sorted(by_type.items()):
        items = [f"<li>[{msg.ticker}] {msg.title}</li>" if msg.ticker else f"<li>{msg.title}</li>"
                 for msg in type_messages[:5]]
        if len(type_messages) > 5:
            items.append(f"<li><em>... and {len(type_messages) - 5} more</em></li>")
        sections.append(f"<h3>{alert_type.replace('_', ' ').title()}</h3><ul>{''.join(items)}</ul>")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
            h1 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Investment Monitor Weekly Digest</h1>
        <p>{week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}</p>
        {synthesis_html}
        {''.join(sections) if sections else '<p>No significant events this week.</p>'}
        <hr>
        <small>Generated by Investment Monitor</small>
    </body>
    </html>
    """
```

**Step 2: Update notifications __init__.py**

```python
# src/investment_monitor/notifications/__init__.py
"""Notification system module."""

from .base import AlertMessage, NotificationChannel, Priority
from .console import ConsoleChannel
from .digest import format_daily_digest, format_weekly_digest
from .manager import NotificationManager

__all__ = [
    "Priority",
    "AlertMessage",
    "NotificationChannel",
    "ConsoleChannel",
    "NotificationManager",
    "format_daily_digest",
    "format_weekly_digest",
]
```

**Step 3: Add tests**

```python
# Add to tests/test_notifications.py

from datetime import date

from investment_monitor.notifications import format_daily_digest, format_weekly_digest


def test_format_daily_digest_empty():
    """Test formatting empty digest."""
    plain, html = format_daily_digest([], digest_date=date(2026, 1, 28))
    assert "DAILY DIGEST" in plain
    assert "January 28, 2026" in plain
    assert "No alerts today" in plain
    assert "<html>" in html


def test_format_daily_digest_with_messages():
    """Test formatting digest with messages."""
    messages = [
        AlertMessage(
            title="AAPL dropped 5%",
            body="Apple stock fell sharply",
            ticker="AAPL",
            alert_type="price_drop",
        ),
        AlertMessage(
            title="Insider sale",
            body="CEO sold shares",
            ticker="AAPL",
            alert_type="insider",
        ),
    ]
    plain, html = format_daily_digest(messages, digest_date=date(2026, 1, 28))

    assert "price drop" in plain.lower()
    assert "insider" in plain.lower()
    assert "AAPL" in plain


def test_format_weekly_digest_with_synthesis():
    """Test weekly digest with AI synthesis."""
    messages = [
        AlertMessage(
            title="Weekly price movement",
            body="Various movements",
            alert_type="price",
        ),
    ]
    plain, html = format_weekly_digest(
        messages,
        week_start=date(2026, 1, 21),
        week_end=date(2026, 1, 28),
        ai_synthesis="The market showed strength this week.",
    )

    assert "WEEKLY DIGEST" in plain
    assert "AI ANALYSIS" in plain
    assert "market showed strength" in plain
```

**Step 4: Commit**

```bash
git add src/investment_monitor/notifications tests/test_notifications.py
git commit -m "feat: add digest formatter (T06)"
```

---

## Phase 2: Data Collectors

### Task 7: Collector Base Class (T07)

**Files:**
- Create: `src/investment_monitor/collectors/__init__.py`
- Create: `src/investment_monitor/collectors/base.py`
- Create: `tests/test_collectors.py`

**Step 1: Create base.py**

```python
# src/investment_monitor/collectors/base.py
"""Base class for data collectors."""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from investment_monitor.config import Settings


class CollectorResult(BaseModel):
    """Result of a collector run."""

    collector_name: str
    success: bool
    records_collected: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        """Duration of collection in seconds."""
        return (self.finished_at - self.started_at).total_seconds()


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    name: str = "base"
    rate_limit_calls: int = 60  # Calls per period
    rate_limit_period: int = 60  # Period in seconds
    max_retries: int = 3
    retry_delay: float = 1.0

    def __init__(self, session: Session, config: Settings):
        """Initialize collector with database session and config."""
        self.session = session
        self.config = config
        self._call_times: list[float] = []

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        # Remove old call times outside the period
        self._call_times = [
            t for t in self._call_times
            if now - t < self.rate_limit_period
        ]

        if len(self._call_times) >= self.rate_limit_calls:
            # Wait until oldest call is outside the period
            sleep_time = self.rate_limit_period - (now - self._call_times[0])
            if sleep_time > 0:
                logger.debug(f"{self.name}: Rate limit reached, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        self._call_times.append(time.time())

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"{self.name}: Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        raise last_error

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
        Must be implemented by subclasses.
        """
        ...

    async def run(self, tickers: list[str]) -> CollectorResult:
        """Run the collector with error handling and timing."""
        started_at = datetime.now()
        errors: list[str] = []
        total_records = 0

        try:
            result = await self.collect(tickers)
            return result
        except Exception as e:
            logger.error(f"{self.name}: Collection failed: {e}")
            errors.append(str(e))
            return CollectorResult(
                collector_name=self.name,
                success=False,
                records_collected=total_records,
                errors=errors,
                started_at=started_at,
                finished_at=datetime.now(),
            )
```

**Step 2: Create collectors __init__.py**

```python
# src/investment_monitor/collectors/__init__.py
"""Data collectors module."""

from .base import BaseCollector, CollectorResult

__all__ = ["BaseCollector", "CollectorResult"]
```

**Step 3: Create tests**

```python
# tests/test_collectors.py
"""Tests for data collectors."""

import asyncio
from datetime import datetime

import pytest

from investment_monitor.collectors import BaseCollector, CollectorResult
from investment_monitor.config import Settings


class MockCollector(BaseCollector):
    """Mock collector for testing base class."""

    name = "mock"
    rate_limit_calls = 3
    rate_limit_period = 1

    def __init__(self):
        # Don't call super().__init__ to avoid needing real session
        self._call_times = []
        self.collect_count = 0

    async def collect(self, tickers: list[str]) -> CollectorResult:
        started_at = datetime.now()
        total = 0
        for ticker in tickers:
            total += await self.collect_single(ticker)
        return CollectorResult(
            collector_name=self.name,
            success=True,
            records_collected=total,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def collect_single(self, ticker: str) -> int:
        self.collect_count += 1
        return 1


def test_collector_result():
    """Test CollectorResult model."""
    started = datetime(2026, 1, 28, 10, 0, 0)
    finished = datetime(2026, 1, 28, 10, 0, 5)

    result = CollectorResult(
        collector_name="test",
        success=True,
        records_collected=10,
        started_at=started,
        finished_at=finished,
    )

    assert result.duration_seconds == 5.0
    assert result.success


@pytest.mark.asyncio
async def test_mock_collector():
    """Test mock collector collects all tickers."""
    collector = MockCollector()
    result = await collector.collect(["AAPL", "MSFT", "GOOGL"])

    assert result.success
    assert result.records_collected == 3
    assert collector.collect_count == 3


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting works."""
    collector = MockCollector()

    # Make 5 calls quickly - should trigger rate limiting
    start = datetime.now()
    for _ in range(5):
        await collector._rate_limit()

    elapsed = (datetime.now() - start).total_seconds()
    # With rate limit of 3 calls per second, 5 calls should take >1 second
    assert elapsed >= 1.0
```

**Step 4: Commit**

```bash
git add src/investment_monitor/collectors tests/test_collectors.py
git commit -m "feat: add collector base class with rate limiting (T07)"
```

---

### Task 8: Price Collector (T08)

**Files:**
- Create: `src/investment_monitor/collectors/prices.py`
- Modify: `src/investment_monitor/collectors/__init__.py`
- Add to: `tests/test_collectors.py`

**Step 1: Create prices.py**

```python
# src/investment_monitor/collectors/prices.py
"""Price data collector using yfinance."""

from datetime import date, datetime, timedelta

import yfinance as yf
from loguru import logger
from sqlalchemy.orm import Session

from investment_monitor.config import Settings
from investment_monitor.storage import Price, get_prices, price_exists, save_price

from .base import BaseCollector, CollectorResult


class PriceCollector(BaseCollector):
    """Collector for daily price and volume data."""

    name = "prices"
    rate_limit_calls = 30
    rate_limit_period = 60

    async def collect(self, tickers: list[str]) -> CollectorResult:
        """Fetch prices for all tickers using batch request."""
        started_at = datetime.now()
        errors: list[str] = []
        total_records = 0

        if not tickers:
            return CollectorResult(
                collector_name=self.name,
                success=True,
                records_collected=0,
                started_at=started_at,
                finished_at=datetime.now(),
            )

        try:
            await self._rate_limit()

            # yfinance supports batch downloads
            logger.info(f"Fetching prices for {len(tickers)} tickers")
            data = yf.download(
                tickers=tickers,
                period="5d",
                interval="1d",
                progress=False,
                threads=True,
            )

            if data.empty:
                logger.warning("No price data returned")
                return CollectorResult(
                    collector_name=self.name,
                    success=True,
                    records_collected=0,
                    started_at=started_at,
                    finished_at=datetime.now(),
                )

            # Handle single vs multiple tickers (yfinance returns different structure)
            if len(tickers) == 1:
                total_records = self._save_single_ticker_data(tickers[0], data)
            else:
                for ticker in tickers:
                    try:
                        ticker_records = self._save_ticker_data(ticker, data)
                        total_records += ticker_records
                    except Exception as e:
                        logger.error(f"Error saving {ticker}: {e}")
                        errors.append(f"{ticker}: {e}")

        except Exception as e:
            logger.error(f"Price collection failed: {e}")
            errors.append(str(e))

        return CollectorResult(
            collector_name=self.name,
            success=len(errors) == 0,
            records_collected=total_records,
            errors=errors,
            started_at=started_at,
            finished_at=datetime.now(),
        )

    async def collect_single(self, ticker: str) -> int:
        """Fetch price for a single ticker."""
        await self._rate_limit()

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")

            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return 0

            return self._save_single_ticker_data(ticker, hist)
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return 0

    def _save_single_ticker_data(self, ticker: str, data) -> int:
        """Save price data for a single ticker."""
        saved = 0
        for idx, row in data.iterrows():
            price_date = idx.date() if hasattr(idx, "date") else idx

            if price_exists(self.session, ticker, price_date):
                continue

            price = Price(
                ticker=ticker,
                date=price_date,
                open=float(row.get("Open", 0)) if row.get("Open") else None,
                high=float(row.get("High", 0)) if row.get("High") else None,
                low=float(row.get("Low", 0)) if row.get("Low") else None,
                close=float(row["Close"]),
                volume=int(row.get("Volume", 0)) if row.get("Volume") else None,
            )
            save_price(self.session, price)
            saved += 1

        return saved

    def _save_ticker_data(self, ticker: str, data) -> int:
        """Save price data for a ticker from multi-ticker download."""
        saved = 0
        try:
            # Multi-ticker data has multi-level columns
            ticker_data = data.xs(ticker, axis=1, level=1, drop_level=True)
        except KeyError:
            logger.warning(f"No data for {ticker} in batch response")
            return 0

        for idx, row in ticker_data.iterrows():
            price_date = idx.date() if hasattr(idx, "date") else idx

            if price_exists(self.session, ticker, price_date):
                continue

            try:
                close_val = row["Close"]
                if close_val is None or (hasattr(close_val, "__len__") and len(close_val) == 0):
                    continue

                price = Price(
                    ticker=ticker,
                    date=price_date,
                    open=float(row["Open"]) if row.get("Open") is not None else None,
                    high=float(row["High"]) if row.get("High") is not None else None,
                    low=float(row["Low"]) if row.get("Low") is not None else None,
                    close=float(close_val),
                    volume=int(row["Volume"]) if row.get("Volume") is not None else None,
                )
                save_price(self.session, price)
                saved += 1
            except Exception as e:
                logger.debug(f"Skipping row for {ticker}: {e}")

        return saved

    def get_price_with_change(self, ticker: str) -> dict | None:
        """
        Get latest price with daily/weekly change calculations.

        Returns dict with price, changes, and volume info.
        """
        prices = get_prices(self.session, ticker, days=30)

        if not prices:
            return None

        latest = prices[0]
        result = {
            "ticker": ticker,
            "price": latest.close,
            "date": latest.date,
            "volume": latest.volume,
        }

        # Daily change (compare to previous day)
        if len(prices) >= 2:
            prev_close = prices[1].close
            daily_change = ((latest.close - prev_close) / prev_close) * 100
            result["daily_change_pct"] = round(daily_change, 2)

        # Weekly change (compare to ~5 trading days ago)
        if len(prices) >= 6:
            week_ago_close = prices[5].close
            weekly_change = ((latest.close - week_ago_close) / week_ago_close) * 100
            result["weekly_change_pct"] = round(weekly_change, 2)

        # Volume vs 20-day average
        if len(prices) >= 20 and latest.volume:
            volumes = [p.volume for p in prices[:20] if p.volume]
            if volumes:
                avg_volume = sum(volumes) / len(volumes)
                result["avg_volume_20d"] = int(avg_volume)
                result["volume_ratio"] = round(latest.volume / avg_volume, 2)

        return result
```

**Step 2: Update collectors __init__.py**

```python
# src/investment_monitor/collectors/__init__.py
"""Data collectors module."""

from .base import BaseCollector, CollectorResult
from .prices import PriceCollector

__all__ = ["BaseCollector", "CollectorResult", "PriceCollector"]
```

**Step 3: Add integration test (skipped in CI)**

```python
# Add to tests/test_collectors.py

from investment_monitor.collectors import PriceCollector


@pytest.mark.skip(reason="Integration test - requires network")
@pytest.mark.asyncio
async def test_price_collector_integration(db_session):
    """Integration test for price collector."""
    from investment_monitor.config import Settings

    collector = PriceCollector(db_session, Settings())
    result = await collector.collect(["AAPL"])

    assert result.success
    assert result.records_collected > 0
```

**Step 4: Commit**

```bash
git add src/investment_monitor/collectors tests/test_collectors.py
git commit -m "feat: add price collector with yfinance (T08)"
```

---

### Task 9-12: Additional Collectors

Due to length, I'll summarize the remaining collectors. Each follows the same pattern:

**Task 9: Insider Transaction Collector (T09)**
- File: `src/investment_monitor/collectors/insider.py`
- Fetches Form 4 filings from SEC EDGAR
- Parses XML to extract transaction details
- Uses SEC rate limit (10 req/sec)

**Task 10: News Collector (T10)**
- File: `src/investment_monitor/collectors/news.py`
- Fetches from RSS feeds (Yahoo Finance, Seeking Alpha)
- Matches headlines to portfolio tickers
- Deduplicates by URL

**Task 11: Earnings Calendar Collector (T11)**
- File: `src/investment_monitor/collectors/earnings.py`
- Uses yfinance earnings calendar
- Stores upcoming earnings dates

**Task 12: ETF Holdings Collector (T12)**
- File: `src/investment_monitor/collectors/etf_holdings.py`
- Fetches Vanguard ETF holdings (JSON API)
- Detects holdings changes

---

## Phase 3: Alert Engine

### Task 13-15: Alert Engine, Priority, Deduplication

**Files:**
- Create: `src/investment_monitor/alerts/__init__.py`
- Create: `src/investment_monitor/alerts/engine.py`
- Create: `src/investment_monitor/alerts/rules.py`
- Create: `src/investment_monitor/alerts/priority.py`
- Create: `src/investment_monitor/alerts/dedup.py`

The alert engine checks all rules against collected data and generates AlertMessage objects, with priority classification and deduplication.

---

## Phase 4: AI Integration

### Task 16-18: Ollama, News Processing, Claude API

**Files:**
- Create: `src/investment_monitor/analysis/__init__.py`
- Create: `src/investment_monitor/analysis/local_llm.py`
- Create: `src/investment_monitor/analysis/prompts.py`
- Create: `src/investment_monitor/analysis/news_processor.py`
- Create: `src/investment_monitor/analysis/claude_api.py`

Local LLM for news relevance scoring, Claude API for weekly synthesis with budget tracking.

---

## Phase 5: Orchestration

### Task 19: Main Orchestrator (T19)

**Files:**
- Create: `src/investment_monitor/main.py`
- Create: `src/investment_monitor/cli.py`

### Task 20: Docker Setup (T20)

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yaml`
- Create: `cron/crontab`
- Create: `scripts/run_regular.sh`
- Create: `scripts/run_digest.sh`
- Create: `scripts/run_weekly.sh`

---

## Execution Checklist

- [ ] Phase 1: Foundation (T01-T06)
- [ ] Phase 2: Collectors (T07-T12)
- [ ] Phase 3: Alert Engine (T13-T15)
- [ ] Phase 4: AI Integration (T16-T18)
- [ ] Phase 5: Orchestration (T19-T20)

---

*Plan created: 2026-01-28*
