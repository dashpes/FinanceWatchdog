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
