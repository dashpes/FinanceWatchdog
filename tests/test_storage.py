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
