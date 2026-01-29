"""Tests for data models."""

import tempfile
from decimal import Decimal
from pathlib import Path

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
