"""Tests for Congressional Trades Collector."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from investment_monitor.collectors.congress import CongressTradesCollector
from investment_monitor.config import Settings
from investment_monitor.storage import (
    CongressionalTrade,
    get_session,
    get_trades_for_ticker,
    init_db,
)


# Sample House trade data matching the API format
# Using recent dates so tests work with get_trades_for_ticker which uses days filter
SAMPLE_HOUSE_TRADES = [
    {
        "representative": "Hon. Nancy Pelosi",
        "ticker": "AAPL",
        "asset_description": "Apple Inc - Common Stock",
        "transaction_date": "2026-01-15",
        "disclosure_date": "2026-01-20",
        "type": "purchase",
        "amount": "$100,001 - $250,000",
        "party": "Democrat",
        "state": "CA",
        "district": "11",
    },
    {
        "representative": "Hon. Dan Crenshaw",
        "ticker": "MSFT",
        "asset_description": "Microsoft Corporation",
        "transaction_date": "2026-01-10",
        "disclosure_date": "2026-01-15",
        "type": "Sale",
        "amount": "$15,001 - $50,000",
        "party": "Republican",
        "state": "TX",
        "district": "02",
    },
    {
        "representative": "Hon. Jane Doe",
        "ticker": "AAPL",
        "asset_description": "Apple Inc - Common Stock",
        "transaction_date": "2026-01-12",
        "disclosure_date": "2026-01-18",
        "type": "sale (partial)",
        "amount": "$1,001 - $15,000",
        "party": "Democrat",
        "state": "NY",
        "district": "10",
    },
]

# Sample Senate trade data matching the API format
# Using recent dates so tests work with get_trades_for_ticker which uses days filter
SAMPLE_SENATE_TRADES = [
    {
        "senator": "Sen. John Smith",
        "ticker": "MSFT",
        "asset_description": "Microsoft Corporation",
        "transaction_date": "2026-01-08",
        "disclosure_date": "2026-01-12",
        "type": "Sale",
        "amount": "$15,001 - $50,000",
        "party": "Republican",
    },
    {
        "senator": "Sen. Alice Johnson",
        "ticker": "AAPL",
        "asset_description": "Apple Inc",
        "transaction_date": "2026-01-05",
        "disclosure_date": "2026-01-10",
        "type": "purchase",
        "amount": "$50,001 - $100,000",
        "party": "Democrat",
    },
]


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def collector(db_session, settings):
    """Create a CongressTradesCollector instance."""
    return CongressTradesCollector(db_session, settings)


class TestParseDate:
    """Tests for date parsing functionality."""

    def test_parse_standard_format(self, collector):
        """Test parsing ISO format date."""
        result = collector._parse_date("2023-11-15")
        assert result == date(2023, 11, 15)

    def test_parse_us_format(self, collector):
        """Test parsing US format date (MM/DD/YYYY)."""
        result = collector._parse_date("11/15/2023")
        assert result == date(2023, 11, 15)

    def test_parse_us_dash_format(self, collector):
        """Test parsing US format date with dashes (MM-DD-YYYY)."""
        result = collector._parse_date("11-15-2023")
        assert result == date(2023, 11, 15)

    def test_parse_empty_string(self, collector):
        """Test parsing empty string returns None."""
        assert collector._parse_date("") is None

    def test_parse_none(self, collector):
        """Test parsing None returns None."""
        assert collector._parse_date(None) is None

    def test_parse_na(self, collector):
        """Test parsing N/A returns None."""
        assert collector._parse_date("N/A") is None
        assert collector._parse_date("n/a") is None

    def test_parse_invalid_format(self, collector):
        """Test parsing invalid format returns None."""
        assert collector._parse_date("invalid") is None
        assert collector._parse_date("2023/99/99") is None


class TestNormalizeTradeType:
    """Tests for trade type normalization."""

    def test_normalize_purchase(self, collector):
        """Test normalizing purchase variants."""
        assert collector._normalize_trade_type("purchase") == "buy"
        assert collector._normalize_trade_type("Purchase") == "buy"
        assert collector._normalize_trade_type("buy") == "buy"
        assert collector._normalize_trade_type("bought") == "buy"

    def test_normalize_sale(self, collector):
        """Test normalizing sale variants."""
        assert collector._normalize_trade_type("sale") == "sell"
        assert collector._normalize_trade_type("Sale") == "sell"
        assert collector._normalize_trade_type("sell") == "sell"
        assert collector._normalize_trade_type("sold") == "sell"
        assert collector._normalize_trade_type("sale (full)") == "sell"
        assert collector._normalize_trade_type("sale (partial)") == "sell"

    def test_normalize_exchange(self, collector):
        """Test normalizing exchange type."""
        assert collector._normalize_trade_type("exchange") == "exchange"

    def test_normalize_empty(self, collector):
        """Test normalizing empty string."""
        assert collector._normalize_trade_type("") == "unknown"
        assert collector._normalize_trade_type(None) == "unknown"

    def test_normalize_unknown_type(self, collector):
        """Test normalizing unknown type keeps original."""
        assert collector._normalize_trade_type("gift") == "gift"
        assert collector._normalize_trade_type("transfer") == "transfer"


class TestParseTrade:
    """Tests for trade parsing functionality."""

    def test_parse_house_trade(self, collector):
        """Test parsing a House trade."""
        raw = SAMPLE_HOUSE_TRADES[0]
        trade = collector.parse_trade(raw, "House")

        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.politician == "Hon. Nancy Pelosi"
        assert trade.party == "Democrat"
        assert trade.chamber == "House"
        assert trade.trade_type == "buy"
        assert trade.amount_range == "$100,001 - $250,000"
        assert trade.trade_date == date(2026, 1, 15)
        assert trade.disclosure_date == date(2026, 1, 20)
        assert trade.description == "Apple Inc - Common Stock"

    def test_parse_senate_trade(self, collector):
        """Test parsing a Senate trade."""
        raw = SAMPLE_SENATE_TRADES[0]
        trade = collector.parse_trade(raw, "Senate")

        assert trade is not None
        assert trade.ticker == "MSFT"
        assert trade.politician == "Sen. John Smith"
        assert trade.party == "Republican"
        assert trade.chamber == "Senate"
        assert trade.trade_type == "sell"
        assert trade.amount_range == "$15,001 - $50,000"
        assert trade.trade_date == date(2026, 1, 8)

    def test_parse_trade_missing_ticker(self, collector):
        """Test parsing trade with missing ticker returns None."""
        raw = {"representative": "Hon. Test", "type": "purchase", "amount": "$1,000"}
        trade = collector.parse_trade(raw, "House")
        assert trade is None

    def test_parse_trade_missing_politician(self, collector):
        """Test parsing trade with missing politician returns None."""
        raw = {"ticker": "AAPL", "type": "purchase", "amount": "$1,000"}
        trade = collector.parse_trade(raw, "House")
        assert trade is None

    def test_parse_trade_missing_date(self, collector):
        """Test parsing trade with missing date returns None."""
        raw = {
            "representative": "Hon. Test",
            "ticker": "AAPL",
            "type": "purchase",
            "amount": "$1,000",
        }
        trade = collector.parse_trade(raw, "House")
        assert trade is None

    def test_parse_trade_invalid_ticker(self, collector):
        """Test parsing trade with invalid ticker returns None."""
        raw = {
            "representative": "Hon. Test",
            "ticker": "--",
            "type": "purchase",
            "amount": "$1,000",
            "transaction_date": "2023-11-15",
        }
        trade = collector.parse_trade(raw, "House")
        assert trade is None

    @pytest.mark.parametrize("junk", ["NONE", "none", "NA", "na", "N\\A", "n/a", "  "])
    def test_parse_trade_shared_junk_ticker_filter(self, collector, junk):
        """#15: junk placeholders are dropped via the SHARED is_junk_ticker helper.

        The old inline check only filtered '--'/'N/A'; centralizing extends congress to
        the full union (NONE/NA/N\\A/...), so non-issuer tokens never become trades.
        """
        raw = {
            "representative": "Hon. Test",
            "ticker": junk,
            "type": "purchase",
            "amount": "$1,000",
            "transaction_date": "2023-11-15",
        }
        assert collector.parse_trade(raw, "House") is None

    def test_parse_trade_sale_partial(self, collector):
        """Test parsing partial sale trade."""
        raw = SAMPLE_HOUSE_TRADES[2]
        trade = collector.parse_trade(raw, "House")

        assert trade is not None
        assert trade.trade_type == "sell"


class TestDeduplication:
    """Tests for trade deduplication."""

    def test_trade_exists_returns_false_for_new(self, collector, db_session):
        """Test _trade_exists returns False for new trade."""
        trade = CongressionalTrade(
            ticker="AAPL",
            politician="Test Politician",
            trade_type="buy",
            amount_range="$1,000 - $5,000",
            trade_date=date(2023, 11, 15),
            chamber="House",
        )
        assert collector._trade_exists(trade) is False

    def test_trade_exists_returns_true_for_existing(self, collector, db_session):
        """Test _trade_exists returns True for existing trade."""
        # First, save a trade
        trade1 = CongressionalTrade(
            ticker="AAPL",
            politician="Test Politician",
            trade_type="buy",
            amount_range="$1,000 - $5,000",
            trade_date=date(2023, 11, 15),
            chamber="House",
        )
        db_session.add(trade1)
        db_session.flush()

        # Now check if same trade exists
        trade2 = CongressionalTrade(
            ticker="AAPL",
            politician="Test Politician",
            trade_type="buy",
            amount_range="$1,000 - $5,000",
            trade_date=date(2023, 11, 15),
            chamber="House",
        )
        assert collector._trade_exists(trade2) is True

    def test_different_amount_not_duplicate(self, collector, db_session):
        """Test trades with different amounts are not duplicates."""
        trade1 = CongressionalTrade(
            ticker="AAPL",
            politician="Test Politician",
            trade_type="buy",
            amount_range="$1,000 - $5,000",
            trade_date=date(2023, 11, 15),
            chamber="House",
        )
        db_session.add(trade1)
        db_session.flush()

        trade2 = CongressionalTrade(
            ticker="AAPL",
            politician="Test Politician",
            trade_type="buy",
            amount_range="$5,001 - $15,000",  # Different amount
            trade_date=date(2023, 11, 15),
            chamber="House",
        )
        assert collector._trade_exists(trade2) is False


class TestFetchTrades:
    """Tests for fetching trades from APIs."""

    @pytest.mark.asyncio
    async def test_fetch_house_trades_success(self, collector):
        """Test successful House trades fetch with mocked response."""
        # Use MagicMock for response since json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            trades = await collector.fetch_house_trades()

            assert len(trades) == 3
            assert trades[0]["representative"] == "Hon. Nancy Pelosi"

    @pytest.mark.asyncio
    async def test_fetch_senate_trades_success(self, collector):
        """Test successful Senate trades fetch with mocked response."""
        # Use MagicMock for response since json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            trades = await collector.fetch_senate_trades()

            assert len(trades) == 2
            assert trades[0]["senator"] == "Sen. John Smith"

    @pytest.mark.asyncio
    async def test_fetch_trades_network_error(self, collector):
        """Test handling of network errors during fetch."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=MagicMock()
            )
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            with pytest.raises(Exception):
                await collector.fetch_house_trades()


class TestCollect:
    """Tests for the main collect method."""

    @pytest.mark.asyncio
    async def test_collect_filters_by_ticker(self, collector, db_session):
        """Test that collect only saves trades for specified tickers."""
        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_house_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            # First call returns House, second returns Senate
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Only collect AAPL trades
            result = await collector.collect(["AAPL"])

            assert result.success is True
            # Should have: 2 House AAPL + 1 Senate AAPL = 3 AAPL trades
            assert result.records_collected == 3

    @pytest.mark.asyncio
    async def test_collect_handles_house_error(self, collector, db_session):
        """Test that collect handles House API errors gracefully."""
        # Disable retries for this test to avoid consuming mock side_effects
        collector.max_retries = 0

        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await collector.collect(["AAPL"])

            # Should still process Senate trades
            assert result.success is False  # Has errors
            assert len(result.errors) > 0
            assert result.records_collected > 0  # Senate trades saved

    @pytest.mark.asyncio
    async def test_collect_deduplicates(self, collector, db_session):
        """Test that collect deduplicates trades."""
        # Use MagicMock for response since json() is synchronous
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = []
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [mock_response, mock_senate_response]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # First collect
            result1 = await collector.collect(["AAPL"])
            count1 = result1.records_collected

        # Reset mocks for second collect
        mock_response2 = MagicMock()
        mock_response2.json.return_value = SAMPLE_HOUSE_TRADES
        mock_response2.raise_for_status = MagicMock()

        mock_senate_response2 = MagicMock()
        mock_senate_response2.json.return_value = []
        mock_senate_response2.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [mock_response2, mock_senate_response2]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Second collect with same data
            result2 = await collector.collect(["AAPL"])
            count2 = result2.records_collected

        # Second collect should save 0 (all duplicates)
        assert count1 > 0
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_collect_case_insensitive_tickers(self, collector, db_session):
        """Test that ticker matching is case-insensitive."""
        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_house_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Use lowercase ticker
            result = await collector.collect(["aapl"])

            assert result.success is True
            assert result.records_collected == 3  # Should still match AAPL trades


class TestCollectSingle:
    """Tests for collect_single method."""

    @pytest.mark.asyncio
    async def test_collect_single_delegates_to_collect(self, collector, db_session):
        """Test that collect_single delegates to collect."""
        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_house_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            count = await collector.collect_single("MSFT")

            # Should have 1 House MSFT + 1 Senate MSFT = 2 trades
            assert count == 2


class TestIntegration:
    """Integration tests for the collector."""

    @pytest.mark.asyncio
    async def test_full_collection_workflow(self, collector, db_session):
        """Test complete collection and retrieval workflow."""
        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.json.return_value = SAMPLE_HOUSE_TRADES
        mock_house_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = SAMPLE_SENATE_TRADES
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await collector.collect(["AAPL", "MSFT"])

            assert result.success is True
            assert result.records_collected == 5  # All sample trades

        # Verify trades are retrievable
        aapl_trades = get_trades_for_ticker(db_session, "AAPL", days=365)
        assert len(aapl_trades) == 3  # 2 House + 1 Senate

        msft_trades = get_trades_for_ticker(db_session, "MSFT", days=365)
        assert len(msft_trades) == 2  # 1 House + 1 Senate

    @pytest.mark.asyncio
    async def test_collector_result_timing(self, collector, db_session):
        """Test that CollectorResult has proper timing info."""
        # Use MagicMock for response since json() is synchronous
        mock_house_response = MagicMock()
        mock_house_response.json.return_value = []
        mock_house_response.raise_for_status = MagicMock()

        mock_senate_response = MagicMock()
        mock_senate_response.json.return_value = []
        mock_senate_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = [
                mock_house_response,
                mock_senate_response,
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await collector.collect(["AAPL"])

            assert result.started_at is not None
            assert result.finished_at is not None
            assert result.finished_at >= result.started_at
            assert result.duration_seconds >= 0
