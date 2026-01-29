"""Tests for the UniverseCollector module."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from investment_monitor.collectors.universe import UniverseCollector
from investment_monitor.collectors.base import CollectorResult
from investment_monitor.config import Settings
from investment_monitor.storage.research_models import StockCandidate


# Sample HTML for mocking Wikipedia responses
SAMPLE_SP500_HTML = """
<html>
<body>
<table id="constituents" class="wikitable">
    <tr>
        <th>Symbol</th>
        <th>Security</th>
        <th>GICS Sector</th>
    </tr>
    <tr>
        <td>AAPL</td>
        <td>Apple Inc.</td>
        <td>Information Technology</td>
    </tr>
    <tr>
        <td>MSFT</td>
        <td>Microsoft Corporation</td>
        <td>Information Technology</td>
    </tr>
    <tr>
        <td>GOOGL</td>
        <td>Alphabet Inc. (Class A)</td>
        <td>Communication Services</td>
    </tr>
    <tr>
        <td>BRK.B</td>
        <td>Berkshire Hathaway Inc.</td>
        <td>Financials</td>
    </tr>
</table>
</body>
</html>
"""

SAMPLE_NASDAQ100_HTML = """
<html>
<body>
<table class="wikitable">
    <tr>
        <th>Company</th>
        <th>Ticker</th>
        <th>GICS Sector</th>
    </tr>
    <tr>
        <td>Apple Inc.</td>
        <td>AAPL</td>
        <td>Information Technology</td>
    </tr>
    <tr>
        <td>Microsoft Corporation</td>
        <td>MSFT</td>
        <td>Information Technology</td>
    </tr>
    <tr>
        <td>Amazon.com Inc.</td>
        <td>AMZN</td>
        <td>Consumer Discretionary</td>
    </tr>
    <tr>
        <td>NVIDIA Corporation</td>
        <td>NVDA</td>
        <td>Information Technology</td>
    </tr>
</table>
</body>
</html>
"""

# Larger sample for testing full NASDAQ parsing (100+ rows needed)
SAMPLE_NASDAQ100_FULL_HTML = """
<html>
<body>
<table class="wikitable">
    <tr>
        <th>Company</th>
        <th>Ticker</th>
        <th>Sector</th>
    </tr>
""" + "\n".join([
    f"""<tr><td>Company {i}</td><td>{'ABCDEFGHIJ'[i % 10] * 4}</td><td>Tech</td></tr>"""
    for i in range(105)
]) + """
</table>
</body>
</html>
"""


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_config():
    """Create a mock settings object."""
    return Settings()


@pytest.fixture
def collector(mock_session, mock_config):
    """Create a UniverseCollector instance."""
    return UniverseCollector(
        mock_session, mock_config,
        collect_sp500=True,
        collect_nasdaq100=True
    )


@pytest.fixture
def collector_no_indices(mock_session, mock_config):
    """Create a UniverseCollector that doesn't collect indices."""
    return UniverseCollector(
        mock_session, mock_config,
        collect_sp500=False,
        collect_nasdaq100=False
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestUniverseCollectorInit:
    """Tests for UniverseCollector initialization."""

    def test_initialization(self, mock_session, mock_config):
        """Should initialize with correct attributes."""
        collector = UniverseCollector(mock_session, mock_config)

        assert collector.session == mock_session
        assert collector.config == mock_config
        assert collector.name == "universe"
        assert collector.rate_limit_calls == 10
        assert collector.rate_limit_period == 60
        assert collector._collect_sp500 is True
        assert collector._collect_nasdaq100 is True

    def test_initialization_with_options(self, mock_session, mock_config):
        """Should respect initialization options."""
        collector = UniverseCollector(
            mock_session, mock_config,
            collect_sp500=False,
            collect_nasdaq100=False
        )

        assert collector._collect_sp500 is False
        assert collector._collect_nasdaq100 is False


# ============================================================================
# S&P 500 Collection Tests
# ============================================================================


class TestCollectSP500:
    """Tests for S&P 500 collection."""

    async def test_collect_sp500_parses_table(self, collector):
        """Should parse S&P 500 constituents table correctly."""
        with patch.object(collector, '_fetch_url', return_value=SAMPLE_SP500_HTML):
            tickers = await collector.collect_sp500()

        assert len(tickers) == 4
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers
        # BRK.B should be converted to BRK-B
        assert "BRK-B" in tickers

    async def test_collect_sp500_handles_empty_table(self, collector):
        """Should handle empty table gracefully."""
        empty_html = "<html><body></body></html>"

        with patch.object(collector, '_fetch_url', return_value=empty_html):
            tickers = await collector.collect_sp500()

        assert tickers == []

    async def test_collect_sp500_handles_request_error(self, collector):
        """Should retry on request errors."""
        import requests

        with patch.object(collector, '_fetch_url') as mock_fetch:
            mock_fetch.side_effect = [
                requests.RequestException("Connection error"),
                SAMPLE_SP500_HTML  # Succeeds on retry
            ]
            tickers = await collector.collect_sp500()

        assert len(tickers) == 4
        assert mock_fetch.call_count == 2


# ============================================================================
# NASDAQ 100 Collection Tests
# ============================================================================


class TestCollectNASDAQ100:
    """Tests for NASDAQ 100 collection."""

    async def test_collect_nasdaq100_parses_table(self, collector):
        """Should parse NASDAQ 100 constituents table correctly."""
        with patch.object(collector, '_fetch_url', return_value=SAMPLE_NASDAQ100_HTML):
            with patch.object(
                collector,
                '_parse_nasdaq100_alternative',
                return_value=["AAPL", "MSFT", "AMZN", "NVDA"]
            ):
                tickers = await collector.collect_nasdaq100()

        # Should fall through to alternative parsing due to < 90 tickers
        assert len(tickers) >= 4

    async def test_collect_nasdaq100_handles_empty_table(self, collector):
        """Should handle empty table gracefully."""
        empty_html = "<html><body></body></html>"

        with patch.object(collector, '_fetch_url', return_value=empty_html):
            tickers = await collector.collect_nasdaq100()

        assert tickers == []


# ============================================================================
# ETF Holdings Collection Tests
# ============================================================================


class TestCollectETFHoldings:
    """Tests for ETF holdings collection."""

    async def test_collect_etf_holdings_success(self, collector):
        """Should collect ETF holdings using yfinance."""
        mock_ticker = MagicMock()
        mock_funds_data = MagicMock()
        mock_funds_data.top_holdings = MagicMock()
        mock_funds_data.top_holdings.index = ["AAPL", "MSFT", "GOOGL"]
        mock_ticker.funds_data = mock_funds_data

        with patch('yfinance.Ticker', return_value=mock_ticker):
            holdings = await collector.collect_etf_holdings("QQQ")

        assert "AAPL" in holdings
        assert "MSFT" in holdings
        assert "GOOGL" in holdings

    async def test_collect_etf_holdings_empty(self, collector):
        """Should handle ETFs with no available holdings data."""
        mock_ticker = MagicMock()
        mock_ticker.funds_data = None

        with patch('yfinance.Ticker', return_value=mock_ticker):
            holdings = await collector.collect_etf_holdings("XYZ")

        assert holdings == []

    async def test_collect_etf_holdings_exception(self, collector):
        """Should handle exceptions gracefully."""
        with patch('yfinance.Ticker', side_effect=Exception("API Error")):
            holdings = await collector.collect_etf_holdings("QQQ")

        assert holdings == []


# ============================================================================
# Deduplication Tests
# ============================================================================


class TestDeduplication:
    """Tests for ticker deduplication logic."""

    def test_deduplicate_tickers_removes_duplicates(self, collector):
        """Should remove duplicate tickers."""
        tickers = ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT", "NVDA"]

        result = collector._deduplicate_tickers(tickers)

        assert len(result) == 4
        assert result == ["AAPL", "MSFT", "GOOGL", "NVDA"]

    def test_deduplicate_tickers_preserves_order(self, collector):
        """Should preserve order of first occurrence."""
        tickers = ["GOOGL", "AAPL", "MSFT", "AAPL", "GOOGL"]

        result = collector._deduplicate_tickers(tickers)

        assert result == ["GOOGL", "AAPL", "MSFT"]

    def test_deduplicate_tickers_normalizes_case(self, collector):
        """Should normalize case and treat as duplicates."""
        tickers = ["aapl", "AAPL", "Aapl"]

        result = collector._deduplicate_tickers(tickers)

        assert len(result) == 1
        assert result[0] == "AAPL"

    def test_deduplicate_tickers_handles_whitespace(self, collector):
        """Should handle whitespace in tickers."""
        tickers = ["AAPL ", " MSFT", "  AAPL  "]

        result = collector._deduplicate_tickers(tickers)

        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result

    def test_deduplicate_tickers_removes_empty(self, collector):
        """Should remove empty tickers."""
        tickers = ["AAPL", "", "MSFT", "   ", "GOOGL"]

        result = collector._deduplicate_tickers(tickers)

        assert len(result) == 3
        assert "" not in result

    def test_deduplicate_tickers_empty_input(self, collector):
        """Should handle empty input list."""
        result = collector._deduplicate_tickers([])

        assert result == []


# ============================================================================
# Save Candidates Tests
# ============================================================================


class TestSaveCandidates:
    """Tests for saving stock candidates."""

    def test_save_candidates_new_tickers(self, collector, mock_session):
        """Should save new candidates to database."""
        tickers = ["AAPL", "MSFT", "GOOGL"]

        with patch(
            'investment_monitor.collectors.universe.get_candidate_by_ticker',
            return_value=None
        ):
            with patch(
                'investment_monitor.collectors.universe.save_candidate'
            ) as mock_save:
                records, errors = collector._save_candidates(tickers, "sp500")

        assert records == 3
        assert len(errors) == 0
        assert mock_save.call_count == 3

    def test_save_candidates_skips_existing(self, collector, mock_session):
        """Should skip existing candidates."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        existing_candidate = StockCandidate(ticker="AAPL", discovery_source="sp500")

        def mock_get_candidate(session, ticker):
            if ticker == "AAPL":
                return existing_candidate
            return None

        with patch(
            'investment_monitor.collectors.universe.get_candidate_by_ticker',
            side_effect=mock_get_candidate
        ):
            with patch(
                'investment_monitor.collectors.universe.save_candidate'
            ) as mock_save:
                records, errors = collector._save_candidates(tickers, "sp500")

        assert records == 2  # Only MSFT and GOOGL saved
        assert len(errors) == 0
        assert mock_save.call_count == 2

    def test_save_candidates_handles_errors(self, collector, mock_session):
        """Should handle save errors gracefully."""
        tickers = ["AAPL", "MSFT"]

        with patch(
            'investment_monitor.collectors.universe.get_candidate_by_ticker',
            return_value=None
        ):
            with patch(
                'investment_monitor.collectors.universe.save_candidate',
                side_effect=Exception("Database error")
            ):
                records, errors = collector._save_candidates(tickers, "sp500")

        assert records == 0
        assert len(errors) == 2
        assert "AAPL" in errors[0]
        assert "MSFT" in errors[1]


# ============================================================================
# Main Collect Tests
# ============================================================================


class TestCollect:
    """Tests for the main collect method."""

    async def test_collect_from_all_sources(self, collector):
        """Should collect from all configured sources."""
        with patch.object(
            collector, 'collect_sp500',
            return_value=["AAPL", "MSFT"]
        ):
            with patch.object(
                collector, 'collect_nasdaq100',
                return_value=["AMZN", "NVDA"]
            ):
                with patch.object(
                    collector, 'collect_etf_holdings',
                    return_value=["TSLA"]
                ):
                    with patch.object(
                        collector, '_save_candidates',
                        return_value=(2, [])
                    ):
                        result = await collector.collect(["QQQ"])

        assert isinstance(result, CollectorResult)
        assert result.collector_name == "universe"
        # 2 records from each source (sp500, nasdaq100, etf)
        assert result.records_collected == 6

    async def test_collect_skips_disabled_indices(self, collector_no_indices):
        """Should skip disabled index collections."""
        with patch.object(
            collector_no_indices, 'collect_sp500'
        ) as mock_sp500:
            with patch.object(
                collector_no_indices, 'collect_nasdaq100'
            ) as mock_nasdaq:
                with patch.object(
                    collector_no_indices, 'collect_etf_holdings',
                    return_value=["AAPL"]
                ):
                    with patch.object(
                        collector_no_indices, '_save_candidates',
                        return_value=(1, [])
                    ):
                        result = await collector_no_indices.collect(["SPY"])

        mock_sp500.assert_not_called()
        mock_nasdaq.assert_not_called()
        assert result.records_collected == 1

    async def test_collect_handles_sp500_error(self, collector):
        """Should continue despite S&P 500 collection error."""
        with patch.object(
            collector, 'collect_sp500',
            side_effect=Exception("Wikipedia error")
        ):
            with patch.object(
                collector, 'collect_nasdaq100',
                return_value=["AMZN"]
            ):
                with patch.object(
                    collector, '_save_candidates',
                    return_value=(1, [])
                ):
                    result = await collector.collect([])

        assert result.records_collected == 1
        assert len(result.errors) == 1
        assert "S&P 500" in result.errors[0]

    async def test_collect_handles_nasdaq_error(self, collector):
        """Should continue despite NASDAQ 100 collection error."""
        with patch.object(
            collector, 'collect_sp500',
            return_value=["AAPL"]
        ):
            with patch.object(
                collector, 'collect_nasdaq100',
                side_effect=Exception("Wikipedia error")
            ):
                with patch.object(
                    collector, '_save_candidates',
                    return_value=(1, [])
                ):
                    result = await collector.collect([])

        assert result.records_collected == 1
        assert len(result.errors) == 1
        assert "NASDAQ 100" in result.errors[0]

    async def test_collect_returns_result_with_timing(self, collector):
        """Should include timing information in result."""
        with patch.object(
            collector, 'collect_sp500',
            return_value=[]
        ):
            with patch.object(
                collector, 'collect_nasdaq100',
                return_value=[]
            ):
                result = await collector.collect([])

        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    async def test_collect_empty_etf_list(self, collector_no_indices):
        """Should handle empty ETF ticker list."""
        result = await collector_no_indices.collect([])

        assert result.collector_name == "universe"
        assert result.records_collected == 0
        assert result.success is True


# ============================================================================
# Collect Single Tests
# ============================================================================


class TestCollectSingle:
    """Tests for the collect_single method."""

    async def test_collect_single_etf(self, collector):
        """Should collect holdings from a single ETF."""
        with patch.object(
            collector, 'collect_etf_holdings',
            return_value=["AAPL", "MSFT"]
        ):
            with patch.object(
                collector, '_save_candidates',
                return_value=(2, [])
            ):
                records = await collector.collect_single("QQQ")

        assert records == 2

    async def test_collect_single_error(self, collector):
        """Should raise exception on error."""
        with patch.object(
            collector, 'collect_etf_holdings',
            side_effect=Exception("API Error")
        ):
            with pytest.raises(Exception):
                await collector.collect_single("QQQ")


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests for UniverseCollector."""

    async def test_full_collection_workflow(self, mock_session, mock_config):
        """Should execute full collection workflow."""
        collector = UniverseCollector(
            mock_session, mock_config,
            collect_sp500=True,
            collect_nasdaq100=True
        )

        # Mock all external calls
        with patch.object(collector, '_fetch_url') as mock_fetch:
            mock_fetch.side_effect = [
                SAMPLE_SP500_HTML,  # S&P 500
                SAMPLE_NASDAQ100_HTML,  # NASDAQ 100
            ]

            with patch(
                'investment_monitor.collectors.universe.get_candidate_by_ticker',
                return_value=None
            ):
                with patch(
                    'investment_monitor.collectors.universe.save_candidate'
                ) as mock_save:
                    with patch.object(
                        collector,
                        '_parse_nasdaq100_alternative',
                        return_value=["AAPL", "MSFT", "AMZN", "NVDA"]
                    ):
                        result = await collector.collect([])

        # Should have collected from both indices
        assert result.collector_name == "universe"
        assert mock_save.call_count > 0

    async def test_deduplication_across_sources(self, mock_session, mock_config):
        """Should track unique tickers across all sources."""
        collector = UniverseCollector(
            mock_session, mock_config,
            collect_sp500=True,
            collect_nasdaq100=True
        )

        # AAPL appears in both indices
        sp500_tickers = ["AAPL", "MSFT"]
        nasdaq_tickers = ["AAPL", "GOOGL"]

        saved_tickers = []

        def track_saves(session, candidate):
            saved_tickers.append(candidate.ticker)
            return len(saved_tickers)

        def mock_get_candidate(session, ticker):
            # First time seeing each ticker returns None
            # After first save, returns existing
            if ticker in saved_tickers:
                return StockCandidate(ticker=ticker, discovery_source="test")
            return None

        with patch.object(collector, 'collect_sp500', return_value=sp500_tickers):
            with patch.object(collector, 'collect_nasdaq100', return_value=nasdaq_tickers):
                with patch(
                    'investment_monitor.collectors.universe.get_candidate_by_ticker',
                    side_effect=mock_get_candidate
                ):
                    with patch(
                        'investment_monitor.collectors.universe.save_candidate',
                        side_effect=track_saves
                    ):
                        await collector.collect([])

        # AAPL should only be saved once (from sp500, first source)
        assert saved_tickers.count("AAPL") == 1
        # Total unique tickers: AAPL, MSFT, GOOGL = 3
        assert len(saved_tickers) == 3


# ============================================================================
# Alternative Parsing Tests
# ============================================================================


class TestAlternativeParsing:
    """Tests for alternative NASDAQ 100 parsing."""

    def test_parse_nasdaq100_alternative_finds_ticker_column(self, collector):
        """Should find ticker column by header name."""
        from bs4 import BeautifulSoup

        html = """
        <table class="wikitable">
            <tr><th>Company</th><th>Symbol</th><th>Sector</th></tr>
            <tr><td>Apple</td><td>AAPL</td><td>Tech</td></tr>
            <tr><td>Microsoft</td><td>MSFT</td><td>Tech</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")

        # This will return empty since we need 90+ tickers for it to accept
        tickers = collector._parse_nasdaq100_alternative(soup)

        # With only 2 tickers, won't meet threshold
        assert tickers == []

    def test_parse_nasdaq100_alternative_handles_no_table(self, collector):
        """Should handle missing table gracefully."""
        from bs4 import BeautifulSoup

        html = "<html><body><p>No table here</p></body></html>"
        soup = BeautifulSoup(html, "lxml")

        tickers = collector._parse_nasdaq100_alternative(soup)

        assert tickers == []
