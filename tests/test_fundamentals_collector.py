"""Tests for the FundamentalsCollector module."""

import math
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from investment_monitor.collectors.fundamentals import (
    FundamentalsCollector,
    FundamentalsData,
)
from investment_monitor.collectors.base import CollectorResult
from investment_monitor.config import Settings


# Sample yfinance info data for mocking
SAMPLE_YFINANCE_INFO = {
    "trailingPE": 25.5,
    "priceToBook": 4.2,
    "priceToSalesTrailing12Months": 6.5,
    "pegRatio": 1.5,
    "revenueGrowth": 0.12,
    "earningsGrowth": 0.15,
    "returnOnEquity": 0.35,
    "profitMargins": 0.25,
    "debtToEquity": 1.2,
    "currentRatio": 1.8,
    "dividendYield": 0.005,
    "payoutRatio": 0.15,
    "freeCashflow": 100000000000,
    "marketCap": 3000000000000,
    "sector": "Technology",
    "industry": "Consumer Electronics",
}

SAMPLE_YFINANCE_INFO_MINIMAL = {
    "trailingPE": 20.0,
    "sector": "Technology",
}

SAMPLE_YFINANCE_INFO_WITH_NANS = {
    "trailingPE": float("nan"),
    "priceToBook": float("inf"),
    "priceToSalesTrailing12Months": None,
    "sector": "Technology",
}

# Sample financials DataFrame for mocking
SAMPLE_FINANCIALS = pd.DataFrame(
    {
        "2024-12-31": [400e9, 100e9],
        "2023-12-31": [380e9, 95e9],
        "2022-12-31": [360e9, 90e9],
        "2021-12-31": [340e9, 85e9],
    },
    index=["Total Revenue", "Net Income"],
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_config():
    """Create a mock settings object."""
    return Settings()


@pytest.fixture
def collector(mock_session, mock_config):
    """Create a FundamentalsCollector instance."""
    return FundamentalsCollector(mock_session, mock_config)


# ============================================================================
# FundamentalsData Model Tests
# ============================================================================


class TestFundamentalsDataModel:
    """Tests for the FundamentalsData Pydantic model."""

    def test_fundamentals_data_creation_full(self):
        """Should create FundamentalsData with all fields."""
        data = FundamentalsData(
            ticker="AAPL",
            pe_ratio=25.5,
            pb_ratio=4.2,
            ps_ratio=6.5,
            peg_ratio=1.5,
            revenue_growth_yoy=0.12,
            revenue_growth_3y=0.10,
            eps_growth_yoy=0.15,
            eps_growth_3y=0.12,
            roe=0.35,
            profit_margin=0.25,
            debt_to_equity=1.2,
            current_ratio=1.8,
            dividend_yield=0.005,
            payout_ratio=0.15,
            free_cash_flow=100000000000,
            market_cap=3000000000000,
            sector="Technology",
            industry="Consumer Electronics",
        )

        assert data.ticker == "AAPL"
        assert data.pe_ratio == 25.5
        assert data.sector == "Technology"

    def test_fundamentals_data_creation_minimal(self):
        """Should create FundamentalsData with only ticker."""
        data = FundamentalsData(ticker="MSFT")

        assert data.ticker == "MSFT"
        assert data.pe_ratio is None
        assert data.sector is None
        assert data.market_cap is None

    def test_fundamentals_data_allows_none_values(self):
        """Should allow None for all optional fields."""
        data = FundamentalsData(
            ticker="GOOGL",
            pe_ratio=None,
            pb_ratio=None,
            ps_ratio=None,
        )

        assert data.ticker == "GOOGL"
        assert data.pe_ratio is None


# ============================================================================
# Initialization Tests
# ============================================================================


class TestFundamentalsCollectorInit:
    """Tests for FundamentalsCollector initialization."""

    def test_initialization(self, mock_session, mock_config):
        """Should initialize with correct attributes."""
        collector = FundamentalsCollector(mock_session, mock_config)

        assert collector.session == mock_session
        assert collector.config == mock_config
        assert collector.name == "fundamentals"
        assert collector.rate_limit_calls == 30
        assert collector.rate_limit_period == 60
        assert collector._fundamentals_cache == {}

    def test_inherits_from_base_collector(self, collector):
        """Should inherit from BaseCollector."""
        from investment_monitor.collectors.base import BaseCollector

        assert isinstance(collector, BaseCollector)


# ============================================================================
# Safe Get Tests
# ============================================================================


class TestSafeGetFloat:
    """Tests for the _safe_get_float helper method."""

    def test_safe_get_float_valid_value(self, collector):
        """Should return valid float values."""
        info = {"pe": 25.5}

        assert collector._safe_get_float(info, "pe") == 25.5

    def test_safe_get_float_int_value(self, collector):
        """Should convert int to float."""
        info = {"pe": 25}

        assert collector._safe_get_float(info, "pe") == 25.0

    def test_safe_get_float_none_value(self, collector):
        """Should return None for None values."""
        info = {"pe": None}

        assert collector._safe_get_float(info, "pe") is None

    def test_safe_get_float_missing_key(self, collector):
        """Should return None for missing keys."""
        info = {"pe": 25.5}

        assert collector._safe_get_float(info, "missing") is None

    def test_safe_get_float_nan_value(self, collector):
        """Should return None for NaN values."""
        info = {"pe": float("nan")}

        assert collector._safe_get_float(info, "pe") is None

    def test_safe_get_float_inf_value(self, collector):
        """Should return None for infinity values."""
        info = {"pe": float("inf"), "pe_neg": float("-inf")}

        assert collector._safe_get_float(info, "pe") is None
        assert collector._safe_get_float(info, "pe_neg") is None

    def test_safe_get_float_zero_value(self, collector):
        """Should return zero (valid value)."""
        info = {"pe": 0.0}

        assert collector._safe_get_float(info, "pe") == 0.0

    def test_safe_get_float_string_value(self, collector):
        """Should return None for string values."""
        info = {"pe": "not a number"}

        assert collector._safe_get_float(info, "pe") is None

    def test_safe_get_float_list_value(self, collector):
        """Should return None for list values."""
        info = {"pe": [1, 2, 3]}

        assert collector._safe_get_float(info, "pe") is None


class TestSafeGetStr:
    """Tests for the _safe_get_str helper method."""

    def test_safe_get_str_valid_value(self, collector):
        """Should return valid string values."""
        info = {"sector": "Technology"}

        assert collector._safe_get_str(info, "sector") == "Technology"

    def test_safe_get_str_none_value(self, collector):
        """Should return None for None values."""
        info = {"sector": None}

        assert collector._safe_get_str(info, "sector") is None

    def test_safe_get_str_missing_key(self, collector):
        """Should return None for missing keys."""
        info = {"sector": "Tech"}

        assert collector._safe_get_str(info, "missing") is None

    def test_safe_get_str_int_value(self, collector):
        """Should return None for int values."""
        info = {"sector": 123}

        assert collector._safe_get_str(info, "sector") is None

    def test_safe_get_str_float_value(self, collector):
        """Should return None for float values."""
        info = {"sector": 1.5}

        assert collector._safe_get_str(info, "sector") is None


# ============================================================================
# CAGR Calculation Tests
# ============================================================================


class TestCAGRCalculation:
    """Tests for the _calculate_cagr helper method."""

    def test_calculate_cagr_valid(self, collector):
        """Should calculate correct CAGR."""
        # 100 -> 121 over 2 years = 10% CAGR
        values = [100, 110, 121]
        cagr = collector._calculate_cagr(values, 2)

        assert cagr is not None
        assert abs(cagr - 0.10) < 0.01

    def test_calculate_cagr_three_years(self, collector):
        """Should calculate 3-year CAGR."""
        # 100 -> 133.1 over 3 years = 10% CAGR
        values = [100, 110, 121, 133.1]
        cagr = collector._calculate_cagr(values, 3)

        assert cagr is not None
        assert abs(cagr - 0.10) < 0.01

    def test_calculate_cagr_empty_list(self, collector):
        """Should return None for empty list."""
        assert collector._calculate_cagr([], 3) is None

    def test_calculate_cagr_single_value(self, collector):
        """Should return None for single value."""
        assert collector._calculate_cagr([100], 3) is None

    def test_calculate_cagr_negative_start(self, collector):
        """Should return None for negative start value."""
        assert collector._calculate_cagr([-100, 110], 1) is None

    def test_calculate_cagr_negative_end(self, collector):
        """Should return None for negative end value."""
        assert collector._calculate_cagr([100, -110], 1) is None

    def test_calculate_cagr_zero_start(self, collector):
        """Should return None for zero start value."""
        assert collector._calculate_cagr([0, 110], 1) is None


# ============================================================================
# Growth Metrics Extraction Tests
# ============================================================================


class TestExtractGrowthMetrics:
    """Tests for growth metrics extraction from yfinance."""

    def test_extract_growth_metrics_from_info(self, collector):
        """Should extract growth metrics from info dict."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "revenueGrowth": 0.12,
            "earningsGrowth": 0.15,
        }
        mock_ticker.financials = pd.DataFrame()

        result = collector._extract_growth_metrics(mock_ticker)

        assert result[0] == 0.12  # revenue_growth_yoy
        assert result[2] == 0.15  # eps_growth_yoy

    def test_extract_growth_metrics_from_financials(self, collector):
        """Should calculate 3-year growth from financials."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = SAMPLE_FINANCIALS

        result = collector._extract_growth_metrics(mock_ticker)

        # revenue_growth_3y should be calculated
        assert result[1] is not None  # revenue_growth_3y
        # eps_growth_3y should be calculated
        assert result[3] is not None  # eps_growth_3y

    def test_extract_growth_metrics_empty_financials(self, collector):
        """Should handle empty financials gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = pd.DataFrame()

        result = collector._extract_growth_metrics(mock_ticker)

        assert result == (None, None, None, None)

    def test_extract_growth_metrics_exception(self, collector):
        """Should handle exceptions gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        type(mock_ticker).financials = PropertyMock(
            side_effect=Exception("API Error")
        )

        result = collector._extract_growth_metrics(mock_ticker)

        # Should return None values on exception
        assert all(v is None for v in result)


# ============================================================================
# Get Fundamentals Tests
# ============================================================================


class TestGetFundamentals:
    """Tests for the get_fundamentals method."""

    async def test_get_fundamentals_success(self, collector):
        """Should fetch fundamentals successfully."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = SAMPLE_FINANCIALS

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("AAPL")

        assert isinstance(result, FundamentalsData)
        assert result.ticker == "AAPL"
        assert result.pe_ratio == 25.5
        assert result.pb_ratio == 4.2
        assert result.sector == "Technology"
        assert result.industry == "Consumer Electronics"

    async def test_get_fundamentals_minimal_data(self, collector):
        """Should handle minimal data gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO_MINIMAL
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("MSFT")

        assert result.ticker == "MSFT"
        assert result.pe_ratio == 20.0
        assert result.sector == "Technology"
        assert result.pb_ratio is None
        assert result.dividend_yield is None

    async def test_get_fundamentals_handles_nans(self, collector):
        """Should convert NaN/Inf values to None."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO_WITH_NANS
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("GOOGL")

        assert result.pe_ratio is None  # Was NaN
        assert result.pb_ratio is None  # Was Inf
        assert result.ps_ratio is None  # Was None
        assert result.sector == "Technology"

    async def test_get_fundamentals_normalizes_ticker(self, collector):
        """Should normalize ticker to uppercase."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"sector": "Tech"}
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("aapl")

        assert result.ticker == "AAPL"

    async def test_get_fundamentals_retries_on_error(self, collector):
        """Should retry on transient errors."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO_MINIMAL
        mock_ticker.financials = pd.DataFrame()

        call_count = 0

        def create_ticker(symbol):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return mock_ticker

        with patch("yfinance.Ticker", side_effect=create_ticker):
            result = await collector.get_fundamentals("AAPL")

        assert result.ticker == "AAPL"
        assert call_count == 2


# ============================================================================
# Collect Tests
# ============================================================================


class TestCollect:
    """Tests for the main collect method."""

    async def test_collect_multiple_tickers(self, collector):
        """Should collect fundamentals for multiple tickers."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.collect(["AAPL", "MSFT", "GOOGL"])

        assert isinstance(result, CollectorResult)
        assert result.collector_name == "fundamentals"
        assert result.records_collected == 3
        assert result.success is True
        assert len(result.errors) == 0

    async def test_collect_empty_list(self, collector):
        """Should handle empty ticker list."""
        result = await collector.collect([])

        assert result.records_collected == 0
        assert result.success is True

    async def test_collect_caches_results(self, collector):
        """Should cache collected fundamentals."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            await collector.collect(["AAPL", "MSFT"])

        assert "AAPL" in collector._fundamentals_cache
        assert "MSFT" in collector._fundamentals_cache

    async def test_collect_handles_partial_failures(self, collector):
        """Should continue on partial failures."""
        call_count = 0

        def create_ticker(symbol):
            nonlocal call_count
            call_count += 1
            if symbol == "BAD":
                raise Exception("API Error")
            mock = MagicMock()
            mock.info = SAMPLE_YFINANCE_INFO_MINIMAL
            mock.financials = pd.DataFrame()
            return mock

        with patch("yfinance.Ticker", side_effect=create_ticker):
            # BAD will fail after retries
            collector.max_retries = 0  # Disable retries for faster test
            result = await collector.collect(["AAPL", "BAD", "MSFT"])

        assert result.records_collected == 2
        assert result.success is False  # Has errors
        assert len(result.errors) == 1
        assert "BAD" in result.errors[0]

    async def test_collect_includes_timing(self, collector):
        """Should include timing information."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.collect(["AAPL"])

        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at


# ============================================================================
# Collect Single Tests
# ============================================================================


class TestCollectSingle:
    """Tests for the collect_single method."""

    async def test_collect_single_success(self, collector):
        """Should collect fundamentals for a single ticker."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.collect_single("AAPL")

        assert result == 1
        assert "AAPL" in collector._fundamentals_cache

    async def test_collect_single_caches_result(self, collector):
        """Should cache the collected fundamentals."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            await collector.collect_single("MSFT")

        cached = collector.get_cached_fundamentals("MSFT")
        assert cached is not None
        assert cached.ticker == "MSFT"

    async def test_collect_single_raises_on_error(self, collector):
        """Should raise exception on persistent error."""
        collector.max_retries = 0

        with patch("yfinance.Ticker", side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await collector.collect_single("AAPL")


# ============================================================================
# Cache Management Tests
# ============================================================================


class TestCacheManagement:
    """Tests for cache management methods."""

    async def test_get_cached_fundamentals_exists(self, collector):
        """Should return cached fundamentals."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            await collector.collect_single("AAPL")

        result = collector.get_cached_fundamentals("AAPL")
        assert result is not None
        assert result.ticker == "AAPL"

    def test_get_cached_fundamentals_not_exists(self, collector):
        """Should return None for non-cached ticker."""
        result = collector.get_cached_fundamentals("NONEXISTENT")
        assert result is None

    def test_get_cached_fundamentals_case_insensitive(self, collector):
        """Should be case-insensitive."""
        collector._fundamentals_cache["AAPL"] = FundamentalsData(ticker="AAPL")

        result = collector.get_cached_fundamentals("aapl")
        assert result is not None

    async def test_get_all_cached_fundamentals(self, collector):
        """Should return all cached fundamentals."""
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO_MINIMAL
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            await collector.collect(["AAPL", "MSFT"])

        all_cached = collector.get_all_cached_fundamentals()
        assert len(all_cached) == 2
        assert "AAPL" in all_cached
        assert "MSFT" in all_cached

    def test_get_all_cached_fundamentals_returns_copy(self, collector):
        """Should return a copy of the cache."""
        collector._fundamentals_cache["AAPL"] = FundamentalsData(ticker="AAPL")

        all_cached = collector.get_all_cached_fundamentals()
        all_cached["NEW"] = FundamentalsData(ticker="NEW")

        # Original cache should not be modified
        assert "NEW" not in collector._fundamentals_cache

    def test_clear_cache(self, collector):
        """Should clear all cached data."""
        collector._fundamentals_cache["AAPL"] = FundamentalsData(ticker="AAPL")
        collector._fundamentals_cache["MSFT"] = FundamentalsData(ticker="MSFT")

        collector.clear_cache()

        assert len(collector._fundamentals_cache) == 0


# ============================================================================
# Missing Data Handling Tests
# ============================================================================


class TestMissingDataHandling:
    """Tests for handling missing or incomplete data."""

    async def test_handles_empty_info_dict(self, collector):
        """Should handle completely empty info dict."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("UNKNOWN")

        assert result.ticker == "UNKNOWN"
        assert result.pe_ratio is None
        assert result.sector is None
        assert result.market_cap is None

    async def test_handles_partial_info(self, collector):
        """Should handle partial info data."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 25.0,
            "sector": "Technology",
            # Missing many other fields
        }
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("PARTIAL")

        assert result.pe_ratio == 25.0
        assert result.sector == "Technology"
        assert result.pb_ratio is None
        assert result.dividend_yield is None

    async def test_handles_invalid_types_gracefully(self, collector):
        """Should handle unexpected types in info dict."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": "not a number",  # String instead of float
            "sector": 123,  # Number instead of string
            "marketCap": [],  # List instead of number
        }
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            # Should not raise exception
            result = await collector.get_fundamentals("INVALID")

        assert result.ticker == "INVALID"
        # Invalid types should be converted to None
        assert result.pe_ratio is None  # String converted to None
        assert result.sector is None  # Int converted to None
        assert result.market_cap is None  # List converted to None

    async def test_handles_financials_with_missing_rows(self, collector):
        """Should handle financials DataFrame missing expected rows."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = pd.DataFrame(
            {"2024-12-31": [100e9]},
            index=["Operating Income"],  # Missing "Total Revenue"
        )

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("MISSING")

        assert result.revenue_growth_3y is None


# ============================================================================
# Data Normalization Tests
# ============================================================================


class TestDataNormalization:
    """Tests for data normalization."""

    async def test_ticker_uppercase_normalization(self, collector):
        """Should normalize ticker to uppercase."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"sector": "Tech"}
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("aapl")

        assert result.ticker == "AAPL"

    async def test_percentage_values_as_decimals(self, collector):
        """Should store percentage values as decimals."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "dividendYield": 0.025,  # 2.5%
            "profitMargins": 0.15,  # 15%
            "returnOnEquity": 0.35,  # 35%
        }
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("AAPL")

        assert result.dividend_yield == 0.025
        assert result.profit_margin == 0.15
        assert result.roe == 0.35

    async def test_large_number_handling(self, collector):
        """Should handle large numbers correctly."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "marketCap": 3000000000000,  # 3 trillion
            "freeCashflow": 100000000000,  # 100 billion
        }
        mock_ticker.financials = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await collector.get_fundamentals("AAPL")

        assert result.market_cap == 3000000000000
        assert result.free_cash_flow == 100000000000


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests for FundamentalsCollector."""

    async def test_full_workflow(self, mock_session, mock_config):
        """Should execute full collection workflow."""
        collector = FundamentalsCollector(mock_session, mock_config)

        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO
        mock_ticker.financials = SAMPLE_FINANCIALS

        with patch("yfinance.Ticker", return_value=mock_ticker):
            # Collect for multiple tickers
            result = await collector.collect(["AAPL", "MSFT"])

            # Verify result
            assert result.success is True
            assert result.records_collected == 2

            # Verify cache
            aapl = collector.get_cached_fundamentals("AAPL")
            assert aapl is not None
            assert aapl.pe_ratio == 25.5

            msft = collector.get_cached_fundamentals("MSFT")
            assert msft is not None

            # Verify all cached
            all_data = collector.get_all_cached_fundamentals()
            assert len(all_data) == 2

            # Clear and verify
            collector.clear_cache()
            assert len(collector._fundamentals_cache) == 0

    async def test_rate_limiting_respected(self, mock_session, mock_config):
        """Should respect rate limiting."""
        collector = FundamentalsCollector(mock_session, mock_config)
        collector.rate_limit_calls = 2
        collector.rate_limit_period = 60

        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.financials = pd.DataFrame()

        call_times = []

        def track_call(symbol):
            import time
            call_times.append(time.monotonic())
            return mock_ticker

        with patch("yfinance.Ticker", side_effect=track_call):
            await collector.collect(["AAPL", "MSFT"])

        # Both calls should complete (under rate limit)
        assert len(call_times) == 2
