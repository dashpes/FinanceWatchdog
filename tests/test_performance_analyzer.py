"""Tests for PerformanceAnalyzer class."""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from investment_monitor.models import ScoringWeights
from investment_monitor.storage.models import Base
from investment_monitor.storage.research_models import (
    CandidateScore,
    PerformanceTracker,
)
from investment_monitor.storage.research_operations import (
    save_performance_record,
    save_score,
)
from investment_monitor.research.performance import PerformanceAnalyzer


@pytest.fixture
def db_session():
    """Create a temporary in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def performance_analyzer(db_session):
    """Create a PerformanceAnalyzer instance."""
    return PerformanceAnalyzer(session=db_session)


class TestUpdatePerformanceData:
    """Tests for update_performance_data method."""

    @pytest.mark.asyncio
    async def test_update_performance_data_fetches_prices_and_updates_records(
        self, db_session, performance_analyzer
    ):
        """Test that update_performance_data fetches prices and updates records."""
        # Create a performance record that needs update (old updated_at)
        record = PerformanceTracker(
            ticker="AAPL",
            entry_date=date.today() - timedelta(days=40),
            entry_price=150.0,
        )
        save_performance_record(db_session, record)
        db_session.commit()

        # Make it need an update by setting updated_at in the past
        old_time = datetime.now() - timedelta(days=10)
        record.updated_at = old_time
        db_session.commit()

        # Mock price fetching to return a current price
        with patch.object(
            performance_analyzer,
            "_fetch_current_price",
            new_callable=AsyncMock,
            return_value=165.0,
        ):
            count = await performance_analyzer.update_performance_data()

        # Should have updated 1 record
        assert count == 1

        # Verify the record was updated
        db_session.refresh(record)
        assert record.current_price == 165.0
        # 30-day return: (165 - 150) / 150 = 0.10 = 10%
        assert record.return_30d == pytest.approx(10.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_update_performance_data_calculates_30_60_90_day_returns(
        self, db_session, performance_analyzer
    ):
        """Test that returns are calculated for 30/60/90 day periods."""
        # Create a record with entry_date 100 days ago
        entry_date = date.today() - timedelta(days=100)
        record = PerformanceTracker(
            ticker="MSFT",
            entry_date=entry_date,
            entry_price=200.0,
        )
        save_performance_record(db_session, record)
        db_session.commit()

        # Set old updated_at
        record.updated_at = datetime.now() - timedelta(days=10)
        db_session.commit()

        # Current price represents 25% gain
        with patch.object(
            performance_analyzer,
            "_fetch_current_price",
            new_callable=AsyncMock,
            return_value=250.0,
        ):
            await performance_analyzer.update_performance_data()

        db_session.refresh(record)
        # All periods should show 25% return since entry was 100 days ago
        assert record.return_30d == pytest.approx(25.0, rel=0.01)
        assert record.return_60d == pytest.approx(25.0, rel=0.01)
        assert record.return_90d == pytest.approx(25.0, rel=0.01)
        assert record.current_price == 250.0

    @pytest.mark.asyncio
    async def test_update_performance_data_returns_zero_when_no_records_need_update(
        self, db_session, performance_analyzer
    ):
        """Test returns 0 when no records need update."""
        # Create a recently updated record
        record = PerformanceTracker(
            ticker="GOOGL",
            entry_date=date.today() - timedelta(days=30),
            entry_price=100.0,
        )
        save_performance_record(db_session, record)
        db_session.commit()

        # Don't modify updated_at - it's recent by default
        count = await performance_analyzer.update_performance_data()
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_performance_data_handles_price_fetch_failure(
        self, db_session, performance_analyzer
    ):
        """Test graceful handling when price fetch fails for a ticker."""
        record = PerformanceTracker(
            ticker="INVALID",
            entry_date=date.today() - timedelta(days=30),
            entry_price=100.0,
        )
        save_performance_record(db_session, record)
        db_session.commit()

        record.updated_at = datetime.now() - timedelta(days=10)
        db_session.commit()

        # Price fetch returns None (failure)
        with patch.object(
            performance_analyzer,
            "_fetch_current_price",
            new_callable=AsyncMock,
            return_value=None,
        ):
            count = await performance_analyzer.update_performance_data()

        # Should not count as updated since we couldn't get the price
        assert count == 0


class TestAnalyzeFactorPerformance:
    """Tests for analyze_factor_performance method."""

    def test_analyze_factor_performance_calculates_correlations(self, db_session, performance_analyzer):
        """Test that correlations are calculated between factor scores and returns."""
        # Create candidates with scores and performance data
        # High value scores correlate with high returns
        test_data = [
            {"ticker": "AAPL", "value": 90.0, "growth": 60.0, "return": 20.0},
            {"ticker": "MSFT", "value": 80.0, "growth": 70.0, "return": 15.0},
            {"ticker": "GOOGL", "value": 70.0, "growth": 80.0, "return": 10.0},
            {"ticker": "AMZN", "value": 60.0, "growth": 90.0, "return": 5.0},
        ]

        for data in test_data:
            # Create score
            score = CandidateScore(
                ticker=data["ticker"],
                value_score=data["value"],
                growth_score=data["growth"],
                quality_score=75.0,
                momentum_score=70.0,
                sentiment_score=65.0,
                composite_score=72.0,
            )
            save_score(db_session, score)

            # Create performance tracker with return
            perf = PerformanceTracker(
                ticker=data["ticker"],
                entry_date=date.today() - timedelta(days=60),
                entry_price=100.0,
                return_30d=data["return"],
            )
            save_performance_record(db_session, perf)

        db_session.commit()

        result = performance_analyzer.analyze_factor_performance()

        # Should return correlation dict
        assert "value" in result
        assert "growth" in result
        assert "quality" in result
        assert "momentum" in result
        assert "sentiment" in result

        # Value should have positive correlation with returns
        assert result["value"] > 0
        # Growth should have negative correlation (inversely related in test data)
        assert result["growth"] < 0

    def test_analyze_factor_performance_returns_empty_with_no_data(
        self, db_session, performance_analyzer
    ):
        """Test returns empty dict when no matching data exists."""
        result = performance_analyzer.analyze_factor_performance()
        assert result == {}

    def test_analyze_factor_performance_ignores_missing_returns(
        self, db_session, performance_analyzer
    ):
        """Test that candidates without return data are excluded."""
        # Score with no performance data
        score = CandidateScore(
            ticker="NVDA",
            value_score=85.0,
            growth_score=80.0,
            quality_score=75.0,
            momentum_score=70.0,
            sentiment_score=65.0,
            composite_score=75.0,
        )
        save_score(db_session, score)
        db_session.commit()

        result = performance_analyzer.analyze_factor_performance()
        assert result == {}


class TestSuggestWeightAdjustments:
    """Tests for suggest_weight_adjustments method."""

    def test_suggest_weight_adjustments_returns_none_with_insufficient_data(
        self, db_session, performance_analyzer
    ):
        """Test returns None when not enough data (less than 20 candidates)."""
        # Create only 5 candidates
        for i in range(5):
            score = CandidateScore(
                ticker=f"TICK{i}",
                value_score=75.0,
                growth_score=70.0,
                quality_score=80.0,
                momentum_score=65.0,
                sentiment_score=60.0,
                composite_score=70.0,
            )
            save_score(db_session, score)

            perf = PerformanceTracker(
                ticker=f"TICK{i}",
                entry_date=date.today() - timedelta(days=60),
                entry_price=100.0,
                return_30d=10.0,
            )
            save_performance_record(db_session, perf)

        db_session.commit()

        result = performance_analyzer.suggest_weight_adjustments()
        assert result is None

    def test_suggest_weight_adjustments_returns_weights_with_sufficient_data(
        self, db_session, performance_analyzer
    ):
        """Test returns ScoringWeights when enough data exists."""
        # Create 25 candidates with scores and performance
        for i in range(25):
            score = CandidateScore(
                ticker=f"TICK{i:02d}",
                value_score=70.0 + (i % 5) * 5,
                growth_score=65.0 + (i % 4) * 5,
                quality_score=75.0 + (i % 3) * 5,
                momentum_score=60.0 + (i % 6) * 5,
                sentiment_score=55.0 + (i % 7) * 5,
                composite_score=65.0 + i,
            )
            save_score(db_session, score)

            perf = PerformanceTracker(
                ticker=f"TICK{i:02d}",
                entry_date=date.today() - timedelta(days=60),
                entry_price=100.0,
                return_30d=5.0 + i * 0.5,
            )
            save_performance_record(db_session, perf)

        db_session.commit()

        result = performance_analyzer.suggest_weight_adjustments()

        # Should return ScoringWeights
        assert result is not None
        assert isinstance(result, ScoringWeights)

    def test_suggested_weights_sum_to_one(self, db_session, performance_analyzer):
        """Test that suggested weights sum to 1.0."""
        # Create 25 candidates with varied scores and returns
        for i in range(25):
            score = CandidateScore(
                ticker=f"SUM{i:02d}",
                value_score=60.0 + i * 1.5,
                growth_score=55.0 + i * 1.2,
                quality_score=70.0 + i * 0.8,
                momentum_score=65.0 + i * 1.0,
                sentiment_score=50.0 + i * 1.3,
                composite_score=60.0 + i,
            )
            save_score(db_session, score)

            perf = PerformanceTracker(
                ticker=f"SUM{i:02d}",
                entry_date=date.today() - timedelta(days=60),
                entry_price=100.0,
                return_30d=2.0 + i * 0.8,
            )
            save_performance_record(db_session, perf)

        db_session.commit()

        result = performance_analyzer.suggest_weight_adjustments()

        assert result is not None
        total = (
            result.value
            + result.growth
            + result.quality
            + result.momentum
            + result.sentiment
        )
        assert total == pytest.approx(1.0, rel=0.01)

    def test_suggested_weights_are_bounded(self, db_session, performance_analyzer):
        """Test that all suggested weights are between 0 and 1."""
        # Create 25 candidates
        for i in range(25):
            score = CandidateScore(
                ticker=f"BND{i:02d}",
                value_score=50.0 + i * 2,
                growth_score=45.0 + i * 1.5,
                quality_score=55.0 + i * 1.8,
                momentum_score=40.0 + i * 2.2,
                sentiment_score=60.0 + i * 1.0,
                composite_score=50.0 + i,
            )
            save_score(db_session, score)

            perf = PerformanceTracker(
                ticker=f"BND{i:02d}",
                entry_date=date.today() - timedelta(days=60),
                entry_price=100.0,
                return_30d=1.0 + i * 0.6,
            )
            save_performance_record(db_session, perf)

        db_session.commit()

        result = performance_analyzer.suggest_weight_adjustments()

        assert result is not None
        assert 0 <= result.value <= 1
        assert 0 <= result.growth <= 1
        assert 0 <= result.quality <= 1
        assert 0 <= result.momentum <= 1
        assert 0 <= result.sentiment <= 1
