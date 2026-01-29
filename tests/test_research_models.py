"""Tests for research database models."""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from investment_monitor.storage.models import Base
from investment_monitor.storage.research_models import (
    CANDIDATE_STATUSES,
    CandidateScore,
    CongressionalTrade,
    PerformanceTracker,
    ResearchProfile,
    ResearchReport,
    StockCandidate,
)


@pytest.fixture
def db_session():
    """Create a temporary in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class TestResearchProfile:
    """Tests for ResearchProfile model."""

    def test_create_profile(self, db_session):
        """Test creating a research profile."""
        profile = ResearchProfile(
            name="default",
            investment_style="value",
            risk_tolerance="moderate",
            sector_preferences='["technology", "healthcare"]',
            value_weight=0.3,
            growth_weight=0.2,
            quality_weight=0.2,
            momentum_weight=0.15,
            sentiment_weight=0.15,
        )
        db_session.add(profile)
        db_session.commit()

        retrieved = db_session.query(ResearchProfile).filter_by(name="default").first()
        assert retrieved is not None
        assert retrieved.investment_style == "value"
        assert retrieved.risk_tolerance == "moderate"
        assert retrieved.value_weight == 0.3

    def test_profile_defaults(self, db_session):
        """Test profile default values."""
        profile = ResearchProfile(name="test_defaults")
        db_session.add(profile)
        db_session.commit()

        retrieved = db_session.query(ResearchProfile).filter_by(name="test_defaults").first()
        assert retrieved.value_weight == 0.2
        assert retrieved.growth_weight == 0.2
        assert retrieved.quality_weight == 0.2
        assert retrieved.momentum_weight == 0.2
        assert retrieved.sentiment_weight == 0.2

    def test_profile_unique_name_constraint(self, db_session):
        """Test that profile name must be unique."""
        profile1 = ResearchProfile(name="unique_test")
        db_session.add(profile1)
        db_session.commit()

        profile2 = ResearchProfile(name="unique_test")
        db_session.add(profile2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_profile_timestamps(self, db_session):
        """Test that timestamps are set automatically."""
        profile = ResearchProfile(name="timestamp_test")
        db_session.add(profile)
        db_session.commit()

        retrieved = db_session.query(ResearchProfile).filter_by(name="timestamp_test").first()
        assert retrieved.created_at is not None
        assert retrieved.updated_at is not None


class TestStockCandidate:
    """Tests for StockCandidate model."""

    def test_create_candidate(self, db_session):
        """Test creating a stock candidate."""
        candidate = StockCandidate(
            ticker="AAPL",
            discovery_source="congressional_trades",
            status="discovered",
            composite_score=75.5,
            notes="Strong buy signals from multiple congress members",
        )
        db_session.add(candidate)
        db_session.commit()

        retrieved = db_session.query(StockCandidate).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.discovery_source == "congressional_trades"
        assert retrieved.status == "discovered"
        assert retrieved.composite_score == 75.5

    def test_candidate_default_status(self, db_session):
        """Test that default status is 'discovered'."""
        candidate = StockCandidate(ticker="MSFT")
        db_session.add(candidate)
        db_session.commit()

        retrieved = db_session.query(StockCandidate).filter_by(ticker="MSFT").first()
        assert retrieved.status == "discovered"

    def test_candidate_unique_ticker_constraint(self, db_session):
        """Test that ticker must be unique."""
        candidate1 = StockCandidate(ticker="GOOGL")
        db_session.add(candidate1)
        db_session.commit()

        candidate2 = StockCandidate(ticker="GOOGL")
        db_session.add(candidate2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_candidate_valid_statuses(self):
        """Test that valid statuses are defined correctly."""
        expected = (
            "discovered",
            "screening",
            "researched",
            "watchlist",
            "rejected",
            "archived",
        )
        assert CANDIDATE_STATUSES == expected


class TestCandidateScore:
    """Tests for CandidateScore model."""

    def test_create_score(self, db_session):
        """Test creating a candidate score."""
        score = CandidateScore(
            ticker="AAPL",
            value_score=80.0,
            growth_score=75.0,
            quality_score=85.0,
            momentum_score=70.0,
            sentiment_score=90.0,
            composite_score=80.0,
            reasoning="Strong fundamentals with positive momentum",
        )
        db_session.add(score)
        db_session.commit()

        retrieved = db_session.query(CandidateScore).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.value_score == 80.0
        assert retrieved.growth_score == 75.0
        assert retrieved.quality_score == 85.0
        assert retrieved.momentum_score == 70.0
        assert retrieved.sentiment_score == 90.0
        assert retrieved.composite_score == 80.0

    def test_score_unique_ticker_constraint(self, db_session):
        """Test that ticker must be unique in scores."""
        score1 = CandidateScore(ticker="NVDA", value_score=85.0)
        db_session.add(score1)
        db_session.commit()

        score2 = CandidateScore(ticker="NVDA", value_score=90.0)
        db_session.add(score2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_score_range_values(self, db_session):
        """Test that scores can be in 0-100 range."""
        score = CandidateScore(
            ticker="TSLA",
            value_score=0.0,
            growth_score=100.0,
            quality_score=50.0,
        )
        db_session.add(score)
        db_session.commit()

        retrieved = db_session.query(CandidateScore).filter_by(ticker="TSLA").first()
        assert retrieved.value_score == 0.0
        assert retrieved.growth_score == 100.0
        assert retrieved.quality_score == 50.0


class TestResearchReport:
    """Tests for ResearchReport model."""

    def test_create_report(self, db_session):
        """Test creating a research report."""
        report = ResearchReport(
            ticker="AAPL",
            summary="Apple is a strong buy based on services growth",
            bull_case="Services revenue continues to grow at 20%+ YoY",
            bear_case="Hardware sales may slow in economic downturn",
            thesis="Long-term hold based on ecosystem strength",
            recommendation="BUY",
            target_price=200.0,
        )
        db_session.add(report)
        db_session.commit()

        retrieved = db_session.query(ResearchReport).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.recommendation == "BUY"
        assert retrieved.target_price == 200.0

    def test_multiple_reports_same_ticker(self, db_session):
        """Test that multiple reports for same ticker are allowed."""
        report1 = ResearchReport(ticker="META", recommendation="HOLD")
        report2 = ResearchReport(ticker="META", recommendation="BUY")
        db_session.add(report1)
        db_session.add(report2)
        db_session.commit()

        reports = db_session.query(ResearchReport).filter_by(ticker="META").all()
        assert len(reports) == 2


class TestPerformanceTracker:
    """Tests for PerformanceTracker model."""

    def test_create_tracker(self, db_session):
        """Test creating a performance tracker."""
        tracker = PerformanceTracker(
            ticker="AAPL",
            entry_date=date(2026, 1, 1),
            entry_price=175.0,
            return_30d=5.5,
            return_60d=8.2,
            return_90d=12.0,
            current_price=196.0,
        )
        db_session.add(tracker)
        db_session.commit()

        retrieved = db_session.query(PerformanceTracker).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.entry_price == 175.0
        assert retrieved.return_30d == 5.5
        assert retrieved.return_60d == 8.2
        assert retrieved.return_90d == 12.0

    def test_tracker_unique_ticker_date_constraint(self, db_session):
        """Test that ticker + entry_date must be unique."""
        tracker1 = PerformanceTracker(
            ticker="MSFT",
            entry_date=date(2026, 1, 15),
            entry_price=400.0,
        )
        db_session.add(tracker1)
        db_session.commit()

        tracker2 = PerformanceTracker(
            ticker="MSFT",
            entry_date=date(2026, 1, 15),
            entry_price=405.0,
        )
        db_session.add(tracker2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_tracker_same_ticker_different_dates(self, db_session):
        """Test that same ticker with different dates is allowed."""
        tracker1 = PerformanceTracker(
            ticker="GOOGL",
            entry_date=date(2026, 1, 1),
            entry_price=140.0,
        )
        tracker2 = PerformanceTracker(
            ticker="GOOGL",
            entry_date=date(2026, 2, 1),
            entry_price=145.0,
        )
        db_session.add(tracker1)
        db_session.add(tracker2)
        db_session.commit()

        trackers = db_session.query(PerformanceTracker).filter_by(ticker="GOOGL").all()
        assert len(trackers) == 2


class TestCongressionalTrade:
    """Tests for CongressionalTrade model."""

    def test_create_trade(self, db_session):
        """Test creating a congressional trade."""
        trade = CongressionalTrade(
            ticker="NVDA",
            politician="Nancy Pelosi",
            party="Democrat",
            chamber="House",
            trade_type="buy",
            amount_range="$1,001-$15,000",
            trade_date=date(2026, 1, 10),
            disclosure_date=date(2026, 1, 25),
            description="Purchase of NVDA stock",
            source_url="https://efdsearch.senate.gov/trade/123",
        )
        db_session.add(trade)
        db_session.commit()

        retrieved = db_session.query(CongressionalTrade).filter_by(politician="Nancy Pelosi").first()
        assert retrieved is not None
        assert retrieved.ticker == "NVDA"
        assert retrieved.amount_range == "$1,001-$15,000"
        assert retrieved.trade_type == "buy"

    def test_trade_amount_range_format(self, db_session):
        """Test that amount_range can store disclosure format strings."""
        ranges = [
            "$1,001-$15,000",
            "$15,001-$50,000",
            "$50,001-$100,000",
            "$100,001-$250,000",
            "$250,001-$500,000",
            "$500,001-$1,000,000",
            "Over $1,000,000",
        ]

        for i, amount_range in enumerate(ranges):
            trade = CongressionalTrade(
                ticker=f"TEST{i}",
                politician=f"Politician {i}",
                trade_type="buy",
                amount_range=amount_range,
                trade_date=date(2026, 1, i + 1),
            )
            db_session.add(trade)

        db_session.commit()

        trades = db_session.query(CongressionalTrade).all()
        assert len(trades) == len(ranges)

    def test_trade_unique_constraint(self, db_session):
        """Test that same trade cannot be recorded twice."""
        trade1 = CongressionalTrade(
            ticker="AAPL",
            politician="John Smith",
            trade_type="buy",
            amount_range="$1,001-$15,000",
            trade_date=date(2026, 1, 20),
        )
        db_session.add(trade1)
        db_session.commit()

        trade2 = CongressionalTrade(
            ticker="AAPL",
            politician="John Smith",
            trade_type="buy",
            amount_range="$1,001-$15,000",
            trade_date=date(2026, 1, 20),
        )
        db_session.add(trade2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_trade_same_politician_different_trades(self, db_session):
        """Test that same politician can have multiple different trades."""
        trade1 = CongressionalTrade(
            ticker="AAPL",
            politician="Jane Doe",
            trade_type="buy",
            amount_range="$1,001-$15,000",
            trade_date=date(2026, 1, 10),
        )
        trade2 = CongressionalTrade(
            ticker="MSFT",
            politician="Jane Doe",
            trade_type="sell",
            amount_range="$15,001-$50,000",
            trade_date=date(2026, 1, 15),
        )
        db_session.add(trade1)
        db_session.add(trade2)
        db_session.commit()

        trades = db_session.query(CongressionalTrade).filter_by(politician="Jane Doe").all()
        assert len(trades) == 2


class TestModelTimestamps:
    """Test timestamp behavior across all models."""

    def test_all_models_have_created_at(self, db_session):
        """Test that all models get created_at timestamp."""
        # Create one of each model
        profile = ResearchProfile(name="ts_test")
        candidate = StockCandidate(ticker="TSTEST")
        score = CandidateScore(ticker="SCTEST")
        report = ResearchReport(ticker="RPTEST")
        tracker = PerformanceTracker(
            ticker="PKTEST", entry_date=date(2026, 1, 1), entry_price=100.0
        )
        trade = CongressionalTrade(
            ticker="CTTEST",
            politician="Test Person",
            trade_type="buy",
            amount_range="$1,001-$15,000",
            trade_date=date(2026, 1, 1),
        )

        db_session.add_all([profile, candidate, score, report, tracker, trade])
        db_session.commit()

        # Verify all have created_at
        assert db_session.query(ResearchProfile).first().created_at is not None
        assert db_session.query(StockCandidate).first().created_at is not None
        assert db_session.query(CandidateScore).first().created_at is not None
        assert db_session.query(ResearchReport).first().created_at is not None
        assert db_session.query(PerformanceTracker).first().created_at is not None
        assert db_session.query(CongressionalTrade).first().created_at is not None
