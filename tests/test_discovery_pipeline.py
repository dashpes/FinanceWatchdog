"""Tests for the discovery pipeline."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.analysis import ResearchScorer
from investment_monitor.collectors import (
    CollectorResult,
    FundamentalsCollector,
    FundamentalsData,
    UniverseCollector,
)
from investment_monitor.config import Settings
from investment_monitor.models import ResearchConfig, ScoringWeights
from investment_monitor.research import DiscoveryPipeline, DiscoveryResult
from investment_monitor.storage import (
    CandidateScore,
    StockCandidate,
    get_session,
    init_db,
    save_candidate,
)


@pytest.fixture(autouse=True)
def _no_network_prices():
    """Stub the pipeline's price-history fetch (step 3.5) so tests never hit yfinance."""
    with patch(
        "investment_monitor.research.discovery.PriceCollector.collect",
        new_callable=AsyncMock,
        return_value=CollectorResult(
            collector_name="prices", success=True, records_collected=0, errors=[],
            started_at=datetime.now(), finished_at=datetime.now(),
        ),
    ):
        yield


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
def research_config():
    """Create test research config."""
    return ResearchConfig(
        scoring_weights=ScoringWeights(
            value=0.2,
            growth=0.2,
            quality=0.2,
            momentum=0.2,
            sentiment=0.2,
        ),
        discovery_batch_size=10,
    )


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamentals data."""
    return FundamentalsData(
        ticker="AAPL",
        pe_ratio=25.5,
        pb_ratio=45.2,
        ps_ratio=7.1,
        peg_ratio=2.3,
        revenue_growth_yoy=0.08,
        revenue_growth_3y=0.15,
        eps_growth_yoy=0.12,
        eps_growth_3y=0.18,
        roe=0.35,
        profit_margin=0.25,
        debt_to_equity=1.5,
        current_ratio=1.2,
        dividend_yield=0.005,
        payout_ratio=0.15,
        free_cash_flow=100_000_000_000,
        market_cap=3_000_000_000_000,
        sector="Technology",
        industry="Consumer Electronics",
    )


@pytest.fixture
def mock_score():
    """Create a mock CandidateScore."""
    return CandidateScore(
        ticker="AAPL",
        value_score=75.0,
        growth_score=80.0,
        quality_score=85.0,
        momentum_score=70.0,
        sentiment_score=65.0,
        composite_score=75.0,
        reasoning="Test reasoning",
    )


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = DiscoveryResult()

        assert result.total_candidates == 0
        assert result.scored_candidates == 0
        assert result.top_candidates == []
        assert result.watchlist_additions == []
        assert result.errors == []
        assert result.finished_at is None

    def test_duration_seconds(self):
        """Test duration calculation."""
        result = DiscoveryResult()
        result.started_at = datetime(2026, 1, 1, 12, 0, 0)
        result.finished_at = datetime(2026, 1, 1, 12, 0, 30)

        assert result.duration_seconds == 30.0

    def test_duration_seconds_no_finish(self):
        """Test duration returns 0 if not finished."""
        result = DiscoveryResult()
        assert result.duration_seconds == 0.0

    def test_success_with_scored_candidates(self):
        """Test success is True when candidates are scored."""
        result = DiscoveryResult()
        result.scored_candidates = 5
        assert result.success is True

    def test_success_with_no_errors(self):
        """Test success is True when no errors."""
        result = DiscoveryResult()
        assert result.success is True

    def test_success_with_errors_but_scored(self):
        """Test success is True if some candidates scored despite errors."""
        result = DiscoveryResult()
        result.scored_candidates = 3
        result.errors = ["Some error"]
        assert result.success is True


class TestDiscoveryPipeline:
    """Tests for DiscoveryPipeline class."""

    def test_init(self, db_session, settings, research_config):
        """Test pipeline initialization."""
        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        assert pipeline.session == db_session
        assert pipeline.config == settings
        assert pipeline.research_config == research_config
        assert pipeline.universe_collector is not None
        assert pipeline.fundamentals_collector is not None
        assert pipeline.scorer is not None

    def test_init_with_custom_model(self, db_session, settings, research_config):
        """Test pipeline initialization with custom Ollama model."""
        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
            ollama_model="llama3:8b",
        )

        assert pipeline.scorer.model == "llama3:8b"

    @pytest.mark.asyncio
    async def test_run_discovery_dry_run(
        self, db_session, settings, research_config, sample_fundamentals, mock_score
    ):
        """Test discovery pipeline in dry run mode."""
        # Create some candidates in the database
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            candidate = StockCandidate(
                ticker=ticker,
                discovery_source="test",
                status="discovered",
            )
            save_candidate(db_session, candidate)
        db_session.commit()

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        # Mock the collectors and scorer
        with patch.object(
            pipeline.universe_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_universe, patch.object(
            pipeline.fundamentals_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_fundamentals, patch.object(
            pipeline.fundamentals_collector,
            "get_cached_fundamentals",
        ) as mock_get_fundamentals, patch.object(
            pipeline.scorer,
            "score_stock",
            new_callable=AsyncMock,
        ) as mock_scorer:
            # Setup mocks
            mock_universe.return_value = CollectorResult(
                collector_name="universe",
                success=True,
                records_collected=3,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_fundamentals.return_value = CollectorResult(
                collector_name="fundamentals",
                success=True,
                records_collected=3,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_get_fundamentals.return_value = sample_fundamentals
            mock_scorer.return_value = mock_score

            # Run pipeline
            result = await pipeline.run_discovery(dry_run=True)

            # Verify result
            assert result.total_candidates == 3
            assert result.scored_candidates == 3
            assert result.finished_at is not None
            assert len(result.errors) == 0
            # In dry run, nothing should be added to watchlist
            assert len(result.watchlist_additions) == 0

    @pytest.mark.asyncio
    async def test_run_discovery_with_scoring_error(
        self, db_session, settings, research_config, sample_fundamentals
    ):
        """Test discovery pipeline handles scoring errors gracefully."""
        # Create a candidate
        candidate = StockCandidate(
            ticker="AAPL",
            discovery_source="test",
            status="discovered",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        # Mock collectors and scorer
        with patch.object(
            pipeline.universe_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_universe, patch.object(
            pipeline.fundamentals_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_fundamentals, patch.object(
            pipeline.fundamentals_collector,
            "get_cached_fundamentals",
        ) as mock_get_fundamentals, patch.object(
            pipeline.scorer,
            "score_stock",
            new_callable=AsyncMock,
        ) as mock_scorer:
            mock_universe.return_value = CollectorResult(
                collector_name="universe",
                success=True,
                records_collected=1,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_fundamentals.return_value = CollectorResult(
                collector_name="fundamentals",
                success=True,
                records_collected=1,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_get_fundamentals.return_value = sample_fundamentals
            mock_scorer.side_effect = Exception("Scoring failed")

            # Run pipeline
            result = await pipeline.run_discovery(dry_run=True)

            # Should have error but not crash
            assert len(result.errors) == 1
            assert "AAPL" in result.errors[0]
            assert "Scoring failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_discovery_auto_watchlist(
        self, db_session, settings, sample_fundamentals
    ):
        """Test discovery pipeline auto-adds high-scoring candidates to watchlist."""
        # Create research config with low watchlist threshold
        research_config = ResearchConfig(
            scoring_weights=ScoringWeights(
                value=0.2,
                growth=0.2,
                quality=0.2,
                momentum=0.2,
                sentiment=0.2,
            ),
            discovery_batch_size=10,
        )
        research_config.thresholds.auto_watchlist_score = 70.0
        research_config.thresholds.min_composite_score = 60.0

        # Create a candidate
        candidate = StockCandidate(
            ticker="AAPL",
            discovery_source="test",
            status="discovered",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        # Create high-scoring mock
        high_score = CandidateScore(
            ticker="AAPL",
            value_score=85.0,
            growth_score=80.0,
            quality_score=85.0,
            momentum_score=80.0,
            sentiment_score=75.0,
            composite_score=81.0,  # Above 70.0 threshold
            reasoning="Excellent stock",
        )

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        with patch.object(
            pipeline.universe_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_universe, patch.object(
            pipeline.fundamentals_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_fundamentals, patch.object(
            pipeline.fundamentals_collector,
            "get_cached_fundamentals",
        ) as mock_get_fundamentals, patch.object(
            pipeline.scorer,
            "score_stock",
            new_callable=AsyncMock,
        ) as mock_scorer:
            mock_universe.return_value = CollectorResult(
                collector_name="universe",
                success=True,
                records_collected=1,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_fundamentals.return_value = CollectorResult(
                collector_name="fundamentals",
                success=True,
                records_collected=1,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_get_fundamentals.return_value = sample_fundamentals
            mock_scorer.return_value = high_score

            # Run pipeline (not dry run)
            result = await pipeline.run_discovery(dry_run=False)

            # Verify watchlist addition
            assert "AAPL" in result.watchlist_additions

    def test_apply_filters_excludes_tickers(
        self, db_session, settings, research_config
    ):
        """Test that explicitly excluded tickers are filtered out."""
        research_config.universe.excluded_tickers = ["AAPL", "MSFT"]

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        filtered = pipeline._apply_filters(tickers)

        assert "AAPL" not in filtered
        assert "MSFT" not in filtered
        assert "GOOGL" in filtered
        assert "AMZN" in filtered

    def test_apply_filters_excludes_sectors(
        self, db_session, settings, research_config, sample_fundamentals
    ):
        """Test that excluded sectors are filtered out."""
        research_config.universe.excluded_sectors = ["technology"]

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        # Pre-cache fundamentals
        pipeline.fundamentals_collector._fundamentals_cache["AAPL"] = sample_fundamentals

        tickers = ["AAPL"]
        filtered = pipeline._apply_filters(tickers)

        # AAPL should be filtered out because it's in Technology sector
        assert "AAPL" not in filtered

    def test_apply_filters_market_cap(
        self, db_session, settings, research_config, sample_fundamentals
    ):
        """Test that stocks below market cap threshold are filtered out."""
        research_config.universe.min_market_cap = 5_000_000_000_000  # $5T

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        # Pre-cache fundamentals (AAPL has $3T market cap)
        pipeline.fundamentals_collector._fundamentals_cache["AAPL"] = sample_fundamentals

        tickers = ["AAPL"]
        filtered = pipeline._apply_filters(tickers)

        # AAPL should be filtered out (market cap below $5T threshold)
        assert "AAPL" not in filtered

    def test_get_discovered_tickers(self, db_session, settings, research_config):
        """Test getting discovered tickers from database."""
        # Add some candidates with different statuses
        for ticker, status in [
            ("AAPL", "discovered"),
            ("MSFT", "discovered"),
            ("GOOGL", "watchlist"),
            ("AMZN", "rejected"),
        ]:
            candidate = StockCandidate(
                ticker=ticker,
                discovery_source="test",
                status=status,
            )
            save_candidate(db_session, candidate)
        db_session.commit()

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        discovered = pipeline._get_discovered_tickers()

        # Only "discovered" status should be returned
        assert "AAPL" in discovered
        assert "MSFT" in discovered
        assert "GOOGL" not in discovered
        assert "AMZN" not in discovered

    @pytest.mark.asyncio
    async def test_score_candidate(
        self, db_session, settings, research_config, sample_fundamentals, mock_score
    ):
        """Test scoring a single candidate."""
        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        with patch.object(
            pipeline.scorer,
            "score_stock",
            new_callable=AsyncMock,
        ) as mock_scorer:
            mock_scorer.return_value = mock_score

            score = await pipeline._score_candidate("AAPL", sample_fundamentals)

            assert score.ticker == "AAPL"
            assert score.composite_score == 75.0
            mock_scorer.assert_called_once()

    def test_update_candidate_status(self, db_session, settings, research_config):
        """Test updating candidate status."""
        # Create a candidate
        candidate = StockCandidate(
            ticker="AAPL",
            discovery_source="test",
            status="discovered",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        pipeline._update_candidate_status("AAPL", "watchlist")
        db_session.flush()

        # Verify status was updated
        from investment_monitor.storage import get_candidate_by_ticker
        updated = get_candidate_by_ticker(db_session, "AAPL")
        assert updated.status == "watchlist"


class TestDiscoveryPipelineIntegration:
    """Integration tests for the discovery pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(
        self, db_session, settings, research_config, sample_fundamentals, mock_score
    ):
        """Test full pipeline flow with mocked external dependencies."""
        # Setup: Create discovered candidates
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            candidate = StockCandidate(
                ticker=ticker,
                discovery_source="sp500",
                status="discovered",
            )
            save_candidate(db_session, candidate)
        db_session.commit()

        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        # Create different scores for each ticker
        scores = {
            "AAPL": CandidateScore(
                ticker="AAPL",
                value_score=80.0,
                growth_score=85.0,
                quality_score=90.0,
                momentum_score=75.0,
                sentiment_score=70.0,
                composite_score=80.0,
                reasoning="Strong fundamentals",
            ),
            "MSFT": CandidateScore(
                ticker="MSFT",
                value_score=75.0,
                growth_score=80.0,
                quality_score=85.0,
                momentum_score=70.0,
                sentiment_score=65.0,
                composite_score=75.0,
                reasoning="Good growth",
            ),
            "GOOGL": CandidateScore(
                ticker="GOOGL",
                value_score=70.0,
                growth_score=75.0,
                quality_score=80.0,
                momentum_score=65.0,
                sentiment_score=60.0,
                composite_score=70.0,
                reasoning="Stable",
            ),
        }

        def get_fundamentals_mock(ticker):
            return FundamentalsData(
                ticker=ticker,
                pe_ratio=25.0,
                market_cap=2_000_000_000_000,
                sector="Technology",
            )

        def score_stock_mock(fundamentals, **kwargs):
            return scores.get(fundamentals.ticker, mock_score)

        async def async_score_stock_mock(fundamentals, **kwargs):
            return scores.get(fundamentals.ticker, mock_score)

        with patch.object(
            pipeline.universe_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_universe, patch.object(
            pipeline.fundamentals_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_fundamentals, patch.object(
            pipeline.fundamentals_collector,
            "get_cached_fundamentals",
            side_effect=get_fundamentals_mock,
        ), patch.object(
            pipeline.scorer,
            "score_stock",
            new_callable=AsyncMock,
            side_effect=async_score_stock_mock,
        ):
            mock_universe.return_value = CollectorResult(
                collector_name="universe",
                success=True,
                records_collected=3,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_fundamentals.return_value = CollectorResult(
                collector_name="fundamentals",
                success=True,
                records_collected=3,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )

            # Run the full pipeline
            result = await pipeline.run_discovery(dry_run=False)

            # Verify results
            assert result.total_candidates == 3
            assert result.scored_candidates == 3
            assert result.success is True
            assert len(result.errors) == 0
            assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_universe(
        self, db_session, settings, research_config
    ):
        """Test pipeline handles empty universe gracefully."""
        pipeline = DiscoveryPipeline(
            session=db_session,
            config=settings,
            research_config=research_config,
        )

        with patch.object(
            pipeline.universe_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_universe, patch.object(
            pipeline.fundamentals_collector,
            "collect",
            new_callable=AsyncMock,
        ) as mock_fundamentals:
            mock_universe.return_value = CollectorResult(
                collector_name="universe",
                success=True,
                records_collected=0,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )
            mock_fundamentals.return_value = CollectorResult(
                collector_name="fundamentals",
                success=True,
                records_collected=0,
                errors=[],
                started_at=datetime.now(),
                finished_at=datetime.now(),
            )

            result = await pipeline.run_discovery(dry_run=True)

            assert result.total_candidates == 0
            assert result.scored_candidates == 0
            assert result.success is True  # No errors = success
