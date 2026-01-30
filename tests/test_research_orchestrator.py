"""Tests for ResearchOrchestrator - coordinates full research flow."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.storage import (
    CandidateScore,
    ResearchReport,
    StockCandidate,
    get_candidate_by_ticker,
    get_session,
    init_db,
    save_candidate,
    save_score,
)
from investment_monitor.research import ResearchOrchestrator, ResearchQueue, ResearchResult


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.anthropic_api_key = "test-api-key"
    return settings


@pytest.fixture
def mock_research_config():
    """Create mock research config."""
    config = MagicMock()
    config.claude_budget = MagicMock()
    config.claude_budget.enabled = True
    config.claude_budget.monthly_limit_usd = 50.0
    return config


class TestResearchResultDataclass:
    """Tests for ResearchResult dataclass."""

    def test_research_result_has_required_fields(self):
        """Test ResearchResult has all required fields."""
        result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=None,
            error=None,
            duration=1.5,
        )

        assert result.ticker == "AAPL"
        assert result.success is True
        assert result.report is None
        assert result.error is None
        assert result.duration == 1.5

    def test_research_result_with_report(self, db_session):
        """Test ResearchResult can hold a ResearchReport."""
        report = ResearchReport(
            ticker="AAPL",
            summary="Test summary",
            recommendation="buy",
        )

        result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=report,
            error=None,
            duration=2.0,
        )

        assert result.report is not None
        assert result.report.ticker == "AAPL"
        assert result.report.summary == "Test summary"

    def test_research_result_with_error(self):
        """Test ResearchResult can hold an error message."""
        result = ResearchResult(
            ticker="AAPL",
            success=False,
            report=None,
            error="API error: connection failed",
            duration=0.5,
        )

        assert result.success is False
        assert result.error == "API error: connection failed"


class TestResearchOrchestrator:
    """Tests for ResearchOrchestrator class initialization."""

    def test_orchestrator_initialization(self, db_session, mock_settings, mock_research_config):
        """Test orchestrator initializes with required dependencies."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        assert orchestrator.session == db_session
        assert orchestrator.config == mock_settings
        assert orchestrator.research_config == mock_research_config


class TestResearchTicker:
    """Tests for research_ticker method."""

    @pytest.mark.asyncio
    async def test_research_ticker_creates_candidate_if_not_exists(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test research_ticker creates candidate if it doesn't exist."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Mock the collectors and report generator
        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_report:

            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_report.return_value = MagicMock(
                success=True,
                report=ResearchReport(ticker="AAPL", summary="Test", recommendation="buy"),
            )

            result = await orchestrator.research_ticker("AAPL")

            assert result.ticker == "AAPL"
            # Verify candidate was created
            candidate = get_candidate_by_ticker(db_session, "AAPL")
            assert candidate is not None

    @pytest.mark.asyncio
    async def test_research_ticker_full_flow_with_mocks(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test research_ticker executes full flow: fundamentals, news, congress, report."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Pre-create candidate
        candidate = StockCandidate(
            ticker="MSFT",
            status="screening",
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        # Create a mock score for the candidate
        score = CandidateScore(
            ticker="MSFT",
            value_score=70.0,
            growth_score=80.0,
            quality_score=75.0,
            momentum_score=65.0,
            sentiment_score=70.0,
            composite_score=72.0,
        )
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_fundamentals.sector = "Technology"
        mock_fundamentals.industry = "Software"

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="MSFT",
            summary="Microsoft looks strong",
            recommendation="buy",
        )

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_gen:

            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            result = await orchestrator.research_ticker("MSFT")

            assert result.success is True
            assert result.ticker == "MSFT"
            assert result.report is not None
            mock_fund.assert_called_once_with("MSFT")
            mock_news.assert_called_once_with("MSFT")
            mock_congress.assert_called_once_with("MSFT")

    @pytest.mark.asyncio
    async def test_research_ticker_updates_status_to_researched(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test research_ticker updates candidate status to 'researched' after success."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Pre-create candidate with screening status
        candidate = StockCandidate(
            ticker="GOOGL",
            status="screening",
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        # Create a mock score
        score = CandidateScore(
            ticker="GOOGL",
            composite_score=75.0,
        )
        save_score(db_session, score)
        db_session.commit()

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="GOOGL",
            summary="Google analysis",
            recommendation="hold",
        )

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_gen:

            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            result = await orchestrator.research_ticker("GOOGL")

            assert result.success is True
            # Refresh candidate from database
            db_session.expire_all()
            updated_candidate = get_candidate_by_ticker(db_session, "GOOGL")
            assert updated_candidate.status == "researched"

    @pytest.mark.asyncio
    async def test_research_ticker_handles_error_gracefully(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test research_ticker handles errors and returns failure result."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund:
            mock_fund.side_effect = Exception("API connection failed")

            result = await orchestrator.research_ticker("FAIL")

            assert result.success is False
            assert result.ticker == "FAIL"
            assert "API connection failed" in result.error

    @pytest.mark.asyncio
    async def test_research_ticker_tracks_duration(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test research_ticker tracks duration of the research process."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(ticker="AAPL", summary="Test", recommendation="buy")

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_gen:

            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            result = await orchestrator.research_ticker("AAPL")

            # Duration should be a positive number
            assert result.duration >= 0


class TestProcessQueue:
    """Tests for process_queue method."""

    @pytest.mark.asyncio
    async def test_process_queue_processes_items_in_priority_order(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue processes items in priority order (highest first)."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Add items to queue with different priorities
        queue = ResearchQueue(db_session)
        queue.add_to_queue("LOW", priority=30)
        queue.add_to_queue("HIGH", priority=90)
        queue.add_to_queue("MED", priority=60)
        db_session.commit()

        # Create scores for all candidates
        for ticker, score_val in [("LOW", 30.0), ("HIGH", 90.0), ("MED", 60.0)]:
            score = CandidateScore(ticker=ticker, composite_score=score_val)
            save_score(db_session, score)
        db_session.commit()

        processed_order = []

        async def mock_research(ticker):
            processed_order.append(ticker)
            return ResearchResult(
                ticker=ticker,
                success=True,
                report=ResearchReport(ticker=ticker, summary="Test", recommendation="hold"),
                error=None,
                duration=1.0,
            )

        with patch.object(orchestrator, 'research_ticker', side_effect=mock_research):
            results = await orchestrator.process_queue(max_items=3)

            assert len(results) == 3
            # Should be processed in priority order: HIGH, MED, LOW
            assert processed_order == ["HIGH", "MED", "LOW"]

    @pytest.mark.asyncio
    async def test_process_queue_respects_max_items(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue respects max_items limit."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Add many items to queue
        queue = ResearchQueue(db_session)
        for i in range(10):
            queue.add_to_queue(f"T{i}", priority=i * 10)
        db_session.commit()

        async def mock_research(ticker):
            return ResearchResult(
                ticker=ticker,
                success=True,
                report=None,
                error=None,
                duration=1.0,
            )

        with patch.object(orchestrator, 'research_ticker', side_effect=mock_research):
            results = await orchestrator.process_queue(max_items=3)

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_process_queue_removes_items_after_processing(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue calls remove_from_queue after processing each item."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        queue = ResearchQueue(db_session)
        queue.add_to_queue("AAPL", priority=80)
        queue.add_to_queue("MSFT", priority=70)
        db_session.commit()

        async def mock_research(ticker):
            return ResearchResult(
                ticker=ticker,
                success=True,
                report=ResearchReport(ticker=ticker, summary="Test", recommendation="hold"),
                error=None,
                duration=1.0,
            )

        with patch.object(orchestrator, 'research_ticker', side_effect=mock_research), \
             patch.object(orchestrator._queue, 'remove_from_queue') as mock_remove:

            await orchestrator.process_queue(max_items=5)

            # Verify remove_from_queue was called for each processed item
            assert mock_remove.call_count == 2
            mock_remove.assert_any_call("AAPL")
            mock_remove.assert_any_call("MSFT")

    @pytest.mark.asyncio
    async def test_process_queue_removes_items_even_on_failure(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue calls remove_from_queue even when research fails."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        queue = ResearchQueue(db_session)
        queue.add_to_queue("FAIL", priority=80)
        db_session.commit()

        async def mock_research(ticker):
            return ResearchResult(
                ticker=ticker,
                success=False,
                report=None,
                error="Simulated failure",
                duration=0.5,
            )

        with patch.object(orchestrator, 'research_ticker', side_effect=mock_research), \
             patch.object(orchestrator._queue, 'remove_from_queue') as mock_remove:

            results = await orchestrator.process_queue(max_items=5)

            # Verify remove_from_queue was called even though research failed
            assert len(results) == 1
            assert results[0].success is False
            mock_remove.assert_called_once_with("FAIL")

    @pytest.mark.asyncio
    async def test_process_queue_continues_on_error(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue continues with other items if one fails."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        queue = ResearchQueue(db_session)
        queue.add_to_queue("GOOD1", priority=90)
        queue.add_to_queue("FAIL", priority=80)
        queue.add_to_queue("GOOD2", priority=70)
        db_session.commit()

        call_count = 0

        async def mock_research(ticker):
            nonlocal call_count
            call_count += 1
            if ticker == "FAIL":
                return ResearchResult(
                    ticker=ticker,
                    success=False,
                    report=None,
                    error="Simulated failure",
                    duration=0.5,
                )
            return ResearchResult(
                ticker=ticker,
                success=True,
                report=ResearchReport(ticker=ticker, summary="Test", recommendation="hold"),
                error=None,
                duration=1.0,
            )

        with patch.object(orchestrator, 'research_ticker', side_effect=mock_research):
            results = await orchestrator.process_queue(max_items=3)

            # Should have processed all 3 items
            assert call_count == 3
            assert len(results) == 3

            # Check results
            success_count = sum(1 for r in results if r.success)
            fail_count = sum(1 for r in results if not r.success)
            assert success_count == 2
            assert fail_count == 1

    @pytest.mark.asyncio
    async def test_process_queue_empty_returns_empty_list(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test process_queue returns empty list when queue is empty."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        results = await orchestrator.process_queue(max_items=5)

        assert results == []


class TestBudgetCheck:
    """Tests for Claude budget checking."""

    @pytest.mark.asyncio
    async def test_budget_check_prevents_overspending(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that budget check prevents report generation when budget exceeded."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Pre-create candidate with score
        candidate = StockCandidate(ticker="AAPL", status="screening", discovery_source="test")
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="AAPL", composite_score=80.0)
        save_score(db_session, score)
        db_session.commit()

        # Mock fundamentals to succeed
        mock_fundamentals = MagicMock()

        # Mock report generator to indicate budget exceeded
        mock_report_result = MagicMock()
        mock_report_result.success = False
        mock_report_result.error_message = "Budget limit reached"
        mock_report_result.report = None

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_gen, \
             patch.object(orchestrator, '_check_budget') as mock_budget:

            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_budget.return_value = False  # Budget exceeded

            result = await orchestrator.research_ticker("AAPL")

            assert result.success is False
            assert "budget" in result.error.lower()

    @pytest.mark.asyncio
    async def test_budget_check_allows_when_within_budget(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that research proceeds when within budget."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Pre-create candidate with score
        candidate = StockCandidate(ticker="AAPL", status="screening", discovery_source="test")
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="AAPL", composite_score=80.0)
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(ticker="AAPL", summary="Test", recommendation="buy")

        with patch.object(orchestrator, '_fetch_fundamentals', new_callable=AsyncMock) as mock_fund, \
             patch.object(orchestrator, '_fetch_news', new_callable=AsyncMock) as mock_news, \
             patch.object(orchestrator, '_fetch_congress_trades', new_callable=AsyncMock) as mock_congress, \
             patch.object(orchestrator, '_generate_report', new_callable=AsyncMock) as mock_gen, \
             patch.object(orchestrator, '_check_budget') as mock_budget:

            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_budget.return_value = True  # Within budget

            result = await orchestrator.research_ticker("AAPL")

            assert result.success is True
            mock_gen.assert_called_once()
