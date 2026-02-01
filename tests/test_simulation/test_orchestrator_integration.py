"""Tests for ResearchOrchestrator simulation integration.

These tests verify that the ResearchOrchestrator correctly integrates with
the MonteCarloAnalyzer to automatically run simulations for high-scoring
candidates.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.research import ResearchOrchestrator, ResearchResult
from investment_monitor.simulation.models import (
    HorizonResult,
    SensitivityResult,
    SimulationOutput,
)
from investment_monitor.storage import (
    CandidateScore,
    ResearchReport,
    StockCandidate,
    get_session,
    init_db,
    save_candidate,
    save_score,
)


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


@pytest.fixture
def mock_simulation_output():
    """Create a mock SimulationOutput for testing."""
    # Create minimal HorizonResult
    horizon_result = HorizonResult(
        days=30,
        base_mean=105.0,
        base_median=104.5,
        base_std=15.0,
        base_skewness=0.3,
        base_percentiles={5: 85.0, 25: 95.0, 50: 104.5, 75: 115.0, 95: 130.0},
        base_ci_80=(90.0, 120.0),
        base_ci_95=(80.0, 130.0),
        base_var_95=-0.15,
        base_cvar_95=-0.20,
        scenarios={},
    )

    sensitivity = SensitivityResult(
        volatility_impact=65.0,
        drift_impact=25.0,
        lookback_impact=10.0,
        primary_driver="volatility",
        volatility_range={0.5: 102.0, 1.0: 105.0, 1.5: 108.0},
        drift_range={"pessimistic": 95.0, "neutral": 105.0, "optimistic": 115.0},
        lookback_range={30: 105.0, 60: 104.5, 90: 104.0},
    )

    return SimulationOutput(
        ticker="AAPL",
        entry_price=100.0,
        composite_score=85.0,
        num_simulations=1000,
        lookback_days=252,
        volatility=0.25,
        drift=0.08,
        results={30: horizon_result},
        sensitivity=sensitivity,
    )


class TestSimulationScoreThreshold:
    """Tests for SIMULATION_SCORE_THRESHOLD constant."""

    def test_simulation_score_threshold_exists(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that SIMULATION_SCORE_THRESHOLD constant exists."""
        from investment_monitor.research.orchestrator import SIMULATION_SCORE_THRESHOLD

        assert SIMULATION_SCORE_THRESHOLD == 80.0

    def test_orchestrator_uses_threshold_constant(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that orchestrator class can access the threshold."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )
        # Should be able to reference the threshold
        from investment_monitor.research.orchestrator import SIMULATION_SCORE_THRESHOLD

        assert SIMULATION_SCORE_THRESHOLD is not None


class TestShouldRunSimulation:
    """Tests for _should_run_simulation helper method."""

    def test_should_run_simulation_high_score_enabled(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that high score (>=80) with run_simulation=True returns True."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Score >= 80 and run_simulation=True should return True
        assert orchestrator._should_run_simulation(85.0, True) is True
        assert orchestrator._should_run_simulation(80.0, True) is True
        assert orchestrator._should_run_simulation(100.0, True) is True

    def test_should_run_simulation_low_score(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that low score (<80) returns False regardless of flag."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Score < 80 should return False
        assert orchestrator._should_run_simulation(79.9, True) is False
        assert orchestrator._should_run_simulation(50.0, True) is False
        assert orchestrator._should_run_simulation(0.0, True) is False

    def test_should_run_simulation_flag_false(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that run_simulation=False returns False regardless of score."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # run_simulation=False should return False even with high score
        assert orchestrator._should_run_simulation(85.0, False) is False
        assert orchestrator._should_run_simulation(100.0, False) is False


class TestRunSimulation:
    """Tests for _run_simulation helper method."""

    @pytest.mark.asyncio
    async def test_run_simulation_calls_analyzer(
        self, db_session, mock_settings, mock_research_config, mock_simulation_output
    ):
        """Test that _run_simulation creates MonteCarloAnalyzer and calls analyze."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        with patch(
            "investment_monitor.research.orchestrator.MonteCarloAnalyzer"
        ) as MockAnalyzer:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = mock_simulation_output
            MockAnalyzer.return_value = mock_analyzer

            result = await orchestrator._run_simulation("AAPL", 100.0, 85.0)

            MockAnalyzer.assert_called_once()
            mock_analyzer.analyze.assert_called_once_with(
                ticker="AAPL",
                entry_price=100.0,
                composite_score=85.0,
                force=True,  # Should use force=True to bypass internal threshold
            )
            assert result == mock_simulation_output

    @pytest.mark.asyncio
    async def test_run_simulation_returns_none_on_error(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that _run_simulation returns None if analyzer fails."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        with patch(
            "investment_monitor.research.orchestrator.MonteCarloAnalyzer"
        ) as MockAnalyzer:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.side_effect = Exception("Simulation failed")
            MockAnalyzer.return_value = mock_analyzer

            result = await orchestrator._run_simulation("AAPL", 100.0, 85.0)

            assert result is None


class TestResearchResultWithSimulation:
    """Tests for ResearchResult with simulation_output field."""

    def test_research_result_has_simulation_output_field(self):
        """Test that ResearchResult dataclass has simulation_output field."""
        result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=None,
            error=None,
            duration=1.5,
            simulation_output=None,
        )

        assert hasattr(result, "simulation_output")
        assert result.simulation_output is None

    def test_research_result_with_simulation_output(self, mock_simulation_output):
        """Test ResearchResult can hold SimulationOutput."""
        result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=None,
            error=None,
            duration=1.5,
            simulation_output=mock_simulation_output,
        )

        assert result.simulation_output is not None
        assert result.simulation_output.ticker == "AAPL"
        assert result.simulation_output.composite_score == 85.0


class TestResearchTickerWithSimulation:
    """Tests for research_ticker with run_simulation parameter."""

    @pytest.mark.asyncio
    async def test_research_ticker_accepts_run_simulation_param(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that research_ticker accepts run_simulation parameter."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="AAPL", summary="Test", recommendation="buy"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
        ):
            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            # Should accept run_simulation parameter without error
            result = await orchestrator.research_ticker("AAPL", run_simulation=False)
            assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_research_ticker_triggers_simulation_for_high_score(
        self, db_session, mock_settings, mock_research_config, mock_simulation_output
    ):
        """Test that high-scoring candidate (>=80) triggers simulation when enabled."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with high score
        candidate = StockCandidate(
            ticker="AAPL", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="AAPL", composite_score=85.0)
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_fundamentals.current_price = 100.0

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="AAPL", summary="Test", recommendation="buy"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_sim.return_value = mock_simulation_output

            result = await orchestrator.research_ticker("AAPL", run_simulation=True)

            # Simulation should have been called
            mock_sim.assert_called_once()
            assert result.simulation_output is not None

    @pytest.mark.asyncio
    async def test_research_ticker_skips_simulation_for_low_score(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that low-scoring candidate (<80) skips simulation."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with low score
        candidate = StockCandidate(
            ticker="MSFT", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="MSFT", composite_score=70.0)
        save_score(db_session, score)
        db_session.commit()

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="MSFT", summary="Test", recommendation="hold"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            result = await orchestrator.research_ticker("MSFT", run_simulation=True)

            # Simulation should NOT have been called due to low score
            mock_sim.assert_not_called()
            assert result.simulation_output is None

    @pytest.mark.asyncio
    async def test_research_ticker_skips_simulation_when_disabled(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that simulation is skipped when run_simulation=False."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with high score
        candidate = StockCandidate(
            ticker="GOOGL", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="GOOGL", composite_score=90.0)
        save_score(db_session, score)
        db_session.commit()

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="GOOGL", summary="Test", recommendation="buy"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            # run_simulation=False (default)
            result = await orchestrator.research_ticker("GOOGL", run_simulation=False)

            # Simulation should NOT be called even with high score
            mock_sim.assert_not_called()
            assert result.simulation_output is None


class TestSimulationResultPersistence:
    """Tests for saving simulation results to database."""

    @pytest.mark.asyncio
    async def test_simulation_result_saved_to_database(
        self, db_session, mock_settings, mock_research_config, mock_simulation_output
    ):
        """Test that simulation result is saved using save_simulation_result."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with high score
        candidate = StockCandidate(
            ticker="NVDA", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="NVDA", composite_score=88.0)
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_fundamentals.current_price = 500.0

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="NVDA", summary="Test", recommendation="buy"
        )

        # Update mock to have NVDA ticker
        mock_sim_output = SimulationOutput(
            ticker="NVDA",
            entry_price=500.0,
            composite_score=88.0,
            num_simulations=1000,
            lookback_days=252,
            volatility=0.35,
            drift=0.12,
            results=mock_simulation_output.results,
            sensitivity=mock_simulation_output.sensitivity,
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
            patch(
                "investment_monitor.research.orchestrator.save_simulation_result"
            ) as mock_save,
        ):
            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_sim.return_value = mock_sim_output

            result = await orchestrator.research_ticker("NVDA", run_simulation=True)

            # save_simulation_result should have been called
            mock_save.assert_called_once_with(db_session, mock_sim_output)
            assert result.simulation_output is not None


class TestEdgeCases:
    """Tests for edge cases in simulation integration."""

    @pytest.mark.asyncio
    async def test_simulation_continues_after_failure(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that research continues even if simulation fails."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with high score
        candidate = StockCandidate(
            ticker="TSLA", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="TSLA", composite_score=82.0)
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_fundamentals.current_price = 200.0

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="TSLA", summary="Test", recommendation="buy"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_sim.return_value = None  # Simulation failed

            result = await orchestrator.research_ticker("TSLA", run_simulation=True)

            # Research should still succeed even if simulation fails
            assert result.success is True
            assert result.simulation_output is None

    @pytest.mark.asyncio
    async def test_boundary_score_exactly_80(
        self, db_session, mock_settings, mock_research_config, mock_simulation_output
    ):
        """Test that score of exactly 80 triggers simulation."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with exactly 80 score
        candidate = StockCandidate(
            ticker="AMD", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        score = CandidateScore(ticker="AMD", composite_score=80.0)
        save_score(db_session, score)
        db_session.commit()

        mock_fundamentals = MagicMock()
        mock_fundamentals.current_price = 120.0

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="AMD", summary="Test", recommendation="buy"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = mock_fundamentals
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result
            mock_sim.return_value = mock_simulation_output

            result = await orchestrator.research_ticker("AMD", run_simulation=True)

            # Score exactly 80 should trigger simulation
            mock_sim.assert_called_once()
            assert result.simulation_output is not None

    @pytest.mark.asyncio
    async def test_default_score_does_not_trigger_simulation(
        self, db_session, mock_settings, mock_research_config
    ):
        """Test that default score (50) doesn't trigger simulation."""
        orchestrator = ResearchOrchestrator(
            session=db_session,
            config=mock_settings,
            research_config=mock_research_config,
        )

        # Create candidate with NO score (will use default 50)
        candidate = StockCandidate(
            ticker="INTC", status="screening", discovery_source="test"
        )
        save_candidate(db_session, candidate)
        # Don't save any score - orchestrator will use DEFAULT_COMPOSITE_SCORE (50)
        db_session.commit()

        mock_report_result = MagicMock()
        mock_report_result.success = True
        mock_report_result.report = ResearchReport(
            ticker="INTC", summary="Test", recommendation="hold"
        )

        with (
            patch.object(
                orchestrator, "_fetch_fundamentals", new_callable=AsyncMock
            ) as mock_fund,
            patch.object(
                orchestrator, "_fetch_news", new_callable=AsyncMock
            ) as mock_news,
            patch.object(
                orchestrator, "_fetch_congress_trades", new_callable=AsyncMock
            ) as mock_congress,
            patch.object(
                orchestrator, "_generate_report", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                orchestrator, "_run_simulation", new_callable=AsyncMock
            ) as mock_sim,
        ):
            mock_fund.return_value = MagicMock()
            mock_news.return_value = []
            mock_congress.return_value = []
            mock_gen.return_value = mock_report_result

            result = await orchestrator.research_ticker("INTC", run_simulation=True)

            # Default score 50 should NOT trigger simulation
            mock_sim.assert_not_called()
            assert result.simulation_output is None
