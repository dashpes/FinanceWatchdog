"""Tests for the research CLI module."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from investment_monitor.research_cli import app
from investment_monitor.research.discovery import DiscoveryResult
from investment_monitor.research.orchestrator import ResearchResult
from investment_monitor.storage import (
    ResearchProfile,
    ResearchReport,
    StockCandidate,
)


runner = CliRunner()


class TestDiscoverCommand:
    """Tests for the discover command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.DiscoveryPipeline")
    def test_discover_dry_run(
        self, mock_pipeline_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test discover command in dry-run mode."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = DiscoveryResult(
            total_candidates=100,
            scored_candidates=50,
            top_candidates=["AAPL", "MSFT"],
            watchlist_additions=[],
            errors=[],
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run_discovery = AsyncMock(return_value=mock_result)
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["discover", "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "dry run" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.DiscoveryPipeline")
    def test_discover_success(
        self, mock_pipeline_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test successful discover command."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = DiscoveryResult(
            total_candidates=100,
            scored_candidates=50,
            top_candidates=["AAPL", "MSFT", "GOOGL"],
            watchlist_additions=["AAPL"],
            errors=[],
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run_discovery = AsyncMock(return_value=mock_result)
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["discover"])

        assert result.exit_code == 0
        assert "100" in result.output  # total candidates
        assert "50" in result.output  # scored candidates

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.DiscoveryPipeline")
    def test_discover_with_errors(
        self, mock_pipeline_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test discover command with pipeline errors."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = DiscoveryResult(
            total_candidates=100,
            scored_candidates=0,
            top_candidates=[],
            watchlist_additions=[],
            errors=["Connection failed", "API error"],
        )
        mock_pipeline = MagicMock()
        mock_pipeline.run_discovery = AsyncMock(return_value=mock_result)
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["discover"])

        # Should still complete but report errors
        assert "error" in result.output.lower()


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchOrchestrator")
    def test_analyze_ticker(
        self, mock_orch_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test analyze command for a single ticker."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_report = MagicMock(spec=ResearchReport)
        mock_report.ticker = "AAPL"
        mock_report.summary = "Strong company with good fundamentals"
        mock_report.recommendation = "BUY"
        mock_report.target_price = 200.0

        mock_result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=mock_report,
            error=None,
            duration=5.0,
        )

        mock_orch = MagicMock()
        mock_orch.research_ticker = AsyncMock(return_value=mock_result)
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(app, ["analyze", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchOrchestrator")
    def test_analyze_ticker_no_report(
        self, mock_orch_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test analyze command with --no-report flag."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = ResearchResult(
            ticker="AAPL",
            success=True,
            report=None,
            error=None,
            duration=5.0,
        )

        mock_orch = MagicMock()
        mock_orch.research_ticker = AsyncMock(return_value=mock_result)
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(app, ["analyze", "AAPL", "--no-report"])

        assert result.exit_code == 0
        assert "AAPL" in result.output

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchOrchestrator")
    def test_analyze_ticker_failure(
        self, mock_orch_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test analyze command when research fails."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = ResearchResult(
            ticker="INVALID",
            success=False,
            report=None,
            error="Ticker not found",
            duration=1.0,
        )

        mock_orch = MagicMock()
        mock_orch.research_ticker = AsyncMock(return_value=mock_result)
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(app, ["analyze", "INVALID"])

        assert result.exit_code == 1
        assert "error" in result.output.lower() or "failed" in result.output.lower()


class TestQueueListCommand:
    """Tests for queue list command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_list_empty(self, mock_queue_cls, mock_settings, mock_session, mock_init_db):
        """Test queue list when empty."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.get_queue.return_value = []
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "list"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_list_with_items(self, mock_queue_cls, mock_settings, mock_session, mock_init_db):
        """Test queue list with items."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        candidate1 = MagicMock(spec=StockCandidate)
        candidate1.ticker = "AAPL"
        candidate1.composite_score = 85.0

        candidate2 = MagicMock(spec=StockCandidate)
        candidate2.ticker = "MSFT"
        candidate2.composite_score = 80.0

        mock_queue = MagicMock()
        mock_queue.get_queue.return_value = [candidate1, candidate2]
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "list"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "MSFT" in result.output


class TestQueueAddCommand:
    """Tests for queue add command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_add_success(self, mock_queue_cls, mock_session, mock_init_db):
        """Test adding ticker to queue."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.add_to_queue.return_value = True
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "add", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        mock_queue.add_to_queue.assert_called_once_with("AAPL", priority=0)

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_add_with_priority(self, mock_queue_cls, mock_session, mock_init_db):
        """Test adding ticker with priority."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.add_to_queue.return_value = True
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "add", "AAPL", "--priority", "100"])

        assert result.exit_code == 0
        mock_queue.add_to_queue.assert_called_once_with("AAPL", priority=100)

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_add_failure(self, mock_queue_cls, mock_session, mock_init_db):
        """Test queue add when it fails."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.add_to_queue.return_value = False
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "add", "INVALID"])

        assert result.exit_code == 1


class TestQueueRemoveCommand:
    """Tests for queue remove command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_remove_success(self, mock_queue_cls, mock_session, mock_init_db):
        """Test removing ticker from queue."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.remove_from_queue.return_value = True
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "remove", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        mock_queue.remove_from_queue.assert_called_once_with("AAPL")

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.ResearchQueue")
    def test_queue_remove_not_found(self, mock_queue_cls, mock_session, mock_init_db):
        """Test removing ticker that's not in queue."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_queue = MagicMock()
        mock_queue.remove_from_queue.return_value = False
        mock_queue_cls.return_value = mock_queue

        result = runner.invoke(app, ["queue", "remove", "INVALID"])

        assert result.exit_code == 1


class TestQueueProcessCommand:
    """Tests for queue process command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchOrchestrator")
    def test_queue_process_default(
        self, mock_orch_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test processing queue with default max."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result1 = ResearchResult(
            ticker="AAPL", success=True, report=MagicMock(), error=None, duration=5.0
        )
        mock_result2 = ResearchResult(
            ticker="MSFT", success=True, report=MagicMock(), error=None, duration=4.0
        )

        mock_orch = MagicMock()
        mock_orch.process_queue = AsyncMock(return_value=[mock_result1, mock_result2])
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(app, ["queue", "process"])

        assert result.exit_code == 0
        mock_orch.process_queue.assert_called_once_with(max_items=5)

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.ResearchOrchestrator")
    def test_queue_process_with_max(
        self, mock_orch_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test processing queue with custom max."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_orch = MagicMock()
        mock_orch.process_queue = AsyncMock(return_value=[])
        mock_orch_cls.return_value = mock_orch

        result = runner.invoke(app, ["queue", "process", "--max", "10"])

        assert result.exit_code == 0
        mock_orch.process_queue.assert_called_once_with(max_items=10)


class TestTopCommand:
    """Tests for top command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_top_candidates")
    def test_top_default(self, mock_get_top, mock_session, mock_init_db):
        """Test top command with defaults."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        candidate1 = MagicMock(spec=StockCandidate)
        candidate1.ticker = "AAPL"
        candidate1.composite_score = 85.0
        candidate1.status = "researched"

        candidate2 = MagicMock(spec=StockCandidate)
        candidate2.ticker = "MSFT"
        candidate2.composite_score = 80.0
        candidate2.status = "watchlist"

        mock_get_top.return_value = [candidate1, candidate2]

        result = runner.invoke(app, ["top"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "MSFT" in result.output
        assert "85" in result.output

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_top_candidates")
    def test_top_with_limit(self, mock_get_top, mock_session, mock_init_db):
        """Test top command with limit."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_top.return_value = []

        result = runner.invoke(app, ["top", "--limit", "5"])

        assert result.exit_code == 0
        mock_get_top.assert_called_once()
        call_kwargs = mock_get_top.call_args[1]
        assert call_kwargs["limit"] == 5

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_top_candidates")
    def test_top_with_min_score(self, mock_get_top, mock_session, mock_init_db):
        """Test top command with min-score."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_top.return_value = []

        result = runner.invoke(app, ["top", "--min-score", "70"])

        assert result.exit_code == 0
        mock_get_top.assert_called_once()
        call_kwargs = mock_get_top.call_args[1]
        assert call_kwargs["min_score"] == 70.0

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_top_candidates")
    def test_top_empty(self, mock_get_top, mock_session, mock_init_db):
        """Test top command with no candidates."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_top.return_value = []

        result = runner.invoke(app, ["top"])

        assert result.exit_code == 0
        assert "no candidate" in result.output.lower()


class TestReportCommand:
    """Tests for report command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_latest_report")
    def test_report_found(self, mock_get_report, mock_session, mock_init_db):
        """Test report command when report exists."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_report = MagicMock(spec=ResearchReport)
        mock_report.ticker = "AAPL"
        mock_report.summary = "Strong company with solid fundamentals"
        mock_report.bull_case = "Growing services revenue"
        mock_report.bear_case = "Hardware saturation risk"
        mock_report.thesis = "Long-term hold"
        mock_report.recommendation = "BUY"
        mock_report.target_price = 200.0
        mock_report.created_at = datetime(2024, 1, 15, 10, 30)

        mock_get_report.return_value = mock_report

        result = runner.invoke(app, ["report", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "Strong company" in result.output
        assert "BUY" in result.output

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_latest_report")
    def test_report_not_found(self, mock_get_report, mock_session, mock_init_db):
        """Test report command when no report exists."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_report.return_value = None

        result = runner.invoke(app, ["report", "UNKNOWN"])

        assert result.exit_code == 1
        assert "no report" in result.output.lower()


class TestProfileCommand:
    """Tests for profile command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_or_create_default_profile")
    def test_profile_show(
        self, mock_get_profile, mock_session, mock_init_db
    ):
        """Test profile show command."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_profile = MagicMock(spec=ResearchProfile)
        mock_profile.name = "default"
        mock_profile.investment_style = "growth"
        mock_profile.risk_tolerance = "moderate"
        mock_profile.value_weight = 0.2
        mock_profile.growth_weight = 0.3
        mock_profile.quality_weight = 0.2
        mock_profile.momentum_weight = 0.15
        mock_profile.sentiment_weight = 0.15

        mock_get_profile.return_value = mock_profile

        result = runner.invoke(app, ["profile", "--show"])

        assert result.exit_code == 0
        assert "default" in result.output
        assert "growth" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_or_create_default_profile")
    def test_profile_no_args(
        self, mock_get_profile, mock_session, mock_init_db
    ):
        """Test profile command without arguments shows profile."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_profile = MagicMock(spec=ResearchProfile)
        mock_profile.name = "default"
        mock_profile.investment_style = "value"
        mock_profile.risk_tolerance = "low"
        mock_profile.value_weight = 0.3
        mock_profile.growth_weight = 0.2
        mock_profile.quality_weight = 0.2
        mock_profile.momentum_weight = 0.15
        mock_profile.sentiment_weight = 0.15

        mock_get_profile.return_value = mock_profile

        result = runner.invoke(app, ["profile"])

        assert result.exit_code == 0


class TestErrorHandling:
    """Tests for CLI error handling."""

    @patch("investment_monitor.research_cli.init_db")
    def test_database_error(self, mock_init_db):
        """Test handling of database initialization error."""
        mock_init_db.side_effect = Exception("Database connection failed")

        result = runner.invoke(app, ["top"])

        assert result.exit_code == 1
        assert "error" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.DiscoveryPipeline")
    def test_pipeline_exception(
        self, mock_pipeline_cls, mock_settings, mock_session, mock_init_db
    ):
        """Test handling of unexpected pipeline exception."""
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_pipeline = MagicMock()
        mock_pipeline.run_discovery = AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["discover"])

        assert result.exit_code == 1
        assert "error" in result.output.lower()
