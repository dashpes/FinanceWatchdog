"""Tests for the main orchestrator module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

import pytest

from investment_monitor.main import (
    RunSummary,
    _build_weekly_data,
    _load_alerts_config,
    _load_portfolio,
    _process_news_ai,
    _run_alert_checks,
    _run_collectors,
    _send_immediate_alerts,
    run_monitor,
)
from investment_monitor.models import AlertsConfig, Portfolio
from investment_monitor.notifications import AlertMessage, Priority


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_run_summary_defaults(self):
        """Test RunSummary with default values."""
        summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
        )
        assert summary.run_type == "regular"
        assert summary.collectors_run == 0
        assert summary.collectors_succeeded == 0
        assert summary.records_collected == 0
        assert summary.alerts_generated == 0
        assert summary.alerts_sent == 0
        assert summary.errors == []
        assert summary.success is True

    def test_run_summary_duration(self):
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 30)
        summary = RunSummary(
            run_type="regular",
            started_at=start,
            finished_at=end,
        )
        assert summary.duration_seconds == 30.0

    def test_run_summary_with_errors(self):
        """Test that errors mark run as not successful."""
        summary = RunSummary(
            run_type="digest",
            started_at=datetime.now(),
            errors=["Something went wrong"],
        )
        assert summary.success is False

    def test_run_summary_str(self):
        """Test string representation."""
        summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
            collectors_run=5,
            collectors_succeeded=4,
            records_collected=100,
            alerts_generated=10,
            alerts_sent=3,
        )
        result = str(summary)
        assert "SUCCESS" in result
        assert "regular" in result
        assert "5" in result  # collectors run
        assert "4" in result  # collectors succeeded
        assert "100" in result  # records
        assert "10" in result  # alerts generated
        assert "3" in result  # alerts sent


class TestLoadPortfolio:
    """Tests for portfolio loading."""

    def test_load_portfolio_missing_file(self, tmp_path):
        """Test loading returns empty portfolio when file doesn't exist."""
        portfolio = _load_portfolio(tmp_path)
        assert isinstance(portfolio, Portfolio)
        assert portfolio.holdings == []
        assert portfolio.watchlist == []

    def test_load_portfolio_valid_file(self, tmp_path):
        """Test loading valid portfolio file."""
        portfolio_file = tmp_path / "portfolio.yaml"
        portfolio_file.write_text("""
holdings:
  - ticker: AAPL
    shares: 10
    cost_basis: 150.00
    thesis: "Tech leader"
watchlist:
  - ticker: MSFT
    reason: "Cloud growth"
""")
        portfolio = _load_portfolio(tmp_path)
        assert len(portfolio.holdings) == 1
        assert portfolio.holdings[0].ticker == "AAPL"
        assert len(portfolio.watchlist) == 1
        assert portfolio.watchlist[0].ticker == "MSFT"


class TestLoadAlertsConfig:
    """Tests for alerts config loading."""

    def test_load_alerts_config_missing_file(self, tmp_path):
        """Test loading returns defaults when file doesn't exist."""
        config = _load_alerts_config(tmp_path)
        assert isinstance(config, AlertsConfig)
        assert config.price.enabled is True  # Default

    def test_load_alerts_config_valid_file(self, tmp_path):
        """Test loading valid alerts config file."""
        alerts_file = tmp_path / "alerts.yaml"
        alerts_file.write_text("""
price:
  enabled: true
  daily_drop_pct: 5.0
volume:
  enabled: false
""")
        config = _load_alerts_config(tmp_path)
        assert config.price.enabled is True
        assert config.price.daily_drop_pct == 5.0
        assert config.volume.enabled is False


class TestRunCollectors:
    """Tests for collector orchestration."""

    @pytest.mark.asyncio
    async def test_run_collectors_empty_tickers(self):
        """Test that empty ticker list returns empty results."""
        session = MagicMock()
        settings = MagicMock()
        portfolio = Portfolio()  # Empty portfolio

        results = await _run_collectors(session, settings, portfolio)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_collectors_with_mocked_collectors(self):
        """Test that collectors are run with proper isolation."""
        session = MagicMock()
        settings = MagicMock()
        portfolio = Portfolio(
            holdings=[
                {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}
            ]
        )

        # Mock the collector classes
        with patch("investment_monitor.main.PriceCollector") as mock_price, \
             patch("investment_monitor.main.InsiderCollector") as mock_insider, \
             patch("investment_monitor.main.NewsCollector") as mock_news, \
             patch("investment_monitor.main.EarningsCollector") as mock_earnings, \
             patch("investment_monitor.main.ETFHoldingsCollector") as mock_etf:

            # Create mock collector instances
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.records_collected = 5

            for mock_collector in [mock_price, mock_insider, mock_news, mock_earnings, mock_etf]:
                instance = MagicMock()
                instance.run = AsyncMock(return_value=mock_result)
                instance.name = "TestCollector"
                mock_collector.return_value = instance

            results = await _run_collectors(session, settings, portfolio)

            assert len(results) == 5
            assert all(r.success for r in results)


class TestProcessNewsAI:
    """Tests for news AI processing."""

    @pytest.mark.asyncio
    async def test_process_news_ai_unavailable(self):
        """Test that unavailable LLM returns 0 processed."""
        session = MagicMock()
        settings = MagicMock()
        settings.ollama_model = "phi3:mini"
        settings.ollama_host = "http://localhost:11434"
        portfolio = Portfolio()

        with patch("investment_monitor.main.LocalLLM") as mock_llm:
            mock_llm.return_value.is_available.return_value = False

            count = await _process_news_ai(session, settings, portfolio)
            assert count == 0


class TestRunAlertChecks:
    """Tests for alert engine orchestration."""

    def test_run_alert_checks(self):
        """Test that alert engine is properly called."""
        session = MagicMock()
        portfolio = Portfolio()
        alerts_config = AlertsConfig()

        with patch("investment_monitor.main.AlertEngine") as mock_engine:
            mock_engine.return_value.run_all_checks.return_value = []

            alerts = _run_alert_checks(session, portfolio, alerts_config)

            mock_engine.assert_called_once_with(session, portfolio, alerts_config)
            mock_engine.return_value.run_all_checks.assert_called_once()
            assert alerts == []


class TestSendImmediateAlerts:
    """Tests for immediate alert sending."""

    @pytest.mark.asyncio
    async def test_send_immediate_alerts_no_high_priority(self):
        """Test that no alerts are sent when none are HIGH priority."""
        alerts = [
            AlertMessage(
                title="Test",
                body="Test body",
                alert_type="price",
                priority=Priority.MEDIUM,
            )
        ]
        session = MagicMock()
        deduplicator = MagicMock()
        notification_manager = MagicMock()

        sent = await _send_immediate_alerts(alerts, session, deduplicator, notification_manager)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_immediate_alerts_high_priority(self):
        """Test that HIGH priority alerts are sent."""
        alerts = [
            AlertMessage(
                title="High Priority Alert",
                body="Test body",
                alert_type="price",
                priority=Priority.HIGH,
            )
        ]
        session = MagicMock()
        deduplicator = MagicMock()
        notification_manager = MagicMock()
        notification_manager.notify = AsyncMock()

        sent = await _send_immediate_alerts(alerts, session, deduplicator, notification_manager)

        assert sent == 1
        notification_manager.notify.assert_called_once()
        deduplicator.mark_sent.assert_called_once()


class TestRunMonitor:
    """Tests for the main run_monitor function."""

    @pytest.mark.asyncio
    async def test_run_monitor_regular(self, tmp_path):
        """Test regular run mode."""
        # Create minimal config files
        portfolio_file = tmp_path / "portfolio.yaml"
        portfolio_file.write_text("holdings: []")

        alerts_file = tmp_path / "alerts.yaml"
        alerts_file.write_text("price:\n  enabled: false")

        with patch("investment_monitor.main.init_db"), \
             patch("investment_monitor.main.get_session") as mock_session, \
             patch("investment_monitor.main._run_collectors", new_callable=AsyncMock) as mock_collectors, \
             patch("investment_monitor.main._process_news_ai", new_callable=AsyncMock) as mock_news, \
             patch("investment_monitor.main._run_alert_checks") as mock_alerts, \
             patch("investment_monitor.main.AlertDeduplicator") as mock_dedup, \
             patch("investment_monitor.main._send_immediate_alerts", new_callable=AsyncMock) as mock_send:

            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_collectors.return_value = []
            mock_news.return_value = 0
            mock_alerts.return_value = []
            mock_dedup.return_value.filter_duplicates.return_value = []
            mock_send.return_value = 0

            summary = await run_monitor(
                config_path=tmp_path,
                run_type="regular",
                log_level="ERROR",  # Suppress output in tests
            )

            assert summary.run_type == "regular"
            assert summary.success is True
            mock_collectors.assert_called_once()
            mock_news.assert_called_once()
            mock_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_monitor_digest(self, tmp_path):
        """Test digest run mode."""
        portfolio_file = tmp_path / "portfolio.yaml"
        portfolio_file.write_text("holdings: []")

        alerts_file = tmp_path / "alerts.yaml"
        alerts_file.write_text("price:\n  enabled: false")

        with patch("investment_monitor.main.init_db"), \
             patch("investment_monitor.main.get_session") as mock_session, \
             patch("investment_monitor.main._run_collectors", new_callable=AsyncMock) as mock_collectors, \
             patch("investment_monitor.main._process_news_ai", new_callable=AsyncMock) as mock_news, \
             patch("investment_monitor.main._run_alert_checks") as mock_alerts, \
             patch("investment_monitor.main.AlertDeduplicator") as mock_dedup, \
             patch("investment_monitor.main._send_immediate_alerts", new_callable=AsyncMock) as mock_send, \
             patch("investment_monitor.main._send_daily_digest", new_callable=AsyncMock) as mock_digest:

            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_collectors.return_value = []
            mock_news.return_value = 0
            mock_alerts.return_value = []
            mock_dedup.return_value.filter_duplicates.return_value = []
            mock_send.return_value = 0

            summary = await run_monitor(
                config_path=tmp_path,
                run_type="digest",
                log_level="ERROR",
            )

            assert summary.run_type == "digest"
            mock_digest.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_monitor_weekly(self, tmp_path):
        """Test weekly run mode."""
        portfolio_file = tmp_path / "portfolio.yaml"
        portfolio_file.write_text("holdings: []")

        alerts_file = tmp_path / "alerts.yaml"
        alerts_file.write_text("price:\n  enabled: false")

        with patch("investment_monitor.main.init_db"), \
             patch("investment_monitor.main.get_session") as mock_session, \
             patch("investment_monitor.main._send_weekly_digest", new_callable=AsyncMock) as mock_weekly:

            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            summary = await run_monitor(
                config_path=tmp_path,
                run_type="weekly",
                log_level="ERROR",
            )

            assert summary.run_type == "weekly"
            mock_weekly.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_monitor_handles_errors(self, tmp_path):
        """Test that errors are captured in summary."""
        portfolio_file = tmp_path / "portfolio.yaml"
        portfolio_file.write_text("holdings: []")

        alerts_file = tmp_path / "alerts.yaml"
        alerts_file.write_text("price:\n  enabled: false")

        with patch("investment_monitor.main.init_db") as mock_init:
            mock_init.side_effect = Exception("Database error")

            summary = await run_monitor(
                config_path=tmp_path,
                run_type="regular",
                log_level="ERROR",
            )

            assert summary.success is False
            assert len(summary.errors) > 0
            assert "Database error" in summary.errors[0]


class TestBuildWeeklyData:
    """Tests for weekly data building."""

    def test_build_weekly_data_empty(self):
        """Test building weekly data with no alerts/news."""
        session = MagicMock()
        portfolio = Portfolio()

        with patch("investment_monitor.main.get_recent_news") as mock_news, \
             patch("investment_monitor.main.get_upcoming_earnings") as mock_earnings, \
             patch("investment_monitor.main.get_recent_alerts") as mock_alerts:

            mock_news.return_value = []
            mock_earnings.return_value = []
            mock_alerts.return_value = []

            from datetime import date, timedelta
            week_end = date.today()
            week_start = week_end - timedelta(days=6)

            data = _build_weekly_data(session, portfolio, week_start, week_end)

            assert data.week_start == week_start
            assert data.week_end == week_end
            assert "No significant" in data.news_summary or "No" in data.news_summary
