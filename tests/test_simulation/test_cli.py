"""Tests for simulation CLI commands."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from investment_monitor.research_cli import app
from investment_monitor.simulation.models import (
    HorizonResult,
    ScenarioResult,
    SensitivityResult,
    SimulationOutput,
)
from investment_monitor.storage import StockCandidate
from investment_monitor.storage.research_models import SimulationResult


runner = CliRunner()


def _create_mock_simulation_output(ticker: str, entry_price: float = 150.0) -> SimulationOutput:
    """Create a mock SimulationOutput for testing."""
    # Create a minimal scenario result
    scenario = ScenarioResult(
        name="2008 Financial Crisis",
        mean=130.0,
        median=128.0,
        std=25.0,
        ci_80=(110.0, 150.0),
        ci_95=(100.0, 160.0),
        var_95=-0.15,
        cvar_95=-0.20,
        prob_loss_20pct=0.25,
    )

    # Create horizon result for 30 days
    horizon_30 = HorizonResult(
        days=30,
        base_mean=155.0,
        base_median=154.0,
        base_std=12.0,
        base_skewness=0.1,
        base_percentiles={5: 140.0, 25: 148.0, 50: 154.0, 75: 162.0, 95: 175.0},
        base_ci_80=(145.0, 165.0),
        base_ci_95=(138.0, 172.0),
        base_var_95=-0.08,
        base_cvar_95=-0.12,
        scenarios={"2008 Financial Crisis": scenario},
    )

    # Create horizon result for 90 days
    horizon_90 = HorizonResult(
        days=90,
        base_mean=165.0,
        base_median=163.0,
        base_std=18.0,
        base_skewness=0.15,
        base_percentiles={5: 138.0, 25: 150.0, 50: 163.0, 75: 178.0, 95: 195.0},
        base_ci_80=(148.0, 182.0),
        base_ci_95=(135.0, 195.0),
        base_var_95=-0.10,
        base_cvar_95=-0.15,
        scenarios={"2008 Financial Crisis": scenario},
    )

    # Create horizon result for 252 days
    horizon_252 = HorizonResult(
        days=252,
        base_mean=180.0,
        base_median=178.0,
        base_std=28.0,
        base_skewness=0.2,
        base_percentiles={5: 130.0, 25: 155.0, 50: 178.0, 75: 202.0, 95: 230.0},
        base_ci_80=(150.0, 210.0),
        base_ci_95=(128.0, 235.0),
        base_var_95=-0.13,
        base_cvar_95=-0.18,
        scenarios={"2008 Financial Crisis": scenario},
    )

    # Create sensitivity result
    sensitivity = SensitivityResult(
        volatility_impact=45.0,
        drift_impact=30.0,
        lookback_impact=25.0,
        primary_driver="volatility",
        volatility_range={0.5: 148.0, 1.0: 155.0, 1.5: 162.0},
        drift_range={"pessimistic": 145.0, "neutral": 155.0, "optimistic": 165.0},
        lookback_range={30: 154.0, 90: 155.0, 252: 156.0},
    )

    return SimulationOutput(
        ticker=ticker,
        entry_price=entry_price,
        composite_score=85.0,
        num_simulations=1000,
        lookback_days=756,
        volatility=0.25,
        drift=0.08,
        results={30: horizon_30, 90: horizon_90, 252: horizon_252},
        sensitivity=sensitivity,
    )


class TestSimulateCommand:
    """Tests for the simulate command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    @patch("investment_monitor.research_cli.save_simulation_result")
    def test_simulate_single_ticker(
        self,
        mock_save_result,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command with a single ticker."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance for current price
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        # Mock analyzer
        mock_output = _create_mock_simulation_output("AAPL", 150.0)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_output
        mock_analyzer_cls.return_value = mock_analyzer

        # Mock save result
        mock_save_result.return_value = MagicMock(id=1)

        result = runner.invoke(app, ["simulate", "--ticker", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        mock_analyzer.analyze.assert_called_once()
        mock_save_result.assert_called_once()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    @patch("investment_monitor.research_cli.save_simulation_result")
    def test_simulate_multiple_tickers(
        self,
        mock_save_result,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command with multiple tickers."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance for current prices
        def mock_ticker_info(ticker):
            prices = {"AAPL": 150.0, "MSFT": 400.0, "GOOGL": 175.0}
            mock_t = MagicMock()
            mock_t.info = {"currentPrice": prices.get(ticker, 100.0)}
            return mock_t

        mock_yf.Ticker.side_effect = mock_ticker_info

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = [
            _create_mock_simulation_output("AAPL", 150.0),
            _create_mock_simulation_output("MSFT", 400.0),
            _create_mock_simulation_output("GOOGL", 175.0),
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        # Mock save result
        mock_save_result.return_value = MagicMock(id=1)

        result = runner.invoke(app, ["simulate", "--tickers", "AAPL,MSFT,GOOGL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "MSFT" in result.output
        assert "GOOGL" in result.output
        assert mock_analyzer.analyze.call_count == 3

    def test_simulate_requires_ticker_or_auto(self):
        """Test that simulate command requires either ticker, tickers, or --auto."""
        result = runner.invoke(app, ["simulate"])

        assert result.exit_code != 0
        # Should indicate that a ticker or --auto is required
        assert "ticker" in result.output.lower() or "auto" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    @patch("investment_monitor.research_cli.save_simulation_result")
    @patch("investment_monitor.research_cli.get_high_scoring_candidates")
    def test_simulate_auto_mode(
        self,
        mock_get_candidates,
        mock_save_result,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command with --auto flag."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock high-scoring candidates
        candidate1 = MagicMock(spec=StockCandidate)
        candidate1.ticker = "AAPL"
        candidate1.composite_score = 85.0

        candidate2 = MagicMock(spec=StockCandidate)
        candidate2.ticker = "MSFT"
        candidate2.composite_score = 82.0

        mock_get_candidates.return_value = [candidate1, candidate2]

        # Mock yfinance
        def mock_ticker_info(ticker):
            prices = {"AAPL": 150.0, "MSFT": 400.0}
            mock_t = MagicMock()
            mock_t.info = {"currentPrice": prices.get(ticker, 100.0)}
            return mock_t

        mock_yf.Ticker.side_effect = mock_ticker_info

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = [
            _create_mock_simulation_output("AAPL", 150.0),
            _create_mock_simulation_output("MSFT", 400.0),
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        # Mock save result
        mock_save_result.return_value = MagicMock(id=1)

        result = runner.invoke(app, ["simulate", "--auto"])

        assert result.exit_code == 0
        assert mock_analyzer.analyze.call_count == 2
        mock_get_candidates.assert_called_once()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    @patch("investment_monitor.research_cli.save_simulation_result")
    def test_simulate_with_custom_horizons(
        self,
        mock_save_result,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command with custom horizons."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        # Mock analyzer
        mock_output = _create_mock_simulation_output("AAPL", 150.0)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_output
        mock_analyzer_cls.return_value = mock_analyzer

        # Mock save result
        mock_save_result.return_value = MagicMock(id=1)

        result = runner.invoke(
            app, ["simulate", "--ticker", "AAPL", "--horizons", "30,60,120"]
        )

        assert result.exit_code == 0
        # Verify the config was set with custom horizons
        call_args = mock_analyzer_cls.call_args
        if call_args and call_args.kwargs.get("config"):
            config = call_args.kwargs["config"]
            assert config.horizons == [30, 60, 120]

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    @patch("investment_monitor.research_cli.save_simulation_result")
    def test_simulate_with_force(
        self,
        mock_save_result,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command with --force flag."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        # Mock analyzer
        mock_output = _create_mock_simulation_output("AAPL", 150.0)
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = mock_output
        mock_analyzer_cls.return_value = mock_analyzer

        # Mock save result
        mock_save_result.return_value = MagicMock(id=1)

        result = runner.invoke(app, ["simulate", "--ticker", "AAPL", "--force"])

        assert result.exit_code == 0
        # Check that force=True was passed to analyze
        mock_analyzer.analyze.assert_called_once()
        call_kwargs = mock_analyzer.analyze.call_args.kwargs
        assert call_kwargs.get("force") is True


class TestSimulationResultsCommand:
    """Tests for the simulation-results command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_by_ticker(
        self, mock_get_results, mock_settings, mock_session, mock_init_db
    ):
        """Test simulation-results command filtered by ticker."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Create mock simulation results
        mock_result = MagicMock(spec=SimulationResult)
        mock_result.ticker = "AAPL"
        mock_result.run_date = date(2026, 1, 31)
        mock_result.entry_price = 150.0
        mock_result.composite_score = 85.0
        mock_result.num_simulations = 1000
        mock_result.volatility = 0.25
        mock_result.drift = 0.08
        mock_result.results_30d = {"base_mean": 155.0, "base_var_95": -0.08}
        mock_result.results_90d = {"base_mean": 165.0, "base_var_95": -0.10}
        mock_result.results_252d = {"base_mean": 180.0, "base_var_95": -0.13}
        mock_result.created_at = datetime(2026, 1, 31, 10, 30)

        mock_get_results.return_value = [mock_result]

        result = runner.invoke(app, ["simulation-results", "--ticker", "AAPL"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        mock_get_results.assert_called_once()
        call_kwargs = mock_get_results.call_args.kwargs
        assert call_kwargs.get("ticker") == "AAPL"

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_latest(
        self, mock_get_results, mock_settings, mock_session, mock_init_db
    ):
        """Test simulation-results command with --latest flag."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Create mock simulation results
        results = []
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL"]):
            mock_result = MagicMock(spec=SimulationResult)
            mock_result.ticker = ticker
            mock_result.run_date = date(2026, 1, 31 - i)
            mock_result.entry_price = 150.0 + i * 50
            mock_result.composite_score = 85.0 - i
            mock_result.num_simulations = 1000
            mock_result.volatility = 0.25
            mock_result.drift = 0.08
            mock_result.results_30d = {"base_mean": 155.0, "base_var_95": -0.08}
            mock_result.results_90d = {"base_mean": 165.0, "base_var_95": -0.10}
            mock_result.results_252d = {"base_mean": 180.0, "base_var_95": -0.13}
            mock_result.created_at = datetime(2026, 1, 31 - i, 10, 30)
            results.append(mock_result)

        mock_get_results.return_value = results

        result = runner.invoke(app, ["simulation-results", "--latest", "3"])

        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "MSFT" in result.output
        assert "GOOGL" in result.output
        mock_get_results.assert_called_once()
        call_kwargs = mock_get_results.call_args.kwargs
        assert call_kwargs.get("limit") == 3

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_empty(
        self, mock_get_results, mock_settings, mock_session, mock_init_db
    ):
        """Test simulation-results command when no results exist."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_results.return_value = []

        result = runner.invoke(app, ["simulation-results"])

        assert result.exit_code == 0
        assert "no simulation" in result.output.lower() or "empty" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.get_simulation_results")
    def test_simulation_results_default(
        self, mock_get_results, mock_settings, mock_session, mock_init_db
    ):
        """Test simulation-results command with defaults (no filters)."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_results.return_value = []

        result = runner.invoke(app, ["simulation-results"])

        assert result.exit_code == 0
        mock_get_results.assert_called_once()
        # Should use default limit of 10
        call_kwargs = mock_get_results.call_args.kwargs
        assert call_kwargs.get("limit") == 10 or call_kwargs.get("limit") is None


class TestSimulateErrorHandling:
    """Tests for error handling in simulate command."""

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    def test_simulate_ticker_not_found(
        self,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command when ticker price cannot be fetched."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance to return no price
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # No currentPrice
        mock_yf.Ticker.return_value = mock_ticker

        result = runner.invoke(app, ["simulate", "--ticker", "INVALID"])

        # CLI completes but reports the error and shows Failed count
        assert result.exit_code == 0
        assert "error" in result.output.lower() or "price" in result.output.lower()
        assert "failed: 1" in result.output.lower()

    @patch("investment_monitor.research_cli.init_db")
    @patch("investment_monitor.research_cli.get_session")
    @patch("investment_monitor.research_cli.get_settings")
    @patch("investment_monitor.research_cli.MonteCarloAnalyzer")
    @patch("investment_monitor.research_cli.yf")
    def test_simulate_analyzer_error(
        self,
        mock_yf,
        mock_analyzer_cls,
        mock_settings,
        mock_session,
        mock_init_db,
    ):
        """Test simulate command when analyzer raises an error."""
        # Setup mocks
        mock_settings.return_value = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yfinance
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 150.0}
        mock_yf.Ticker.return_value = mock_ticker

        # Mock analyzer to raise error
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = ValueError("Score below threshold")
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(app, ["simulate", "--ticker", "AAPL"])

        # CLI completes but reports the error and shows Failed count
        assert result.exit_code == 0
        assert "error" in result.output.lower()
        assert "failed: 1" in result.output.lower()
