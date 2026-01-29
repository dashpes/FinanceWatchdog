"""Tests for the command-line interface module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from investment_monitor.cli import create_parser, main
from investment_monitor.main import RunSummary


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_default_type(self):
        """Test default run type is 'regular'."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.type == "regular"

    def test_parser_type_choices(self):
        """Test all type choices are accepted."""
        parser = create_parser()

        for run_type in ["regular", "digest", "weekly"]:
            args = parser.parse_args(["--type", run_type])
            assert args.type == run_type

    def test_parser_type_short_flag(self):
        """Test -t short flag works."""
        parser = create_parser()
        args = parser.parse_args(["-t", "digest"])
        assert args.type == "digest"

    def test_parser_invalid_type(self):
        """Test invalid type raises error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--type", "invalid"])

    def test_parser_config_path(self):
        """Test config path argument."""
        parser = create_parser()
        args = parser.parse_args(["--config", "/path/to/config"])
        assert str(args.config) == "/path/to/config"

    def test_parser_config_short_flag(self):
        """Test -c short flag works."""
        parser = create_parser()
        args = parser.parse_args(["-c", "/path/to/config"])
        assert str(args.config) == "/path/to/config"

    def test_parser_log_level_default(self):
        """Test default log level is INFO."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.log_level == "INFO"

    def test_parser_log_level_choices(self):
        """Test all log level choices are accepted."""
        parser = create_parser()

        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parser.parse_args(["--log-level", level])
            assert args.log_level == level

    def test_parser_log_level_short_flag(self):
        """Test -l short flag works."""
        parser = create_parser()
        args = parser.parse_args(["-l", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_parser_dry_run(self):
        """Test dry-run flag."""
        parser = create_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parser_dry_run_short_flag(self):
        """Test -n short flag works."""
        parser = create_parser()
        args = parser.parse_args(["-n"])
        assert args.dry_run is True

    def test_parser_quiet(self):
        """Test quiet flag."""
        parser = create_parser()
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_parser_quiet_short_flag(self):
        """Test -q short flag works."""
        parser = create_parser()
        args = parser.parse_args(["-q"])
        assert args.quiet is True

    def test_parser_combined_flags(self):
        """Test combining multiple flags."""
        parser = create_parser()
        args = parser.parse_args([
            "-t", "digest",
            "-c", "/my/config",
            "-l", "DEBUG",
            "-q",
        ])
        assert args.type == "digest"
        assert str(args.config) == "/my/config"
        assert args.log_level == "DEBUG"
        assert args.quiet is True


class TestMain:
    """Tests for the main CLI function."""

    def test_main_dry_run(self, capsys):
        """Test dry run mode outputs correctly."""
        result = main(["--dry-run"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Dry run mode" in captured.out
        assert "regular" in captured.out

    def test_main_dry_run_with_options(self, capsys):
        """Test dry run shows all options."""
        result = main(["--dry-run", "-t", "weekly", "-c", "/custom/path"])
        assert result == 0

        captured = capsys.readouterr()
        assert "weekly" in captured.out
        assert "/custom/path" in captured.out

    def test_main_successful_run(self):
        """Test successful run returns 0."""
        mock_summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
            errors=[],
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            result = main([])

        assert result == 0
        mock_run.assert_called_once()

    def test_main_failed_run(self):
        """Test failed run returns 1."""
        mock_summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
            errors=["Something went wrong"],
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            result = main([])

        assert result == 1

    def test_main_passes_arguments(self):
        """Test arguments are passed to run_monitor_sync."""
        mock_summary = RunSummary(
            run_type="digest",
            started_at=datetime.now(),
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            main(["-t", "digest", "-l", "DEBUG"])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["run_type"] == "digest"
        assert call_kwargs["log_level"] == "DEBUG"

    def test_main_quiet_mode_uses_error_level(self):
        """Test quiet mode sets log level to ERROR."""
        mock_summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            main(["--quiet"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["log_level"] == "ERROR"

    def test_main_exception_returns_1(self, capsys):
        """Test unhandled exception returns 1."""
        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.side_effect = Exception("Unexpected error")
            result = main([])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_main_keyboard_interrupt_returns_130(self):
        """Test keyboard interrupt returns 130."""
        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            result = main([])

        assert result == 130

    def test_main_prints_summary(self, capsys):
        """Test summary is printed when not quiet."""
        mock_summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
            collectors_run=3,
            collectors_succeeded=3,
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            main([])

        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out

    def test_main_quiet_no_summary(self, capsys):
        """Test no summary is printed in quiet mode."""
        mock_summary = RunSummary(
            run_type="regular",
            started_at=datetime.now(),
        )

        with patch("investment_monitor.cli.run_monitor_sync") as mock_run:
            mock_run.return_value = mock_summary
            main(["--quiet"])

        captured = capsys.readouterr()
        # Should have no stdout output in quiet mode (unless errors)
        assert captured.out == ""
