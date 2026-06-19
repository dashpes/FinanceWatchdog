"""Command-line interface for the investment monitor.

This module provides the CLI entry point for running the investment monitor
from the command line or as a cron job.

Usage:
    # Run regular monitoring (default)
    investment-monitor

    # Run with specific type
    investment-monitor --type regular
    investment-monitor --type digest
    investment-monitor --type weekly

    # Specify custom config directory
    investment-monitor --config /path/to/config

    # Set log level
    investment-monitor --log-level DEBUG

    # Show version
    investment-monitor --version

    # Dry run (show what would be done without actually doing it)
    investment-monitor --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from investment_monitor import __version__
from investment_monitor.main import run_monitor_sync


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="investment-monitor",
        description="Personal investment monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  investment-monitor                    Run regular monitoring
  investment-monitor --type digest      Send daily digest
  investment-monitor --type weekly      Send weekly synthesis
  investment-monitor --config ./myconfig  Use custom config directory
  investment-monitor --log-level DEBUG  Enable debug logging

Cron examples:
  # Run every hour during market hours (9 AM - 4 PM, Mon-Fri)
  0 9-16 * * 1-5 investment-monitor --type regular

  # Send daily digest at 5 PM on weekdays
  0 17 * * 1-5 investment-monitor --type digest

  # Send weekly synthesis on Sunday at 8 PM
  0 20 * * 0 investment-monitor --type weekly
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["regular", "digest", "weekly", "collect-broad", "insights", "prune"],
        default="regular",
        help="Type of run: regular (collect data, check alerts), "
        "digest (daily summary), weekly (AI synthesis), "
        "collect-broad (market-wide event ingestion, universe-independent), "
        "insights (run the confluence engine over collected data), "
        "prune (delete data older than the --*-days windows). "
        "Default: regular",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to configuration directory containing portfolio.yaml and alerts.yaml",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    parser.add_argument("--insider-days", type=int, default=0,
                        help="For --type prune: delete insider rows older than N days (0=keep all).")
    parser.add_argument("--news-days", type=int, default=0,
                        help="For --type prune: delete news rows older than N days (0=keep all).")
    parser.add_argument("--price-days", type=int, default=0,
                        help="For --type prune: delete price rows older than N days (0=keep all).")
    parser.add_argument("--findings-days", type=int, default=0,
                        help="For --type prune: delete findings older than N days (0=keep all).")

    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="For --type collect-broad: recent business days of SEC indexes to "
        "ingest (default 1; dedup makes overlap safe for catch-up).",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle quiet mode
    log_level = "ERROR" if args.quiet else args.log_level

    # Handle dry run
    if args.dry_run:
        print(f"Dry run mode - would execute:")
        print(f"  Run type: {args.type}")
        print(f"  Config path: {args.config or 'default'}")
        print(f"  Log level: {log_level}")
        return 0

    # Broad, universe-independent collection runs a different pipeline.
    if args.type == "collect-broad":
        from investment_monitor.broad_collect import run_broad_collection_sync

        try:
            results = run_broad_collection_sync(days_back=args.days_back)
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            for r in results:
                print(str(r))
        return 0 if all(r.success for r in results) else 1

    # Confluence/insight engine: turn collected data into stated cross-source insights.
    if args.type == "insights":
        from investment_monitor.analysis.confluence import run_confluence

        try:
            findings = run_confluence()
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if not findings:
            print("No confluence findings.")
        else:
            print(f"{len(findings)} confluence finding(s):\n")
            for f in findings:
                print(f"  [{f['score']:>5.1f}] {f['narrative']}")
        return 0

    # Retention: delete data older than the requested per-source windows (0 = keep).
    if args.type == "prune":
        from investment_monitor.config import get_settings
        from investment_monitor.storage import (
            RetentionConfig, get_session, init_db, prune_old_data,
        )

        cfg = RetentionConfig(
            insider_days=args.insider_days, news_days=args.news_days,
            price_days=args.price_days, findings_days=args.findings_days,
        )
        try:
            init_db(get_settings().db_path)
            with get_session() as s:
                deleted = prune_old_data(s, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print("Pruned:" if deleted else "Nothing to prune (all windows 0 / no old rows).")
        for table, n in (deleted or {}).items():
            print(f"  {table}: {n} rows deleted")
        return 0

    try:
        # Run the monitor
        summary = run_monitor_sync(
            config_path=args.config,
            run_type=args.type,
            log_level=log_level,
        )

        # Print summary unless quiet
        if not args.quiet:
            print(str(summary))

        # Return exit code based on success
        return 0 if summary.success else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
