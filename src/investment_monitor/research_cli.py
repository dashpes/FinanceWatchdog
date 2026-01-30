"""Command-line interface for the investment research system.

This module provides CLI commands for stock research operations including:
- Discovery pipeline execution
- Single ticker analysis
- Research queue management
- Viewing top candidates and reports
- Managing research profiles

Usage:
    investment-research discover [--dry-run]
    investment-research analyze TICKER [--no-report]
    investment-research queue list
    investment-research queue add TICKER [--priority N]
    investment-research queue remove TICKER
    investment-research queue process [--max N]
    investment-research top [--limit N] [--min-score N]
    investment-research report TICKER
    investment-research profile [--show]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

from investment_monitor.config import get_settings
from investment_monitor.models import ResearchConfig
from investment_monitor.research.discovery import DiscoveryPipeline
from investment_monitor.research.orchestrator import ResearchOrchestrator
from investment_monitor.research.queue import ResearchQueue
from investment_monitor.storage import (
    get_latest_report,
    get_or_create_default_profile,
    get_session,
    get_top_candidates,
    init_db,
)

# Create the main app and queue subcommand
app = typer.Typer(
    name="investment-research",
    help="Stock research and discovery CLI",
    no_args_is_help=True,
)

queue_app = typer.Typer(
    name="queue",
    help="Research queue management commands",
    no_args_is_help=True,
)
app.add_typer(queue_app, name="queue")


def _load_research_config(config_dir: Path) -> ResearchConfig:
    """Load research config from YAML or use defaults."""
    config_path = config_dir / "research.yaml"
    if config_path.exists():
        return ResearchConfig.from_yaml(config_path)
    return ResearchConfig()


@app.command()
def discover(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Run discovery without persisting changes",
    ),
) -> None:
    """Run the discovery pipeline to find and score stock candidates.

    The discovery pipeline:
    1. Collects stock universe from S&P 500, NASDAQ 100, and ETFs
    2. Applies filters (market cap, sectors)
    3. Fetches fundamentals for candidates
    4. Scores candidates using AI analysis
    5. Identifies top candidates above threshold
    6. Auto-adds high-scoring candidates to watchlist
    """
    try:
        settings = get_settings()
        init_db(settings.db_path)
        research_config = _load_research_config(settings.config_dir)

        with get_session() as session:
            pipeline = DiscoveryPipeline(
                session=session,
                config=settings,
                research_config=research_config,
                ollama_model=settings.ollama_model,
            )

            if dry_run:
                typer.echo("Running discovery pipeline in dry-run mode...")
            else:
                typer.echo("Running discovery pipeline...")

            result = asyncio.run(pipeline.run_discovery(dry_run=dry_run))

            # Display results
            typer.echo("\n--- Discovery Results ---")
            typer.echo(f"Total candidates: {result.total_candidates}")
            typer.echo(f"Scored candidates: {result.scored_candidates}")
            typer.echo(f"Top candidates: {len(result.top_candidates)}")
            typer.echo(f"Watchlist additions: {len(result.watchlist_additions)}")
            typer.echo(f"Duration: {result.duration_seconds:.1f}s")

            if result.top_candidates:
                typer.echo(f"\nTop candidates: {', '.join(result.top_candidates[:10])}")

            if result.watchlist_additions:
                typer.echo(
                    f"Added to watchlist: {', '.join(result.watchlist_additions)}"
                )

            if result.errors:
                typer.echo(f"\nErrors ({len(result.errors)}):")
                for error in result.errors[:5]:
                    typer.echo(f"  - {error}")
                if len(result.errors) > 5:
                    typer.echo(f"  ... and {len(result.errors) - 5} more")

            if dry_run:
                typer.echo("\n(Dry run - no changes persisted)")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker symbol to analyze"),
    no_report: bool = typer.Option(
        False,
        "--no-report",
        help="Skip generating full research report",
    ),
) -> None:
    """Analyze a single ticker with deep research.

    Performs comprehensive analysis including:
    - Fetching fundamentals
    - Gathering news
    - Checking congressional trades
    - Generating AI research report (unless --no-report)
    """
    try:
        settings = get_settings()
        init_db(settings.db_path)
        research_config = _load_research_config(settings.config_dir)

        ticker = ticker.upper()
        typer.echo(f"Analyzing {ticker}...")

        with get_session() as session:
            orchestrator = ResearchOrchestrator(
                session=session,
                config=settings,
                research_config=research_config,
            )

            result = asyncio.run(orchestrator.research_ticker(ticker))

            if result.success:
                typer.echo(f"\n--- Research Complete for {ticker} ---")
                typer.echo(f"Duration: {result.duration:.1f}s")

                if result.report and not no_report:
                    report = result.report
                    typer.echo(f"\nRecommendation: {report.recommendation or 'N/A'}")
                    if report.target_price:
                        typer.echo(f"Target Price: ${report.target_price:.2f}")
                    if report.summary:
                        typer.echo(f"\nSummary:\n{report.summary}")
                else:
                    typer.echo("\nResearch data collected successfully.")
            else:
                typer.echo(f"\nResearch failed for {ticker}: {result.error}", err=True)
                raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@queue_app.command("list")
def queue_list(
    limit: int = typer.Option(
        50,
        "--limit",
        "-l",
        help="Maximum number of queue items to show",
    ),
) -> None:
    """Show items in the research queue."""
    try:
        settings = get_settings()
        init_db(settings.db_path)

        with get_session() as session:
            queue = ResearchQueue(session)
            items = queue.get_queue(limit=limit)

            if not items:
                typer.echo("Research queue is empty.")
                return

            typer.echo(f"\n--- Research Queue ({len(items)} items) ---")
            typer.echo(f"{'#':<4} {'Ticker':<10} {'Priority':<10}")
            typer.echo("-" * 30)

            for i, candidate in enumerate(items, 1):
                priority = candidate.composite_score or 0
                typer.echo(f"{i:<4} {candidate.ticker:<10} {priority:<10.1f}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@queue_app.command("add")
def queue_add(
    ticker: str = typer.Argument(..., help="Stock ticker symbol to add"),
    priority: int = typer.Option(
        0,
        "--priority",
        "-p",
        help="Priority score (higher = processed first)",
    ),
) -> None:
    """Add a ticker to the research queue."""
    try:
        settings = get_settings()
        init_db(settings.db_path)

        ticker = ticker.upper()

        with get_session() as session:
            queue = ResearchQueue(session)
            success = queue.add_to_queue(ticker, priority=priority)

            if success:
                typer.echo(f"Added {ticker} to research queue with priority {priority}")
            else:
                typer.echo(f"Failed to add {ticker} to queue", err=True)
                raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@queue_app.command("remove")
def queue_remove(
    ticker: str = typer.Argument(..., help="Stock ticker symbol to remove"),
) -> None:
    """Remove a ticker from the research queue."""
    try:
        settings = get_settings()
        init_db(settings.db_path)

        ticker = ticker.upper()

        with get_session() as session:
            queue = ResearchQueue(session)
            success = queue.remove_from_queue(ticker)

            if success:
                typer.echo(f"Removed {ticker} from research queue")
            else:
                typer.echo(
                    f"Failed to remove {ticker} from queue (not found or not in queue)",
                    err=True,
                )
                raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@queue_app.command("process")
def queue_process(
    max_items: int = typer.Option(
        5,
        "--max",
        "-m",
        help="Maximum number of items to process",
    ),
) -> None:
    """Process items from the research queue.

    Processes queue items in priority order, generating research reports
    for each ticker.
    """
    try:
        settings = get_settings()
        init_db(settings.db_path)
        research_config = _load_research_config(settings.config_dir)

        typer.echo(f"Processing up to {max_items} items from research queue...")

        with get_session() as session:
            orchestrator = ResearchOrchestrator(
                session=session,
                config=settings,
                research_config=research_config,
            )

            results = asyncio.run(orchestrator.process_queue(max_items=max_items))

            if not results:
                typer.echo("Research queue is empty.")
                return

            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful

            typer.echo(f"\n--- Queue Processing Complete ---")
            typer.echo(f"Processed: {len(results)} items")
            typer.echo(f"Successful: {successful}")
            typer.echo(f"Failed: {failed}")

            if results:
                typer.echo(f"\n{'Ticker':<10} {'Status':<10} {'Duration':<10}")
                typer.echo("-" * 35)
                for result in results:
                    status = "OK" if result.success else "FAILED"
                    typer.echo(
                        f"{result.ticker:<10} {status:<10} {result.duration:.1f}s"
                    )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def top(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of candidates to show",
    ),
    min_score: Optional[float] = typer.Option(
        None,
        "--min-score",
        "-s",
        help="Minimum composite score threshold",
    ),
) -> None:
    """Show top-scoring stock candidates.

    Lists candidates ordered by composite score, optionally filtered
    by minimum score.
    """
    try:
        settings = get_settings()
        init_db(settings.db_path)

        with get_session() as session:
            candidates = get_top_candidates(
                session, limit=limit, min_score=min_score
            )

            if not candidates:
                typer.echo("No candidates found matching criteria.")
                return

            typer.echo(f"\n--- Top Candidates ({len(candidates)}) ---")
            typer.echo(f"{'#':<4} {'Ticker':<10} {'Score':<10} {'Status':<12}")
            typer.echo("-" * 40)

            for i, candidate in enumerate(candidates, 1):
                score = candidate.composite_score or 0
                typer.echo(
                    f"{i:<4} {candidate.ticker:<10} {score:<10.1f} {candidate.status:<12}"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def report(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
) -> None:
    """Show the latest research report for a ticker."""
    try:
        settings = get_settings()
        init_db(settings.db_path)

        ticker = ticker.upper()

        with get_session() as session:
            research_report = get_latest_report(session, ticker)

            if not research_report:
                typer.echo(f"No report found for {ticker}", err=True)
                raise typer.Exit(code=1)

            typer.echo(f"\n{'='*60}")
            typer.echo(f"Research Report: {ticker}")
            typer.echo(f"{'='*60}")

            if research_report.created_at:
                typer.echo(
                    f"Generated: {research_report.created_at.strftime('%Y-%m-%d %H:%M')}"
                )

            if research_report.recommendation:
                typer.echo(f"\nRecommendation: {research_report.recommendation}")

            if research_report.target_price:
                typer.echo(f"Target Price: ${research_report.target_price:.2f}")

            if research_report.summary:
                typer.echo(f"\n--- Summary ---\n{research_report.summary}")

            if research_report.bull_case:
                typer.echo(f"\n--- Bull Case ---\n{research_report.bull_case}")

            if research_report.bear_case:
                typer.echo(f"\n--- Bear Case ---\n{research_report.bear_case}")

            if research_report.thesis:
                typer.echo(f"\n--- Investment Thesis ---\n{research_report.thesis}")

            typer.echo(f"\n{'='*60}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def profile(
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Show current research profile",
    ),
) -> None:
    """Show or manage research profile settings.

    The research profile contains scoring weights and preferences
    that influence how candidates are evaluated.
    """
    try:
        settings = get_settings()
        init_db(settings.db_path)

        with get_session() as session:
            research_profile = get_or_create_default_profile(session)

            typer.echo(f"\n--- Research Profile: {research_profile.name} ---")

            if research_profile.investment_style:
                typer.echo(f"Investment Style: {research_profile.investment_style}")

            if research_profile.risk_tolerance:
                typer.echo(f"Risk Tolerance: {research_profile.risk_tolerance}")

            typer.echo("\nScoring Weights:")
            typer.echo(f"  Value:     {research_profile.value_weight:.1%}")
            typer.echo(f"  Growth:    {research_profile.growth_weight:.1%}")
            typer.echo(f"  Quality:   {research_profile.quality_weight:.1%}")
            typer.echo(f"  Momentum:  {research_profile.momentum_weight:.1%}")
            typer.echo(f"  Sentiment: {research_profile.sentiment_weight:.1%}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> int:
    """Main entry point for the CLI."""
    try:
        app()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
