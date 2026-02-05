"""Main orchestrator for the investment monitor.

This module ties all components together into a single entry point that can be
scheduled via cron. It supports three run modes:

- regular: Collect data, check alerts, send immediate (HIGH priority) notifications
- digest: Compile and send daily digest of MEDIUM priority alerts
- weekly: Run weekly synthesis with Claude API (if available)

Example usage:
    import asyncio
    from investment_monitor.main import run_monitor

    # Run regular monitoring
    asyncio.run(run_monitor(run_type="regular"))

    # Run daily digest
    asyncio.run(run_monitor(run_type="digest"))

    # Run weekly synthesis
    asyncio.run(run_monitor(run_type="weekly"))

Cron example (crontab):
    # Run regular monitoring every hour during market hours
    0 9-16 * * 1-5 cd /path/to/project && poetry run investment-monitor --type regular

    # Run daily digest at 5 PM on weekdays
    0 17 * * 1-5 cd /path/to/project && poetry run investment-monitor --type digest

    # Run weekly synthesis on Sunday at 8 PM
    0 20 * * 0 cd /path/to/project && poetry run investment-monitor --type weekly
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from investment_monitor.alerts import AlertDeduplicator, AlertEngine, classify_priority
from investment_monitor.analysis import ClaudeAnalyzer, LocalLLM, NewsProcessor, WeeklyData
from investment_monitor.collectors import (
    CollectorResult,
    EarningsCollector,
    ETFHoldingsCollector,
    InsiderCollector,
    NewsCollector,
    PriceCollector,
)
from investment_monitor.config import Settings
from investment_monitor.logging_config import setup_logging
from investment_monitor.models import AlertsConfig, Portfolio
from investment_monitor.notifications import (
    AlertMessage,
    ConsoleChannel,
    DiscordChannel,
    NotificationManager,
    Priority,
    format_daily_digest,
    format_weekly_digest,
)
from investment_monitor.storage import (
    get_recent_alerts,
    get_recent_news,
    get_session,
    get_upcoming_earnings,
    init_db,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class RunSummary:
    """Summary of a monitor run for logging and reporting."""

    run_type: str
    started_at: datetime
    finished_at: datetime = field(default_factory=datetime.now)
    collectors_run: int = 0
    collectors_succeeded: int = 0
    records_collected: int = 0
    alerts_generated: int = 0
    alerts_sent: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate run duration in seconds."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        """Check if run was successful (no critical errors)."""
        return len(self.errors) == 0

    def __str__(self) -> str:
        """Format summary for logging."""
        status = "SUCCESS" if self.success else "COMPLETED WITH ERRORS"
        lines = [
            f"Run Summary: {status}",
            f"  Type: {self.run_type}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Collectors: {self.collectors_succeeded}/{self.collectors_run} succeeded",
            f"  Records collected: {self.records_collected}",
            f"  Alerts: {self.alerts_generated} generated, {self.alerts_sent} sent",
        ]
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for err in self.errors[:5]:  # Show first 5 errors
                lines.append(f"    - {err}")
        return "\n".join(lines)


async def run_monitor(
    config_path: str | Path | None = None,
    run_type: str = "regular",
    log_level: str = "INFO",
) -> RunSummary:
    """Main entry point for the investment monitor.

    Args:
        config_path: Path to configuration directory. If None, uses Settings default.
        run_type: Type of run - "regular", "digest", or "weekly"
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        RunSummary with details about the run.

    Run types:
        regular: Collect data, check alerts, send immediate notifications
        digest: Compile and send daily digest
        weekly: Run weekly synthesis with Claude API
    """
    start_time = datetime.now()
    summary = RunSummary(run_type=run_type, started_at=start_time)

    # Setup logging
    settings = Settings()
    setup_logging(log_dir=str(settings.log_dir), log_level=log_level)

    logger.info(f"Starting investment monitor run: {run_type}")

    # Override config directory if provided
    if config_path:
        settings.config_dir = Path(config_path)

    # Load configuration files
    try:
        portfolio = _load_portfolio(settings.config_dir)
        alerts_config = _load_alerts_config(settings.config_dir)
    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        logger.error(error_msg)
        summary.errors.append(error_msg)
        summary.finished_at = datetime.now()
        return summary

    # Initialize database
    try:
        init_db(settings.db_path)
    except Exception as e:
        error_msg = f"Failed to initialize database: {e}"
        logger.error(error_msg)
        summary.errors.append(error_msg)
        summary.finished_at = datetime.now()
        return summary

    # Create notification channels
    console_channel = ConsoleChannel()

    # Set up Discord channels - supports separate daily/weekly webhooks
    # Priority: specific URL > fallback URL > None
    daily_discord_url = settings.discord_daily_webhook_url or settings.discord_webhook_url
    weekly_discord_url = settings.discord_weekly_webhook_url or settings.discord_webhook_url

    discord_daily: DiscordChannel | None = None
    discord_weekly: DiscordChannel | None = None

    if daily_discord_url:
        try:
            discord_daily = DiscordChannel(daily_discord_url)
            logger.info("Discord daily notifications enabled")
        except ValueError as e:
            logger.warning("Discord daily channel not configured: {error}", error=str(e))

    if weekly_discord_url:
        try:
            # Reuse daily channel if URLs are the same
            if weekly_discord_url == daily_discord_url and discord_daily:
                discord_weekly = discord_daily
            else:
                discord_weekly = DiscordChannel(weekly_discord_url)
            logger.info("Discord weekly notifications enabled")
        except ValueError as e:
            logger.warning("Discord weekly channel not configured: {error}", error=str(e))

    # Build channel list for regular notifications (immediate alerts use daily channel)
    channels: list[ConsoleChannel | DiscordChannel] = [console_channel]
    if discord_daily:
        channels.append(discord_daily)

    notification_manager = NotificationManager(channels)

    with get_session() as session:
        try:
            if run_type in ("regular", "digest"):
                # Run collectors
                collector_results = await _run_collectors(session, settings, portfolio)
                summary.collectors_run = len(collector_results)
                summary.collectors_succeeded = sum(1 for r in collector_results if r.success)
                summary.records_collected = sum(r.records_collected for r in collector_results)

                # Process news with AI (if available)
                await _process_news_ai(session, settings, portfolio)

                # Run alert engine
                alerts = _run_alert_checks(session, portfolio, alerts_config)
                summary.alerts_generated = len(alerts)

                # Deduplicate and classify priority
                deduplicator = AlertDeduplicator(session)
                alerts = deduplicator.filter_duplicates(alerts)

                for alert in alerts:
                    alert.priority = classify_priority(alert, alerts_config)

                # Save ALL alerts to database for digest (not just sent ones)
                for alert in alerts:
                    if alert.priority != Priority.HIGH:
                        # Save non-HIGH alerts as "queued" for digest
                        deduplicator.mark_sent(alert, "queued")

                # Send immediate alerts (HIGH priority)
                sent_count = await _send_immediate_alerts(
                    alerts, session, deduplicator, notification_manager
                )
                summary.alerts_sent = sent_count

            if run_type == "digest":
                await _send_daily_digest(session, portfolio, console_channel, discord_daily)

            if run_type == "weekly":
                await _send_weekly_digest(session, portfolio, alerts_config, settings, console_channel, discord_weekly)

        except Exception as e:
            error_msg = f"Monitor run failed: {e}"
            logger.exception(error_msg)
            summary.errors.append(error_msg)

    summary.finished_at = datetime.now()
    logger.info(str(summary))

    return summary


def _load_portfolio(config_dir: Path) -> Portfolio:
    """Load portfolio configuration from YAML.

    Args:
        config_dir: Directory containing portfolio.yaml

    Returns:
        Portfolio instance

    Raises:
        FileNotFoundError: If portfolio.yaml doesn't exist
    """
    portfolio_path = config_dir / "portfolio.yaml"
    if not portfolio_path.exists():
        logger.warning(f"Portfolio file not found: {portfolio_path}, using empty portfolio")
        return Portfolio()
    return Portfolio.from_yaml(portfolio_path)


def _load_alerts_config(config_dir: Path) -> AlertsConfig:
    """Load alerts configuration from YAML.

    Args:
        config_dir: Directory containing alerts.yaml

    Returns:
        AlertsConfig instance
    """
    alerts_path = config_dir / "alerts.yaml"
    if not alerts_path.exists():
        logger.warning(f"Alerts config not found: {alerts_path}, using defaults")
        return AlertsConfig()
    return AlertsConfig.from_yaml(alerts_path)


async def _run_collectors(
    session: Session,
    settings: Settings,
    portfolio: Portfolio,
) -> list[CollectorResult]:
    """Run all data collectors with error isolation.

    Each collector is run independently - failures in one collector
    don't prevent others from running.

    Args:
        session: Database session
        settings: Application settings
        portfolio: Portfolio with tickers to collect

    Returns:
        List of CollectorResult objects
    """
    tickers = portfolio.all_tickers
    if not tickers:
        logger.warning("No tickers to collect data for")
        return []

    logger.info(f"Running collectors for {len(tickers)} tickers: {', '.join(tickers)}")

    collectors = [
        PriceCollector(session, settings),
        InsiderCollector(session, settings),
        NewsCollector(session, settings),
        EarningsCollector(session, settings),
        ETFHoldingsCollector(session, settings),
    ]

    results: list[CollectorResult] = []

    for collector in collectors:
        try:
            result = await collector.run(tickers)
            results.append(result)
            logger.info(f"{collector.name}: collected {result.records_collected} records")
        except Exception as e:
            logger.error(f"{collector.name} failed: {e}")
            # Create a failed result for tracking
            results.append(
                CollectorResult(
                    collector_name=collector.name,
                    success=False,
                    records_collected=0,
                    errors=[str(e)],
                    started_at=datetime.now(),
                    finished_at=datetime.now(),
                )
            )

    return results


async def _process_news_ai(
    session: Session,
    settings: Settings,
    portfolio: Portfolio,
) -> int:
    """Process news with local LLM for relevance scoring.

    This is optional - if Ollama is not available, news will be
    unscored but still available for keyword-based alerts.

    Args:
        session: Database session
        settings: Application settings
        portfolio: Portfolio for context

    Returns:
        Number of news items processed
    """
    llm = LocalLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_host,
    )

    if not llm.is_available():
        logger.debug("Local LLM not available, skipping news AI processing")
        return 0

    processor = NewsProcessor(
        session=session,
        llm=llm,
        portfolio=portfolio,
        min_relevance=5.0,
    )

    try:
        count = await processor.process_unscored_news(batch_size=50)
        if count > 0:
            logger.info(f"Processed {count} news items with AI")
        return count
    except Exception as e:
        logger.warning(f"News AI processing failed: {e}")
        return 0


def _run_alert_checks(
    session: Session,
    portfolio: Portfolio,
    alerts_config: AlertsConfig,
) -> list[AlertMessage]:
    """Run all alert checks and return triggered alerts.

    Args:
        session: Database session
        portfolio: Portfolio configuration
        alerts_config: Alert configuration

    Returns:
        List of triggered AlertMessage objects
    """
    engine = AlertEngine(session, portfolio, alerts_config)
    alerts = engine.run_all_checks()

    logger.info(f"Alert engine generated {len(alerts)} alerts")
    return alerts


async def _send_immediate_alerts(
    alerts: list[AlertMessage],
    session: Session,
    deduplicator: AlertDeduplicator,
    notification_manager: NotificationManager,
) -> int:
    """Send HIGH priority alerts immediately.

    Args:
        alerts: List of alerts to process
        session: Database session
        deduplicator: Alert deduplicator for tracking sent alerts
        notification_manager: Notification manager

    Returns:
        Number of alerts sent
    """
    high_priority = [a for a in alerts if a.priority == Priority.HIGH]

    if not high_priority:
        logger.debug("No high-priority alerts to send")
        return 0

    logger.info(f"Sending {len(high_priority)} high-priority alerts")

    sent_count = 0
    for alert in high_priority:
        try:
            await notification_manager.notify(alert)
            # Mark as sent in database
            deduplicator.mark_sent(alert, "immediate")
            sent_count += 1
        except Exception as e:
            logger.error(f"Failed to send alert: {alert.title}: {e}")

    return sent_count


async def _send_daily_digest(
    session: Session,
    portfolio: Portfolio,
    console_channel: ConsoleChannel,
    discord_channel: DiscordChannel | None,
) -> None:
    """Compile and send daily digest.

    Retrieves recent alerts from the database and formats them
    into a daily digest.

    Args:
        session: Database session
        portfolio: Portfolio for context
        console_channel: Console channel for logging
        discord_channel: Discord channel for daily notifications (optional)
    """
    # Get alerts from the last 24 hours
    recent_alerts = get_recent_alerts(session, hours=24)

    # Convert database records to AlertMessage objects
    messages = []
    for alert_record in recent_alerts:
        try:
            msg = AlertMessage(
                title=f"{alert_record.alert_type.title()} Alert: {alert_record.ticker}",
                body=alert_record.message,
                ticker=alert_record.ticker if alert_record.ticker else None,
                alert_type=alert_record.alert_type,
                priority=Priority(alert_record.priority) if alert_record.priority else Priority.MEDIUM,
            )
            messages.append(msg)
        except Exception as e:
            logger.warning(f"Failed to convert alert record: {e}")

    if not messages:
        logger.info("No alerts for daily digest")
        return

    logger.info(f"Sending daily digest with {len(messages)} alerts")

    # Format and log the digest
    plain_text, html = format_daily_digest(messages, portfolio, date.today())
    logger.debug(f"Daily digest:\n{plain_text}")

    # Send to console
    try:
        await console_channel.send_digest(messages)
    except Exception as e:
        logger.error(f"Failed to send daily digest via console: {e}")

    # Send to Discord if configured
    if discord_channel:
        try:
            await discord_channel.send_digest(messages, portfolio=portfolio, is_weekly=False)
        except Exception as e:
            logger.error(f"Failed to send daily digest via Discord: {e}")


async def _send_weekly_digest(
    session: Session,
    portfolio: Portfolio,
    alerts_config: AlertsConfig,
    settings: Settings,
    console_channel: ConsoleChannel,
    discord_channel: DiscordChannel | None,
) -> None:
    """Compile and send weekly digest with optional AI synthesis.

    Args:
        session: Database session
        portfolio: Portfolio for context
        alerts_config: Alerts configuration
        settings: Application settings
        console_channel: Console channel for logging
        discord_channel: Discord channel for weekly notifications (optional)
    """
    week_end = date.today()
    week_start = week_end - timedelta(days=6)

    # Get alerts from the last 7 days
    recent_alerts = get_recent_alerts(session, hours=168)  # 7 * 24 = 168 hours

    # Convert to AlertMessage objects
    messages = []
    for alert_record in recent_alerts:
        try:
            msg = AlertMessage(
                title=f"{alert_record.alert_type.title()} Alert: {alert_record.ticker}",
                body=alert_record.message,
                ticker=alert_record.ticker if alert_record.ticker else None,
                alert_type=alert_record.alert_type,
                priority=Priority(alert_record.priority) if alert_record.priority else Priority.MEDIUM,
            )
            messages.append(msg)
        except Exception as e:
            logger.warning(f"Failed to convert alert record: {e}")

    # Prepare data for AI synthesis
    ai_synthesis = None

    if settings.anthropic_api_key:
        analyzer = ClaudeAnalyzer(
            api_key=settings.anthropic_api_key,
            max_monthly_spend=5.00,
        )

        if analyzer.is_available():
            # Build weekly data summary
            week_data = _build_weekly_data(session, portfolio, week_start, week_end)

            result = await analyzer.weekly_synthesis(portfolio, week_data)

            if result.success:
                ai_synthesis = result.synthesis
                logger.info(
                    f"Generated AI synthesis ({result.input_tokens} input, "
                    f"{result.output_tokens} output tokens, ${result.cost:.4f})"
                )
            else:
                logger.warning(f"AI synthesis failed: {result.error_message}")
        else:
            logger.info("Claude API not available for weekly synthesis")

    # Format and log the digest
    plain_text, html = format_weekly_digest(
        messages, portfolio, week_start, week_end, ai_synthesis
    )
    logger.info(f"Weekly digest:\n{plain_text}")

    # Send via each channel
    if messages or ai_synthesis:
        logger.info(f"Weekly digest contains {len(messages)} alerts")

        # Send to console
        try:
            await console_channel.send_digest(messages)
        except Exception as e:
            logger.error(f"Failed to send weekly digest via console: {e}")

        # Send to Discord if configured
        if discord_channel:
            try:
                await discord_channel.send_digest(
                    messages,
                    portfolio=portfolio,
                    is_weekly=True,
                    ai_synthesis=ai_synthesis,
                )
            except Exception as e:
                logger.error(f"Failed to send weekly digest via Discord: {e}")
    else:
        logger.info("No content for weekly digest")


def _build_weekly_data(
    session: Session,
    portfolio: Portfolio,
    week_start: date,
    week_end: date,
) -> WeeklyData:
    """Build summary data for weekly AI synthesis.

    Args:
        session: Database session
        portfolio: Portfolio configuration
        week_start: Start of the week
        week_end: End of the week

    Returns:
        WeeklyData with summaries for AI analysis
    """
    # Get recent news
    news_items = get_recent_news(session, hours=168)
    news_summary = "No significant news this week."
    if news_items:
        headlines = [f"- {item.ticker}: {item.headline}" for item in news_items[:10]]
        news_summary = "\n".join(headlines)

    # Get upcoming earnings
    earnings = get_upcoming_earnings(session, days=7)
    earnings_summary = "No upcoming earnings in the next 7 days."
    if earnings:
        earnings_list = [f"- {e.ticker}: {e.date}" for e in earnings]
        earnings_summary = "\n".join(earnings_list)

    # Get recent alerts for price/insider summaries
    alerts = get_recent_alerts(session, hours=168)

    price_alerts = [a for a in alerts if a.alert_type == "price"]
    price_summary = "No significant price movements this week."
    if price_alerts:
        price_list = [f"- {a.ticker}: {a.message}" for a in price_alerts[:5]]
        price_summary = "\n".join(price_list)

    insider_alerts = [a for a in alerts if a.alert_type == "insider"]
    insider_summary = "No insider transactions reported."
    if insider_alerts:
        insider_list = [f"- {a.ticker}: {a.message}" for a in insider_alerts[:5]]
        insider_summary = "\n".join(insider_list)

    return WeeklyData(
        price_summary=price_summary,
        insider_summary=insider_summary,
        news_summary=news_summary,
        earnings_summary=earnings_summary,
        week_start=week_start,
        week_end=week_end,
    )


# Convenience function for synchronous usage
def run_monitor_sync(
    config_path: str | Path | None = None,
    run_type: str = "regular",
    log_level: str = "INFO",
) -> RunSummary:
    """Synchronous wrapper for run_monitor.

    Args:
        config_path: Path to configuration directory
        run_type: Type of run - "regular", "digest", or "weekly"
        log_level: Logging level

    Returns:
        RunSummary with details about the run
    """
    return asyncio.run(run_monitor(config_path, run_type, log_level))
