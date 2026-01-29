"""Alert engine for processing all alert rules.

The AlertEngine orchestrates running all configured alert checks and
collecting the resulting alerts. It respects the enabled/disabled state
of each alert type in the configuration.
"""

import logging
from typing import Callable

from sqlalchemy.orm import Session

from investment_monitor.models import AlertsConfig, Portfolio
from investment_monitor.notifications import AlertMessage

from .rules import (
    check_earnings_alerts,
    check_insider_alerts,
    check_news_keyword_alerts,
    check_price_alerts,
    check_volume_alerts,
)

logger = logging.getLogger(__name__)


class AlertEngine:
    """Engine for running alert checks against portfolio data.

    The engine runs all enabled alert checks and collects triggered alerts.
    Each check is run independently, so a failure in one check does not
    prevent others from running.

    Example:
        with get_session() as session:
            engine = AlertEngine(session, portfolio, alerts_config)
            alerts = engine.run_all_checks()

            for alert in alerts:
                print(alert.format_full())
    """

    def __init__(
        self,
        session: Session,
        portfolio: Portfolio,
        alerts_config: AlertsConfig,
    ) -> None:
        """Initialize the alert engine.

        Args:
            session: Database session for querying data
            portfolio: Portfolio with holdings and watchlist
            alerts_config: Configuration for all alert types
        """
        self.session = session
        self.portfolio = portfolio
        self.alerts_config = alerts_config

    def run_all_checks(self) -> list[AlertMessage]:
        """Run all enabled alert checks and return triggered alerts.

        Each check is run independently. If a check fails, it logs an error
        but continues with other checks.

        Returns:
            List of all triggered AlertMessage objects, sorted by priority
        """
        alerts: list[AlertMessage] = []

        # Define checks with their enabled flags
        checks: list[tuple[bool, str, Callable[[], list[AlertMessage]]]] = [
            (
                self.alerts_config.price.enabled,
                "price",
                lambda: check_price_alerts(
                    self.session, self.portfolio, self.alerts_config.price
                ),
            ),
            (
                self.alerts_config.volume.enabled,
                "volume",
                lambda: check_volume_alerts(
                    self.session, self.portfolio, self.alerts_config.volume
                ),
            ),
            (
                self.alerts_config.insider.enabled,
                "insider",
                lambda: check_insider_alerts(
                    self.session, self.portfolio, self.alerts_config.insider
                ),
            ),
            (
                self.alerts_config.earnings.enabled,
                "earnings",
                lambda: check_earnings_alerts(
                    self.session, self.portfolio, self.alerts_config.earnings
                ),
            ),
            (
                self.alerts_config.news.enabled,
                "news",
                lambda: check_news_keyword_alerts(
                    self.session, self.portfolio, self.alerts_config.news
                ),
            ),
        ]

        for enabled, name, check_fn in checks:
            if not enabled:
                logger.debug("Skipping disabled %s alerts", name)
                continue

            try:
                logger.debug("Running %s alert check", name)
                check_alerts = check_fn()
                alerts.extend(check_alerts)
                logger.info("Found %d %s alerts", len(check_alerts), name)
            except Exception as e:
                logger.error("Error in %s alert check: %s", name, e, exc_info=True)

        # Sort by priority (HIGH first, then MEDIUM, then LOW)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        alerts.sort(key=lambda a: priority_order.get(a.priority.value, 99))

        logger.info("Total alerts triggered: %d", len(alerts))
        return alerts

    def run_price_checks(self) -> list[AlertMessage]:
        """Run only price-related alert checks.

        Returns:
            List of triggered price alerts
        """
        if not self.alerts_config.price.enabled:
            return []

        return check_price_alerts(
            self.session, self.portfolio, self.alerts_config.price
        )

    def run_volume_checks(self) -> list[AlertMessage]:
        """Run only volume-related alert checks.

        Returns:
            List of triggered volume alerts
        """
        if not self.alerts_config.volume.enabled:
            return []

        return check_volume_alerts(
            self.session, self.portfolio, self.alerts_config.volume
        )

    def run_insider_checks(self) -> list[AlertMessage]:
        """Run only insider trading alert checks.

        Returns:
            List of triggered insider alerts
        """
        if not self.alerts_config.insider.enabled:
            return []

        return check_insider_alerts(
            self.session, self.portfolio, self.alerts_config.insider
        )

    def run_earnings_checks(self) -> list[AlertMessage]:
        """Run only earnings-related alert checks.

        Returns:
            List of triggered earnings alerts
        """
        if not self.alerts_config.earnings.enabled:
            return []

        return check_earnings_alerts(
            self.session, self.portfolio, self.alerts_config.earnings
        )

    def run_news_checks(self) -> list[AlertMessage]:
        """Run only news keyword alert checks.

        Returns:
            List of triggered news alerts
        """
        if not self.alerts_config.news.enabled:
            return []

        return check_news_keyword_alerts(
            self.session, self.portfolio, self.alerts_config.news
        )
