"""Alert engine and rule-based alert system.

This module provides:
- AlertEngine for running all alert checks
- Alert rules for detecting notable market conditions
- Alert priority classification based on alert type, severity, and user configuration
- Alert deduplication to prevent sending duplicate notifications

Alert Types:
    - Price alerts: Daily/weekly drops, rises, below cost basis
    - Volume alerts: Unusual trading volume spikes
    - Insider alerts: Significant insider trades, executive activity, cluster trades
    - Earnings alerts: Upcoming earnings announcements
    - News alerts: Headlines matching configured keywords

Priority affects notification routing:
- HIGH: Send immediately via all channels
- MEDIUM: Include in daily digest
- LOW: Include in weekly digest or log only

Example usage:
    from investment_monitor.alerts import AlertEngine
    from investment_monitor.models import AlertsConfig, Portfolio
    from investment_monitor.storage import get_session

    portfolio = Portfolio.from_yaml("portfolio.yaml")
    config = AlertsConfig.from_yaml("alerts.yaml")

    with get_session() as session:
        engine = AlertEngine(session, portfolio, config)
        alerts = engine.run_all_checks()

        for alert in alerts:
            print(alert.format_full())
"""

from .dedup import (
    DEFAULT_DEDUP_WINDOW,
    DEDUP_WINDOWS,
    AlertDeduplicator,
)
from .engine import AlertEngine
from .priority import (
    HIGH_PRIORITY_EXECUTIVES,
    HIGH_PRIORITY_KEYWORDS,
    LOW_PRIORITY_KEYWORDS,
    classify_priority,
    classify_priority_batch,
    get_alerts_by_priority,
)
from .rules import (
    check_earnings_alerts,
    check_insider_alerts,
    check_news_keyword_alerts,
    check_price_alerts,
    check_volume_alerts,
)

__all__ = [
    # Engine
    "AlertEngine",
    # Rules
    "check_earnings_alerts",
    "check_insider_alerts",
    "check_news_keyword_alerts",
    "check_price_alerts",
    "check_volume_alerts",
    # Deduplication
    "AlertDeduplicator",
    "DEDUP_WINDOWS",
    "DEFAULT_DEDUP_WINDOW",
    # Priority classification
    "HIGH_PRIORITY_EXECUTIVES",
    "HIGH_PRIORITY_KEYWORDS",
    "LOW_PRIORITY_KEYWORDS",
    "classify_priority",
    "classify_priority_batch",
    "get_alerts_by_priority",
]
