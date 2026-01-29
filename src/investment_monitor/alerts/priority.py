"""Alert priority classification module.

This module provides logic for assigning priority levels to alerts based on
alert type, severity, and user configuration. Priority affects notification
routing - high-priority alerts are sent immediately while lower-priority
ones go into digests.
"""

import re
from typing import Any

from investment_monitor.models import AlertsConfig
from investment_monitor.notifications import AlertMessage, Priority


# High-priority keywords that trigger immediate notification
HIGH_PRIORITY_KEYWORDS = [
    "SEC investigation",
    "fraud",
    "lawsuit",
    "bankruptcy",
    "delisted",
    "halt",
    "criminal",
    "indictment",
    "restatement",
    "default",
]

# Low-priority keywords that indicate routine/minor events
LOW_PRIORITY_KEYWORDS = [
    "minor",
    "routine",
    "scheduled",
    "expected",
    "unchanged",
]

# Executive titles that trigger high priority for insider sales
HIGH_PRIORITY_EXECUTIVES = ["CEO", "CFO", "Chief Executive", "Chief Financial"]


def _check_keywords(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the text (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def _extract_percentage(text: str) -> float | None:
    """Extract percentage value from alert body if present.

    Looks for patterns like "dropped 5.2%", "rose 3%", etc.
    """
    # Match patterns like "5%", "5.2%", "-3.5%"
    pattern = r"[-+]?\d+\.?\d*%"
    matches = re.findall(pattern, text)
    if matches:
        # Return the first percentage found, stripped of % sign
        return float(matches[0].rstrip("%"))
    return None


def _is_severe_price_drop(alert: AlertMessage, config: AlertsConfig) -> bool:
    """Check if alert indicates a severe price drop (> 2x threshold)."""
    if alert.alert_type != "price":
        return False

    pct = _extract_percentage(alert.body)
    if pct is None:
        return False

    # Severe if drop exceeds 2x the configured threshold
    threshold = config.price.daily_drop_pct
    # Use absolute value since drops might be expressed as negative
    return abs(pct) > threshold * 2


def _is_executive_insider_sale(alert: AlertMessage) -> bool:
    """Check if alert is about CEO/CFO insider sale."""
    if alert.alert_type != "insider":
        return False

    text = f"{alert.title} {alert.body}".lower()
    # Check for sale/sold AND executive title
    is_sale = any(word in text for word in ["sale", "sold", "selling"])
    is_executive = any(title.lower() in text for title in HIGH_PRIORITY_EXECUTIVES)
    return is_sale and is_executive


def _has_high_priority_keywords(alert: AlertMessage) -> bool:
    """Check if alert contains high-priority keywords."""
    text = f"{alert.title} {alert.body}"
    return _check_keywords(text, HIGH_PRIORITY_KEYWORDS)


def _has_low_priority_keywords(alert: AlertMessage) -> bool:
    """Check if alert contains low-priority keywords."""
    text = f"{alert.title} {alert.body}"
    return _check_keywords(text, LOW_PRIORITY_KEYWORDS)


def _is_minor_price_movement(alert: AlertMessage, config: AlertsConfig) -> bool:
    """Check if alert is for a minor price movement (< 50% of threshold)."""
    if alert.alert_type != "price":
        return False

    pct = _extract_percentage(alert.body)
    if pct is None:
        return False

    threshold = config.price.daily_drop_pct
    return abs(pct) < threshold * 0.5


def classify_priority(
    alert: AlertMessage,
    config: AlertsConfig,
    user_override: Priority | None = None,
) -> Priority:
    """Assign priority based on alert type and severity.

    Priority levels:
        HIGH (immediate):
        - Price drop > 2x threshold (e.g., >6% if threshold is 3%)
        - CEO/CFO insider sale
        - Keywords: "SEC investigation", "lawsuit", "fraud", etc.

        MEDIUM (daily digest):
        - Normal threshold breaches
        - Insider transactions meeting criteria
        - Upcoming earnings

        LOW (weekly digest or log only):
        - Minor price movements
        - Low-relevance news
        - Routine/expected events

    Args:
        alert: The alert message to classify
        config: Alerts configuration with thresholds
        user_override: Optional user-specified priority override

    Returns:
        The classified priority level
    """
    # User override takes precedence
    if user_override is not None:
        return user_override

    # Check for HIGH priority conditions
    if _has_high_priority_keywords(alert):
        return Priority.HIGH

    if _is_severe_price_drop(alert, config):
        return Priority.HIGH

    if _is_executive_insider_sale(alert):
        return Priority.HIGH

    # Check for LOW priority conditions
    if _has_low_priority_keywords(alert):
        return Priority.LOW

    if _is_minor_price_movement(alert, config):
        return Priority.LOW

    # Default to MEDIUM for standard threshold breaches
    return Priority.MEDIUM


def classify_priority_batch(
    alerts: list[AlertMessage],
    config: AlertsConfig,
    user_overrides: dict[str, Priority] | None = None,
) -> list[tuple[AlertMessage, Priority]]:
    """Classify priorities for multiple alerts.

    Args:
        alerts: List of alert messages to classify
        config: Alerts configuration with thresholds
        user_overrides: Optional dict mapping alert titles to priority overrides

    Returns:
        List of (alert, priority) tuples in the same order as input
    """
    user_overrides = user_overrides or {}
    result = []
    for alert in alerts:
        override = user_overrides.get(alert.title)
        priority = classify_priority(alert, config, override)
        result.append((alert, priority))
    return result


def get_alerts_by_priority(
    alerts: list[AlertMessage],
    config: AlertsConfig,
    user_overrides: dict[str, Priority] | None = None,
) -> dict[Priority, list[AlertMessage]]:
    """Group alerts by their classified priority.

    Args:
        alerts: List of alert messages to classify
        config: Alerts configuration with thresholds
        user_overrides: Optional dict mapping alert titles to priority overrides

    Returns:
        Dict mapping priority levels to lists of alerts
    """
    classifications = classify_priority_batch(alerts, config, user_overrides)

    result: dict[Priority, list[AlertMessage]] = {
        Priority.HIGH: [],
        Priority.MEDIUM: [],
        Priority.LOW: [],
    }

    for alert, priority in classifications:
        result[priority].append(alert)

    return result
