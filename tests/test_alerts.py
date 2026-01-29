"""Tests for alert priority classification."""

import pytest

from investment_monitor.alerts import (
    HIGH_PRIORITY_EXECUTIVES,
    HIGH_PRIORITY_KEYWORDS,
    LOW_PRIORITY_KEYWORDS,
    classify_priority,
    classify_priority_batch,
    get_alerts_by_priority,
)
from investment_monitor.models import AlertsConfig
from investment_monitor.notifications import AlertMessage, Priority


@pytest.fixture
def default_config() -> AlertsConfig:
    """Create default alerts configuration."""
    return AlertsConfig()


@pytest.fixture
def custom_config() -> AlertsConfig:
    """Create custom alerts configuration with different thresholds."""
    return AlertsConfig.model_validate({
        "price": {"daily_drop_pct": 5.0},
        "volume": {"multiplier": 3.0},
    })


class TestHighPriorityKeywords:
    """Tests for high-priority keyword detection."""

    @pytest.mark.parametrize("keyword", HIGH_PRIORITY_KEYWORDS)
    def test_high_priority_keyword_in_title(
        self, keyword: str, default_config: AlertsConfig
    ):
        """Alert with high-priority keyword in title gets HIGH priority."""
        alert = AlertMessage(
            title=f"Breaking: {keyword} reported",
            body="Details about the event.",
            ticker="AAPL",
            alert_type="news",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    @pytest.mark.parametrize("keyword", HIGH_PRIORITY_KEYWORDS)
    def test_high_priority_keyword_in_body(
        self, keyword: str, default_config: AlertsConfig
    ):
        """Alert with high-priority keyword in body gets HIGH priority."""
        alert = AlertMessage(
            title="Important news",
            body=f"The company is facing {keyword}.",
            ticker="AAPL",
            alert_type="news",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_high_priority_keyword_case_insensitive(self, default_config: AlertsConfig):
        """Keyword matching is case-insensitive."""
        alert = AlertMessage(
            title="SEC INVESTIGATION announced",
            body="Upper case keyword detected.",
            ticker="TSLA",
            alert_type="news",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH


class TestSeverePriceDrop:
    """Tests for severe price drop detection."""

    def test_severe_price_drop_high_priority(self, default_config: AlertsConfig):
        """Price drop > 2x threshold (>6% when threshold is 3%) is HIGH priority."""
        alert = AlertMessage(
            title="AAPL dropped significantly",
            body="Stock dropped 7% in trading today.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_severe_price_drop_custom_threshold(self, custom_config: AlertsConfig):
        """Uses custom threshold when configured (>10% when threshold is 5%)."""
        alert = AlertMessage(
            title="MSFT dropped sharply",
            body="Stock dropped 11% following news.",
            ticker="MSFT",
            alert_type="price",
        )
        assert classify_priority(alert, custom_config) == Priority.HIGH

    def test_normal_price_drop_medium_priority(self, default_config: AlertsConfig):
        """Normal price drop at threshold gets MEDIUM priority."""
        alert = AlertMessage(
            title="AAPL dropped",
            body="Stock dropped 4% today.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_negative_percentage_format(self, default_config: AlertsConfig):
        """Handles negative percentage format correctly."""
        alert = AlertMessage(
            title="Price decline",
            body="Stock showed -8% decline.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH


class TestExecutiveInsiderSale:
    """Tests for CEO/CFO insider sale detection."""

    def test_ceo_sale_high_priority(self, default_config: AlertsConfig):
        """CEO selling shares triggers HIGH priority."""
        alert = AlertMessage(
            title="Insider Trading Alert",
            body="CEO John Smith sold 100,000 shares.",
            ticker="AAPL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_cfo_sale_high_priority(self, default_config: AlertsConfig):
        """CFO selling shares triggers HIGH priority."""
        alert = AlertMessage(
            title="Insider Activity",
            body="CFO Jane Doe sale of 50,000 shares.",
            ticker="GOOGL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_chief_executive_title_detected(self, default_config: AlertsConfig):
        """Alternative executive title format is detected."""
        alert = AlertMessage(
            title="Executive Transaction",
            body="Chief Executive Officer sold shares worth $5M.",
            ticker="META",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_ceo_purchase_not_high_priority(self, default_config: AlertsConfig):
        """CEO buying shares is not HIGH priority (only sales)."""
        alert = AlertMessage(
            title="Insider Trading Alert",
            body="CEO John Smith purchased 100,000 shares.",
            ticker="AAPL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_non_executive_sale_medium_priority(self, default_config: AlertsConfig):
        """Non-executive insider sale is MEDIUM priority."""
        alert = AlertMessage(
            title="Insider Trading Alert",
            body="Director Bob Jones sold 10,000 shares.",
            ticker="AAPL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM


class TestLowPriority:
    """Tests for LOW priority classification."""

    @pytest.mark.parametrize("keyword", LOW_PRIORITY_KEYWORDS)
    def test_low_priority_keywords(
        self, keyword: str, default_config: AlertsConfig
    ):
        """Alert with low-priority keyword gets LOW priority."""
        alert = AlertMessage(
            title=f"Stock update: {keyword} movement",
            body="Nothing significant to report.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.LOW

    def test_minor_price_movement_low_priority(self, default_config: AlertsConfig):
        """Price movement < 50% of threshold is LOW priority."""
        alert = AlertMessage(
            title="AAPL price change",
            body="Stock moved 1% today.",  # < 1.5% (50% of 3%)
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.LOW


class TestMediumPriority:
    """Tests for MEDIUM priority (default) classification."""

    def test_normal_threshold_breach(self, default_config: AlertsConfig):
        """Normal threshold breach gets MEDIUM priority."""
        alert = AlertMessage(
            title="AAPL volume spike",
            body="Trading volume 3x normal levels.",
            ticker="AAPL",
            alert_type="volume",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_earnings_alert_medium_priority(self, default_config: AlertsConfig):
        """Upcoming earnings alert gets MEDIUM priority."""
        alert = AlertMessage(
            title="Earnings Reminder",
            body="AAPL reports earnings in 5 days.",
            ticker="AAPL",
            alert_type="earnings",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_standard_insider_transaction(self, default_config: AlertsConfig):
        """Standard insider transaction gets MEDIUM priority."""
        alert = AlertMessage(
            title="Insider Transaction",
            body="VP of Sales acquired 5,000 shares.",
            ticker="AAPL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM


class TestUserOverride:
    """Tests for user priority override functionality."""

    def test_user_override_to_high(self, default_config: AlertsConfig):
        """User can override to HIGH priority."""
        alert = AlertMessage(
            title="Minor update",
            body="Small change detected.",
            ticker="AAPL",
            alert_type="price",
        )
        result = classify_priority(alert, default_config, user_override=Priority.HIGH)
        assert result == Priority.HIGH

    def test_user_override_to_low(self, default_config: AlertsConfig):
        """User can override even HIGH-priority events to LOW."""
        alert = AlertMessage(
            title="SEC investigation announced",
            body="Company facing regulatory scrutiny.",
            ticker="AAPL",
            alert_type="news",
        )
        result = classify_priority(alert, default_config, user_override=Priority.LOW)
        assert result == Priority.LOW

    def test_none_override_uses_classification(self, default_config: AlertsConfig):
        """None override falls through to normal classification."""
        alert = AlertMessage(
            title="Fraud allegations",
            body="Serious allegations emerged.",
            ticker="AAPL",
            alert_type="news",
        )
        result = classify_priority(alert, default_config, user_override=None)
        assert result == Priority.HIGH


class TestBatchClassification:
    """Tests for batch priority classification."""

    def test_classify_priority_batch(self, default_config: AlertsConfig):
        """Batch classification works for multiple alerts."""
        alerts = [
            AlertMessage(
                title="SEC investigation",
                body="Details here.",
                ticker="AAPL",
                alert_type="news",
            ),
            AlertMessage(
                title="Normal update",
                body="Stock dropped 3%.",
                ticker="MSFT",
                alert_type="price",
            ),
            AlertMessage(
                title="Routine check",
                body="Minor movement observed.",
                ticker="GOOGL",
                alert_type="price",
            ),
        ]
        result = classify_priority_batch(alerts, default_config)
        # Returns list of (alert, priority) tuples in same order
        assert len(result) == 3
        assert result[0][0] == alerts[0]
        assert result[0][1] == Priority.HIGH
        assert result[1][0] == alerts[1]
        assert result[1][1] == Priority.MEDIUM
        assert result[2][0] == alerts[2]
        assert result[2][1] == Priority.LOW

    def test_batch_with_user_overrides(self, default_config: AlertsConfig):
        """Batch classification respects user overrides."""
        alerts = [
            AlertMessage(
                title="Alert One",
                body="Content one.",
                ticker="AAPL",
                alert_type="news",
            ),
            AlertMessage(
                title="Alert Two",
                body="Content two.",
                ticker="MSFT",
                alert_type="news",
            ),
        ]
        overrides = {"Alert One": Priority.HIGH}
        result = classify_priority_batch(alerts, default_config, overrides)
        # Returns list of (alert, priority) tuples
        assert len(result) == 2
        assert result[0][1] == Priority.HIGH
        assert result[1][1] == Priority.MEDIUM


class TestGetAlertsByPriority:
    """Tests for grouping alerts by priority."""

    def test_groups_alerts_by_priority(self, default_config: AlertsConfig):
        """Alerts are correctly grouped by priority level."""
        alerts = [
            AlertMessage(
                title="Fraud detected",
                body="Serious issue.",
                ticker="AAPL",
                alert_type="news",
            ),
            AlertMessage(
                title="Normal drop",
                body="Stock dropped 4%.",
                ticker="MSFT",
                alert_type="price",
            ),
            AlertMessage(
                title="Bankruptcy filing",
                body="Company filed.",
                ticker="TSLA",
                alert_type="news",
            ),
            AlertMessage(
                title="Minor change",
                body="Routine update observed.",
                ticker="GOOGL",
                alert_type="price",
            ),
        ]
        result = get_alerts_by_priority(alerts, default_config)

        assert len(result[Priority.HIGH]) == 2
        assert len(result[Priority.MEDIUM]) == 1
        assert len(result[Priority.LOW]) == 1

    def test_empty_alerts_list(self, default_config: AlertsConfig):
        """Empty list returns empty groups."""
        result = get_alerts_by_priority([], default_config)
        assert result[Priority.HIGH] == []
        assert result[Priority.MEDIUM] == []
        assert result[Priority.LOW] == []


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_multiple_high_priority_triggers(self, default_config: AlertsConfig):
        """Alert with multiple HIGH triggers still returns HIGH."""
        alert = AlertMessage(
            title="CEO sells amid fraud investigation",
            body="CEO sold 1M shares while SEC investigation ongoing. Stock dropped 15%.",
            ticker="AAPL",
            alert_type="insider",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_high_priority_overrides_low(self, default_config: AlertsConfig):
        """HIGH priority triggers take precedence over LOW keywords."""
        alert = AlertMessage(
            title="Routine fraud investigation",
            body="Scheduled review uncovered fraud.",
            ticker="AAPL",
            alert_type="news",
        )
        # Contains both "routine" (LOW) and "fraud" (HIGH)
        # HIGH should take precedence
        assert classify_priority(alert, default_config) == Priority.HIGH

    def test_no_percentage_in_price_alert(self, default_config: AlertsConfig):
        """Price alert without percentage defaults to MEDIUM."""
        alert = AlertMessage(
            title="AAPL price alert",
            body="Price moved significantly.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_percentage_in_non_price_alert(self, default_config: AlertsConfig):
        """Percentage in non-price alert is not used for severity."""
        alert = AlertMessage(
            title="Volume spike",
            body="Volume up 200% today.",
            ticker="AAPL",
            alert_type="volume",
        )
        # Should be MEDIUM, not affected by 200%
        assert classify_priority(alert, default_config) == Priority.MEDIUM

    def test_decimal_percentage(self, default_config: AlertsConfig):
        """Decimal percentages are parsed correctly."""
        alert = AlertMessage(
            title="Price drop",
            body="Stock dropped 6.5% today.",
            ticker="AAPL",
            alert_type="price",
        )
        assert classify_priority(alert, default_config) == Priority.HIGH
