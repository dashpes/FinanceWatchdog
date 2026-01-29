"""Tests for alert priority classification and deduplication."""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from investment_monitor.alerts import (
    DEFAULT_DEDUP_WINDOW,
    DEDUP_WINDOWS,
    AlertDeduplicator,
    HIGH_PRIORITY_EXECUTIVES,
    HIGH_PRIORITY_KEYWORDS,
    LOW_PRIORITY_KEYWORDS,
    classify_priority,
    classify_priority_batch,
    get_alerts_by_priority,
)
from investment_monitor.models import AlertsConfig
from investment_monitor.notifications import AlertMessage, Priority
from investment_monitor.storage import (
    AlertSent,
    get_session,
    init_db,
)


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


# ============================================================================
# Alert Deduplication Tests
# ============================================================================


@pytest.fixture
def db_session():
    """Create a temporary database for testing deduplication."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


@pytest.fixture
def deduplicator(db_session):
    """Create an AlertDeduplicator with a test database."""
    return AlertDeduplicator(db_session)


@pytest.fixture
def sample_alert():
    """Create a sample alert for testing."""
    return AlertMessage(
        title="AAPL dropped 5%",
        body="Apple stock dropped 5% today.",
        ticker="AAPL",
        alert_type="price",
        priority=Priority.MEDIUM,
    )


class TestDedupWindowsConfig:
    """Tests for deduplication window configuration."""

    def test_price_alerts_have_24_hour_window(self):
        """Test price-related alerts have 24-hour dedup window."""
        assert DEDUP_WINDOWS["price_drop"] == timedelta(hours=24)
        assert DEDUP_WINDOWS["price_rise"] == timedelta(hours=24)
        assert DEDUP_WINDOWS["price"] == timedelta(hours=24)

    def test_volume_alerts_have_12_hour_window(self):
        """Test volume alerts have 12-hour dedup window."""
        assert DEDUP_WINDOWS["volume_spike"] == timedelta(hours=12)
        assert DEDUP_WINDOWS["volume"] == timedelta(hours=12)

    def test_insider_alerts_have_7_day_window(self):
        """Test insider alerts have 7-day dedup window."""
        assert DEDUP_WINDOWS["insider_transaction"] == timedelta(days=7)
        assert DEDUP_WINDOWS["insider"] == timedelta(days=7)

    def test_earnings_alerts_have_3_day_window(self):
        """Test earnings alerts have 3-day dedup window."""
        assert DEDUP_WINDOWS["earnings_upcoming"] == timedelta(days=3)
        assert DEDUP_WINDOWS["earnings"] == timedelta(days=3)

    def test_news_alerts_have_1_day_window(self):
        """Test news alerts have 1-day dedup window."""
        assert DEDUP_WINDOWS["news_keyword"] == timedelta(days=1)
        assert DEDUP_WINDOWS["news"] == timedelta(days=1)

    def test_other_alert_types(self):
        """Test other alert types have appropriate windows."""
        assert DEDUP_WINDOWS["dividend"] == timedelta(days=7)
        assert DEDUP_WINDOWS["filing"] == timedelta(days=7)
        assert DEDUP_WINDOWS["analyst"] == timedelta(days=1)
        assert DEDUP_WINDOWS["system"] == timedelta(hours=1)

    def test_default_dedup_window(self):
        """Test default dedup window for unknown types."""
        assert DEFAULT_DEDUP_WINDOW == timedelta(hours=24)


class TestGenerateDedupKey:
    """Tests for dedup key generation."""

    def test_generates_key_with_ticker(self, deduplicator):
        """Test key includes alert type and ticker."""
        alert = AlertMessage(
            title="AAPL dropped 5%",
            body="Apple stock dropped.",
            ticker="AAPL",
            alert_type="price",
        )
        key = deduplicator.generate_dedup_key(alert)

        assert "price" in key
        assert "AAPL" in key
        assert key.count(":") == 2  # Format: type:ticker:hash

    def test_generates_key_without_ticker(self, deduplicator):
        """Test key works without ticker."""
        alert = AlertMessage(
            title="System maintenance",
            body="Scheduled downtime.",
            alert_type="system",
        )
        key = deduplicator.generate_dedup_key(alert)

        assert "system" in key
        assert "::" in key  # Empty ticker section

    def test_different_titles_generate_different_keys(self, deduplicator):
        """Test that different titles produce different keys."""
        alert1 = AlertMessage(
            title="AAPL dropped 5%",
            body="Body text",
            ticker="AAPL",
            alert_type="price",
        )
        alert2 = AlertMessage(
            title="AAPL dropped 3%",
            body="Body text",
            ticker="AAPL",
            alert_type="price",
        )

        key1 = deduplicator.generate_dedup_key(alert1)
        key2 = deduplicator.generate_dedup_key(alert2)

        assert key1 != key2

    def test_same_alert_generates_same_key(self, deduplicator, sample_alert):
        """Test that the same alert always generates the same key."""
        key1 = deduplicator.generate_dedup_key(sample_alert)
        key2 = deduplicator.generate_dedup_key(sample_alert)

        assert key1 == key2

    def test_key_format(self, deduplicator):
        """Test key follows expected format."""
        alert = AlertMessage(
            title="Test alert",
            body="Test body",
            ticker="TEST",
            alert_type="price",
        )
        key = deduplicator.generate_dedup_key(alert)

        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "price"
        assert parts[1] == "TEST"
        assert len(parts[2]) == 8  # SHA256 hash truncated to 8 chars


class TestGetDedupWindow:
    """Tests for getting dedup windows by alert type."""

    def test_known_alert_types(self, deduplicator):
        """Test known alert types return correct windows."""
        assert deduplicator.get_dedup_window("price") == timedelta(hours=24)
        assert deduplicator.get_dedup_window("volume") == timedelta(hours=12)
        assert deduplicator.get_dedup_window("insider") == timedelta(days=7)

    def test_unknown_alert_type_returns_default(self, deduplicator):
        """Test unknown alert types return default window."""
        window = deduplicator.get_dedup_window("unknown_type")
        assert window == DEFAULT_DEDUP_WINDOW


class TestIsDuplicate:
    """Tests for duplicate detection."""

    def test_new_alert_is_not_duplicate(self, deduplicator, sample_alert):
        """Test that a new alert is not considered a duplicate."""
        assert deduplicator.is_duplicate(sample_alert) is False

    def test_recently_sent_alert_is_duplicate(self, deduplicator, db_session, sample_alert):
        """Test that a recently sent alert is considered a duplicate."""
        # Mark alert as sent
        deduplicator.mark_sent(sample_alert, "console")
        db_session.commit()

        # Should now be detected as duplicate
        assert deduplicator.is_duplicate(sample_alert) is True

    def test_different_alert_not_duplicate(self, deduplicator, db_session):
        """Test that different alerts are not duplicates of each other."""
        alert1 = AlertMessage(
            title="AAPL dropped 5%",
            body="Apple stock dropped.",
            ticker="AAPL",
            alert_type="price",
        )
        alert2 = AlertMessage(
            title="MSFT rose 3%",
            body="Microsoft stock rose.",
            ticker="MSFT",
            alert_type="price",
        )

        deduplicator.mark_sent(alert1, "console")
        db_session.commit()

        assert deduplicator.is_duplicate(alert2) is False

    def test_same_ticker_different_type_not_duplicate(self, deduplicator, db_session):
        """Test that same ticker with different alert type is not a duplicate."""
        alert1 = AlertMessage(
            title="AAPL price alert",
            body="Price changed.",
            ticker="AAPL",
            alert_type="price",
        )
        alert2 = AlertMessage(
            title="AAPL volume alert",
            body="Volume spiked.",
            ticker="AAPL",
            alert_type="volume",
        )

        deduplicator.mark_sent(alert1, "console")
        db_session.commit()

        assert deduplicator.is_duplicate(alert2) is False


class TestMarkSent:
    """Tests for marking alerts as sent."""

    def test_mark_sent_returns_id(self, deduplicator, sample_alert):
        """Test that mark_sent returns the alert record ID."""
        alert_id = deduplicator.mark_sent(sample_alert, "console")

        assert isinstance(alert_id, int)
        assert alert_id > 0

    def test_mark_sent_creates_record(self, deduplicator, db_session, sample_alert):
        """Test that mark_sent creates a database record."""
        deduplicator.mark_sent(sample_alert, "console")
        db_session.commit()

        # Query directly to verify
        record = db_session.query(AlertSent).first()
        assert record is not None
        assert record.alert_type == "price"
        assert record.ticker == "AAPL"
        assert record.channel == "console"
        assert record.dedup_key is not None

    def test_mark_sent_stores_message_body(self, deduplicator, db_session, sample_alert):
        """Test that mark_sent stores the message body."""
        deduplicator.mark_sent(sample_alert, "console")
        db_session.commit()

        record = db_session.query(AlertSent).first()
        assert record.message == sample_alert.body

    def test_mark_sent_stores_priority(self, deduplicator, db_session, sample_alert):
        """Test that mark_sent stores the priority."""
        deduplicator.mark_sent(sample_alert, "console")
        db_session.commit()

        record = db_session.query(AlertSent).first()
        assert record.priority == sample_alert.priority.value

    def test_mark_sent_stores_channel(self, deduplicator, db_session, sample_alert):
        """Test that mark_sent stores different channels correctly."""
        deduplicator.mark_sent(sample_alert, "slack")
        db_session.commit()

        record = db_session.query(AlertSent).first()
        assert record.channel == "slack"

    def test_mark_sent_without_ticker(self, deduplicator, db_session):
        """Test marking sent for alert without ticker."""
        alert = AlertMessage(
            title="System alert",
            body="System message.",
            alert_type="system",
        )
        deduplicator.mark_sent(alert, "console")
        db_session.commit()

        record = db_session.query(AlertSent).first()
        assert record.ticker == ""


class TestFilterDuplicates:
    """Tests for filtering duplicate alerts."""

    def test_filter_empty_list(self, deduplicator):
        """Test filtering an empty list returns empty list."""
        result = deduplicator.filter_duplicates([])
        assert result == []

    def test_filter_all_unique(self, deduplicator):
        """Test filtering all unique alerts returns all."""
        alerts = [
            AlertMessage(
                title="AAPL alert",
                body="Body",
                ticker="AAPL",
                alert_type="price",
            ),
            AlertMessage(
                title="MSFT alert",
                body="Body",
                ticker="MSFT",
                alert_type="price",
            ),
            AlertMessage(
                title="GOOGL alert",
                body="Body",
                ticker="GOOGL",
                alert_type="price",
            ),
        ]

        result = deduplicator.filter_duplicates(alerts)
        assert len(result) == 3

    def test_filter_removes_duplicates_in_batch(self, deduplicator):
        """Test that duplicate alerts in the same batch are filtered."""
        alert = AlertMessage(
            title="Same alert",
            body="Body",
            ticker="AAPL",
            alert_type="price",
        )
        # Same alert appears multiple times
        alerts = [alert, alert, alert]

        result = deduplicator.filter_duplicates(alerts)
        assert len(result) == 1

    def test_filter_removes_previously_sent(self, deduplicator, db_session):
        """Test that previously sent alerts are filtered."""
        alert1 = AlertMessage(
            title="Old alert",
            body="Body",
            ticker="AAPL",
            alert_type="price",
        )
        alert2 = AlertMessage(
            title="New alert",
            body="Body",
            ticker="MSFT",
            alert_type="price",
        )

        # Mark first alert as sent
        deduplicator.mark_sent(alert1, "console")
        db_session.commit()

        # Filter both alerts
        result = deduplicator.filter_duplicates([alert1, alert2])

        assert len(result) == 1
        assert result[0].ticker == "MSFT"

    def test_filter_preserves_order(self, deduplicator):
        """Test that filtering preserves order of unique alerts."""
        alerts = [
            AlertMessage(title="First", body="B", ticker="A", alert_type="price"),
            AlertMessage(title="Second", body="B", ticker="B", alert_type="price"),
            AlertMessage(title="Third", body="B", ticker="C", alert_type="price"),
        ]

        result = deduplicator.filter_duplicates(alerts)

        assert [a.title for a in result] == ["First", "Second", "Third"]

    def test_filter_with_mixed_duplicates(self, deduplicator, db_session):
        """Test filtering with mix of old and new duplicates."""
        old_alert = AlertMessage(
            title="Old",
            body="Body",
            ticker="OLD",
            alert_type="price",
        )
        new_alert = AlertMessage(
            title="New",
            body="Body",
            ticker="NEW",
            alert_type="price",
        )
        same_new = AlertMessage(
            title="New",  # Same title as new_alert
            body="Body",
            ticker="NEW",
            alert_type="price",
        )

        # Mark old alert as sent
        deduplicator.mark_sent(old_alert, "console")
        db_session.commit()

        # Filter: old (sent), new, same_new (batch duplicate)
        result = deduplicator.filter_duplicates([old_alert, new_alert, same_new])

        assert len(result) == 1
        assert result[0].title == "New"


class TestDeduplicatorIntegration:
    """Integration tests for the complete deduplication flow."""

    def test_full_workflow(self, deduplicator, db_session):
        """Test complete deduplication workflow."""
        # Generate alerts
        alerts = [
            AlertMessage(
                title="AAPL dropped 5%",
                body="Apple dropped.",
                ticker="AAPL",
                alert_type="price",
            ),
            AlertMessage(
                title="MSFT rose 3%",
                body="Microsoft rose.",
                ticker="MSFT",
                alert_type="price",
            ),
        ]

        # Filter - all should pass
        filtered = deduplicator.filter_duplicates(alerts)
        assert len(filtered) == 2

        # Mark as sent
        for alert in filtered:
            deduplicator.mark_sent(alert, "console")
        db_session.commit()

        # Try to filter again - should all be blocked
        filtered_again = deduplicator.filter_duplicates(alerts)
        assert len(filtered_again) == 0

        # New alert should still pass
        new_alert = AlertMessage(
            title="GOOGL volume spike",
            body="Google volume.",
            ticker="GOOGL",
            alert_type="volume",
        )
        filtered_new = deduplicator.filter_duplicates([new_alert] + alerts)
        assert len(filtered_new) == 1
        assert filtered_new[0].ticker == "GOOGL"

    def test_multiple_channels(self, deduplicator, db_session):
        """Test that alerts sent to different channels are tracked."""
        alert = AlertMessage(
            title="Multi-channel alert",
            body="Body",
            ticker="AAPL",
            alert_type="price",
        )

        # Send to multiple channels
        deduplicator.mark_sent(alert, "console")
        deduplicator.mark_sent(alert, "slack")
        deduplicator.mark_sent(alert, "email")
        db_session.commit()

        # Should still be considered duplicate
        assert deduplicator.is_duplicate(alert) is True

        # All three records should exist
        records = db_session.query(AlertSent).all()
        assert len(records) == 3
        channels = {r.channel for r in records}
        assert channels == {"console", "slack", "email"}


# ============================================================================
# Alert Rules Tests
# ============================================================================

from datetime import date, datetime
from decimal import Decimal

from investment_monitor.alerts import (
    AlertEngine,
    check_earnings_alerts,
    check_insider_alerts,
    check_news_keyword_alerts,
    check_price_alerts,
    check_volume_alerts,
)
from investment_monitor.models import (
    EarningsAlertSettings,
    Holding,
    InsiderAlertSettings,
    NewsAlertSettings,
    Portfolio,
    PriceAlertSettings,
    VolumeAlertSettings,
)
from investment_monitor.storage import (
    EarningsDate,
    InsiderTransaction,
    NewsItem,
    Price,
    save_earnings_date,
    save_insider_transaction,
    save_news_item,
    save_price,
)


@pytest.fixture
def portfolio():
    """Create a test portfolio."""
    return Portfolio(
        holdings=[
            Holding(ticker="AAPL", shares=Decimal("100"), cost_basis=Decimal("150.00")),
            Holding(ticker="MSFT", shares=Decimal("50"), cost_basis=Decimal("350.00")),
        ],
        watchlist=[],
    )


class TestPriceAlertRules:
    """Tests for price-based alert rules."""

    def test_daily_drop_alert(self, db_session, portfolio):
        """Test that daily drop alert triggers when threshold is exceeded."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Create prices showing a 6% drop
        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=100.0))
        save_price(db_session, Price(ticker="AAPL", date=today, close=94.0))

        config = PriceAlertSettings(daily_drop_pct=3.0)
        alerts = check_price_alerts(db_session, portfolio, config)

        aapl_alerts = [a for a in alerts if a.ticker == "AAPL"]
        assert len(aapl_alerts) >= 1
        assert any("dropped" in a.title.lower() for a in aapl_alerts)
        assert any(a.alert_type == "price_daily_drop" for a in aapl_alerts)

    def test_daily_rise_alert(self, db_session, portfolio):
        """Test that daily rise alert triggers when threshold is exceeded."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # Create prices showing a 7% rise
        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=100.0))
        save_price(db_session, Price(ticker="AAPL", date=today, close=107.0))

        config = PriceAlertSettings(daily_rise_pct=5.0)
        alerts = check_price_alerts(db_session, portfolio, config)

        aapl_alerts = [a for a in alerts if a.ticker == "AAPL"]
        assert len(aapl_alerts) >= 1
        assert any("rose" in a.title.lower() for a in aapl_alerts)

    def test_weekly_drop_alert(self, db_session, portfolio):
        """Test that weekly drop alert triggers."""
        today = date.today()

        # Create 7 days of prices showing 10% weekly drop
        for i in range(7):
            d = today - timedelta(days=i)
            # Latest price 100, price 6 days ago was ~111 (10% drop)
            close = 100.0 if i == 0 else 100.0 + (i * 1.85)
            save_price(db_session, Price(ticker="AAPL", date=d, close=close))

        config = PriceAlertSettings(weekly_drop_pct=7.0)
        alerts = check_price_alerts(db_session, portfolio, config)

        weekly_alerts = [a for a in alerts if "week" in a.title.lower()]
        assert len(weekly_alerts) >= 1

    def test_below_cost_basis_alert(self, db_session, portfolio):
        """Test alert when price falls below cost basis."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # AAPL cost basis is 150, price is 140
        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=145.0))
        save_price(db_session, Price(ticker="AAPL", date=today, close=140.0))

        config = PriceAlertSettings(below_cost_basis=True, daily_drop_pct=10.0)
        alerts = check_price_alerts(db_session, portfolio, config)

        below_cost_alerts = [a for a in alerts if "cost basis" in a.title.lower()]
        assert len(below_cost_alerts) == 1
        assert below_cost_alerts[0].ticker == "AAPL"

    def test_no_alert_when_threshold_not_exceeded(self, db_session, portfolio):
        """Test that no alert triggers when changes are below threshold."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        # 1% drop, below 3% threshold
        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=100.0))
        save_price(db_session, Price(ticker="AAPL", date=today, close=99.0))

        config = PriceAlertSettings(daily_drop_pct=3.0, below_cost_basis=False)
        alerts = check_price_alerts(db_session, portfolio, config)

        aapl_drop_alerts = [a for a in alerts if a.ticker == "AAPL" and "drop" in a.title.lower()]
        assert len(aapl_drop_alerts) == 0

    def test_handles_missing_price_data(self, db_session, portfolio):
        """Test graceful handling when no price data exists."""
        config = PriceAlertSettings()
        alerts = check_price_alerts(db_session, portfolio, config)
        assert isinstance(alerts, list)


class TestVolumeAlertRules:
    """Tests for volume-based alert rules."""

    def test_volume_spike_alert(self, db_session, portfolio):
        """Test that volume spike alert triggers."""
        today = date.today()

        # Create historical prices with normal volume
        for i in range(25):
            d = today - timedelta(days=i)
            # Today: 10M volume, historical: 2M average
            volume = 10_000_000 if i == 0 else 2_000_000
            save_price(db_session, Price(ticker="AAPL", date=d, close=150.0, volume=volume))

        config = VolumeAlertSettings(lookback_days=20, multiplier=2.5)
        alerts = check_volume_alerts(db_session, portfolio, config)

        volume_alerts = [a for a in alerts if a.ticker == "AAPL"]
        assert len(volume_alerts) >= 1
        assert any("volume" in a.title.lower() for a in volume_alerts)

    def test_no_volume_alert_when_normal(self, db_session, portfolio):
        """Test no alert when volume is normal."""
        today = date.today()

        # All normal volume
        for i in range(25):
            d = today - timedelta(days=i)
            save_price(db_session, Price(ticker="AAPL", date=d, close=150.0, volume=2_000_000))

        config = VolumeAlertSettings(lookback_days=20, multiplier=2.5)
        alerts = check_volume_alerts(db_session, portfolio, config)

        volume_alerts = [a for a in alerts if a.alert_type == "volume_spike"]
        assert len(volume_alerts) == 0

    def test_handles_missing_volume_data(self, db_session, portfolio):
        """Test graceful handling when volume data is missing."""
        today = date.today()

        for i in range(5):
            d = today - timedelta(days=i)
            save_price(db_session, Price(ticker="AAPL", date=d, close=150.0, volume=None))

        config = VolumeAlertSettings()
        alerts = check_volume_alerts(db_session, portfolio, config)
        assert isinstance(alerts, list)


class TestInsiderAlertRules:
    """Tests for insider trading alert rules."""

    def test_significant_buy_alert(self, db_session, portfolio):
        """Test alert for significant insider buy."""
        today = date.today()

        txn = InsiderTransaction(
            ticker="AAPL",
            filing_date=today,
            trade_date=today,
            owner_name="John Smith",
            owner_title="Director",
            transaction_type="P",
            shares=10000,
            price_per_share=150.0,
            total_value=1_500_000.0,
            sec_url="https://sec.gov/filing/rule_test_123",
        )
        save_insider_transaction(db_session, txn)

        config = InsiderAlertSettings(min_buy_value=100_000)
        alerts = check_insider_alerts(db_session, portfolio, config)

        buy_alerts = [a for a in alerts if "buy" in a.alert_type.lower()]
        assert len(buy_alerts) >= 1
        assert any("John Smith" in a.title for a in buy_alerts)

    def test_significant_sell_alert(self, db_session, portfolio):
        """Test alert for significant insider sell."""
        today = date.today()

        txn = InsiderTransaction(
            ticker="AAPL",
            filing_date=today,
            trade_date=today,
            owner_name="Jane Doe",
            owner_title="VP",
            transaction_type="S",
            shares=5000,
            price_per_share=150.0,
            total_value=750_000.0,
            sec_url="https://sec.gov/filing/rule_test_456",
        )
        save_insider_transaction(db_session, txn)

        config = InsiderAlertSettings(min_sell_value=500_000)
        alerts = check_insider_alerts(db_session, portfolio, config)

        sell_alerts = [a for a in alerts if "sell" in a.alert_type.lower()]
        assert len(sell_alerts) >= 1
        assert any("Jane Doe" in a.title for a in sell_alerts)

    def test_ceo_cfo_alert_any_size(self, db_session, portfolio):
        """Test alert for CEO/CFO transactions at any size."""
        today = date.today()

        # Small CEO purchase
        txn = InsiderTransaction(
            ticker="AAPL",
            filing_date=today,
            trade_date=today,
            owner_name="Tim Cook",
            owner_title="CEO",
            transaction_type="P",
            shares=100,
            price_per_share=150.0,
            total_value=15_000.0,  # Below normal threshold
            sec_url="https://sec.gov/filing/rule_test_789",
        )
        save_insider_transaction(db_session, txn)

        config = InsiderAlertSettings(
            min_buy_value=100_000,
            alert_ceo_cfo_any=True,
        )
        alerts = check_insider_alerts(db_session, portfolio, config)

        exec_alerts = [a for a in alerts if "executive" in a.alert_type.lower() or "Tim Cook" in a.title]
        assert len(exec_alerts) >= 1

    def test_cluster_buying_alert(self, db_session, portfolio):
        """Test alert for cluster buying (multiple insiders)."""
        today = date.today()

        for i, name in enumerate(["Insider A", "Insider B", "Insider C"]):
            txn = InsiderTransaction(
                ticker="AAPL",
                filing_date=today,
                trade_date=today - timedelta(days=i),
                owner_name=name,
                owner_title="Director",
                transaction_type="P",
                shares=1000,
                price_per_share=150.0,
                total_value=150_000.0,
                sec_url=f"https://sec.gov/filing/cluster_rule_{i}",
            )
            save_insider_transaction(db_session, txn)

        config = InsiderAlertSettings(cluster_threshold=3, cluster_days=7)
        alerts = check_insider_alerts(db_session, portfolio, config)

        cluster_alerts = [a for a in alerts if "cluster" in a.alert_type.lower()]
        assert len(cluster_alerts) >= 1
        assert any("3 insiders" in a.title for a in cluster_alerts)

    def test_no_alert_below_threshold(self, db_session, portfolio):
        """Test no alert when transaction is below threshold."""
        today = date.today()

        txn = InsiderTransaction(
            ticker="AAPL",
            filing_date=today,
            trade_date=today,
            owner_name="Small Timer",
            owner_title="Director",
            transaction_type="P",
            shares=100,
            price_per_share=150.0,
            total_value=15_000.0,
            sec_url="https://sec.gov/filing/small_rule",
        )
        save_insider_transaction(db_session, txn)

        config = InsiderAlertSettings(min_buy_value=100_000, alert_ceo_cfo_any=False)
        alerts = check_insider_alerts(db_session, portfolio, config)

        buy_alerts = [a for a in alerts if a.alert_type == "insider_buy"]
        assert len(buy_alerts) == 0


class TestEarningsAlertRules:
    """Tests for earnings-related alert rules."""

    def test_earnings_tomorrow_alert(self, db_session, portfolio):
        """Test high-priority alert for earnings tomorrow."""
        tomorrow = date.today() + timedelta(days=1)

        earnings = EarningsDate(ticker="AAPL", earnings_date=tomorrow, confirmed=True)
        save_earnings_date(db_session, earnings)

        config = EarningsAlertSettings(lookahead_days=7)
        alerts = check_earnings_alerts(db_session, portfolio, config)

        assert len(alerts) >= 1
        aapl_alerts = [a for a in alerts if a.ticker == "AAPL"]
        assert len(aapl_alerts) >= 1
        assert any(a.priority == Priority.HIGH for a in aapl_alerts)

    def test_earnings_in_week_alert(self, db_session, portfolio):
        """Test medium-priority alert for earnings in a few days."""
        earnings_date = date.today() + timedelta(days=3)

        earnings = EarningsDate(ticker="MSFT", earnings_date=earnings_date, confirmed=False)
        save_earnings_date(db_session, earnings)

        config = EarningsAlertSettings(lookahead_days=7)
        alerts = check_earnings_alerts(db_session, portfolio, config)

        msft_alerts = [a for a in alerts if a.ticker == "MSFT"]
        assert len(msft_alerts) >= 1
        assert any(a.priority == Priority.MEDIUM for a in msft_alerts)

    def test_no_alert_beyond_lookahead(self, db_session, portfolio):
        """Test no alert for earnings beyond lookahead window."""
        far_future = date.today() + timedelta(days=30)

        earnings = EarningsDate(ticker="AAPL", earnings_date=far_future, confirmed=True)
        save_earnings_date(db_session, earnings)

        config = EarningsAlertSettings(lookahead_days=7)
        alerts = check_earnings_alerts(db_session, portfolio, config)

        aapl_alerts = [a for a in alerts if a.ticker == "AAPL"]
        assert len(aapl_alerts) == 0


class TestNewsAlertRules:
    """Tests for news keyword alert rules."""

    def test_keyword_match_alert(self, db_session, portfolio):
        """Test alert when headline matches keyword."""
        news = NewsItem(
            ticker="AAPL",
            headline="Apple faces SEC investigation over accounting practices",
            source="Reuters",
            url="https://example.com/news/rule_1",
            published_at=datetime.now(),
            relevance_score=8.0,
        )
        save_news_item(db_session, news)

        config = NewsAlertSettings(keywords=["SEC", "investigation", "lawsuit"])
        alerts = check_news_keyword_alerts(db_session, portfolio, config)

        assert len(alerts) >= 1
        assert any("SEC" in a.title or "investigation" in a.title for a in alerts)
        assert any(a.priority == Priority.HIGH for a in alerts)

    def test_multiple_keyword_match(self, db_session, portfolio):
        """Test alert captures multiple matching keywords."""
        news = NewsItem(
            ticker="AAPL",
            headline="Apple announces merger and acquisition of AI startup",
            source="Bloomberg",
            url="https://example.com/news/rule_2",
            published_at=datetime.now(),
            relevance_score=7.0,
        )
        save_news_item(db_session, news)

        config = NewsAlertSettings(keywords=["merger", "acquisition", "buyback"])
        alerts = check_news_keyword_alerts(db_session, portfolio, config)

        assert len(alerts) >= 1
        assert any("merger" in a.body.lower() for a in alerts)

    def test_no_alert_low_relevance(self, db_session, portfolio):
        """Test no alert when relevance score is too low."""
        news = NewsItem(
            ticker="AAPL",
            headline="Apple releases new product with minor SEC filing",
            source="TechNews",
            url="https://example.com/news/rule_3",
            published_at=datetime.now(),
            relevance_score=2.0,
        )
        save_news_item(db_session, news)

        config = NewsAlertSettings(keywords=["SEC"], min_relevance_score=5.0)
        alerts = check_news_keyword_alerts(db_session, portfolio, config)

        assert len(alerts) == 0

    def test_no_alert_no_keyword_match(self, db_session, portfolio):
        """Test no alert when no keywords match."""
        news = NewsItem(
            ticker="AAPL",
            headline="Apple reports strong quarterly results",
            source="CNBC",
            url="https://example.com/news/rule_4",
            published_at=datetime.now(),
        )
        save_news_item(db_session, news)

        config = NewsAlertSettings(keywords=["lawsuit", "bankruptcy", "fraud"])
        alerts = check_news_keyword_alerts(db_session, portfolio, config)

        assert len(alerts) == 0


class TestAlertEngineRules:
    """Tests for the AlertEngine class."""

    def test_run_all_checks(self, db_session, portfolio, default_config):
        """Test running all checks at once."""
        today = date.today()
        yesterday = today - timedelta(days=1)

        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=100.0, volume=1_000_000))
        save_price(db_session, Price(ticker="AAPL", date=today, close=92.0, volume=5_000_000))

        engine = AlertEngine(db_session, portfolio, default_config)
        alerts = engine.run_all_checks()

        assert isinstance(alerts, list)

    def test_disabled_checks_skipped(self, db_session, portfolio):
        """Test that disabled checks are skipped."""
        config = AlertsConfig(
            price=PriceAlertSettings(enabled=False),
            volume=VolumeAlertSettings(enabled=False),
            insider=InsiderAlertSettings(enabled=False),
            earnings=EarningsAlertSettings(enabled=False),
            news=NewsAlertSettings(enabled=False),
        )

        engine = AlertEngine(db_session, portfolio, config)
        alerts = engine.run_all_checks()

        assert len(alerts) == 0

    def test_individual_check_methods(self, db_session, portfolio, default_config):
        """Test individual check methods."""
        engine = AlertEngine(db_session, portfolio, default_config)

        assert isinstance(engine.run_price_checks(), list)
        assert isinstance(engine.run_volume_checks(), list)
        assert isinstance(engine.run_insider_checks(), list)
        assert isinstance(engine.run_earnings_checks(), list)
        assert isinstance(engine.run_news_checks(), list)

    def test_alerts_sorted_by_priority(self, db_session, portfolio):
        """Test that alerts are sorted by priority (HIGH first)."""
        today = date.today()
        tomorrow = today + timedelta(days=1)
        yesterday = today - timedelta(days=1)

        save_price(db_session, Price(ticker="AAPL", date=yesterday, close=100.0))
        save_price(db_session, Price(ticker="AAPL", date=today, close=90.0))

        save_earnings_date(db_session, EarningsDate(ticker="MSFT", earnings_date=tomorrow, confirmed=True))

        config = AlertsConfig(
            price=PriceAlertSettings(enabled=True, daily_drop_pct=3.0),
            earnings=EarningsAlertSettings(enabled=True, lookahead_days=7),
            volume=VolumeAlertSettings(enabled=False),
            insider=InsiderAlertSettings(enabled=False),
            news=NewsAlertSettings(enabled=False),
        )

        engine = AlertEngine(db_session, portfolio, config)
        alerts = engine.run_all_checks()

        if len(alerts) >= 2:
            high_alerts = [a for a in alerts if a.priority == Priority.HIGH]
            if high_alerts:
                first_high_idx = alerts.index(high_alerts[0])
                low_alerts = [a for a in alerts if a.priority == Priority.LOW]
                if low_alerts:
                    first_low_idx = alerts.index(low_alerts[0])
                    assert first_high_idx < first_low_idx
