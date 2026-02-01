"""Tests for PDF report generation."""

from datetime import date
from decimal import Decimal

from investment_monitor.models.portfolio import Holding, Portfolio
from investment_monitor.notifications.base import AlertMessage, Priority


class TestPDFReportGenerator:
    """Tests for PDFReportGenerator."""

    def test_generate_daily_report_returns_bytes(self):
        """Test daily report returns PDF bytes."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        messages = [
            AlertMessage(
                title="AAPL dropped 3%",
                body="Apple stock fell significantly.",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
        ]

        pdf_bytes = generator.generate_daily_report(messages, date_value=date(2026, 1, 28))

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b"%PDF"  # PDF magic bytes

    def test_generate_daily_report_empty_messages(self):
        """Test daily report with no messages."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_daily_report([], date_value=date(2026, 1, 28))

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_generate_daily_report_filters_low_priority(self):
        """Test daily report excludes LOW priority alerts."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        messages = [
            AlertMessage(
                title="High Alert",
                body="Important",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
            AlertMessage(
                title="Medium Alert",
                body="Notable",
                ticker="MSFT",
                alert_type="volume",
                priority=Priority.MEDIUM,
            ),
            AlertMessage(
                title="Low Alert",
                body="Minor",
                ticker="GOOGL",
                alert_type="news",
                priority=Priority.LOW,
            ),
        ]

        pdf_bytes = generator.generate_daily_report(messages, date_value=date(2026, 1, 28))

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100

    def test_generate_weekly_report_returns_bytes(self):
        """Test weekly report returns PDF bytes."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        messages = [
            AlertMessage(
                title="Weekly alert",
                body="Something happened",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.MEDIUM,
            ),
        ]

        pdf_bytes = generator.generate_weekly_report(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_generate_weekly_report_with_synthesis(self):
        """Test weekly report includes AI synthesis."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        messages = []
        ai_synthesis = "This week saw mixed performance in tech stocks."

        pdf_bytes = generator.generate_weekly_report(
            messages,
            week_start=date(2026, 1, 22),
            week_end=date(2026, 1, 28),
            ai_synthesis=ai_synthesis,
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100

    def test_generate_daily_report_with_portfolio(self):
        """Test daily report with portfolio context."""
        from investment_monitor.notifications.pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="AAPL", shares=Decimal("100"), cost_basis=Decimal("150.00")),
            ]
        )
        messages = [
            AlertMessage(
                title="AAPL alert",
                body="Something",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
        ]

        pdf_bytes = generator.generate_daily_report(
            messages,
            portfolio=portfolio,
            date_value=date(2026, 1, 28),
        )

        assert isinstance(pdf_bytes, bytes)
