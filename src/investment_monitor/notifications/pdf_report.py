"""PDF report generation for investment digests."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from loguru import logger

from .base import AlertMessage, Priority

if TYPE_CHECKING:
    from investment_monitor.models.portfolio import Portfolio


class PDFReportGenerator:
    """Generates PDF reports for daily and weekly investment digests."""

    # Colors (RGB)
    COLOR_PRIMARY = (33, 37, 41)  # Dark gray for text
    COLOR_SECONDARY = (108, 117, 125)  # Medium gray
    COLOR_ACCENT = (0, 123, 255)  # Blue
    COLOR_SUCCESS = (40, 167, 69)  # Green
    COLOR_DANGER = (220, 53, 69)  # Red
    COLOR_WARNING = (255, 193, 7)  # Yellow/amber

    def __init__(self) -> None:
        """Initialize the PDF generator."""
        self._logger = logger.bind(component="pdf_report")

    def generate_daily_report(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        date_value: date | None = None,
    ) -> bytes:
        """Generate a daily report PDF.

        Args:
            messages: Alert messages to include (filters out LOW priority).
            portfolio: Optional portfolio for context.
            date_value: Report date. Defaults to today.

        Returns:
            PDF file contents as bytes.
        """
        if date_value is None:
            date_value = date.today()

        # Filter to MEDIUM+ priority only (curated)
        curated_messages = [m for m in messages if m.priority != Priority.LOW]

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        self._add_title(pdf, "DAILY INVESTMENT REPORT")
        self._add_subtitle(pdf, date_value.strftime("%B %d, %Y"))

        # Portfolio snapshot (if available)
        if portfolio and portfolio.holdings:
            self._add_section_header(pdf, "PORTFOLIO SNAPSHOT")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            pdf.cell(0, 6, f"Holdings tracked: {len(portfolio.holdings)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)

        # No alerts case
        if not curated_messages:
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            pdf.cell(0, 10, "No alerts for today.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # Group by priority
            high_priority = [m for m in curated_messages if m.priority == Priority.HIGH]
            medium_priority = [m for m in curated_messages if m.priority == Priority.MEDIUM]

            if high_priority:
                self._add_section_header(pdf, "HIGH PRIORITY ALERTS")
                self._add_alerts(pdf, high_priority, highlight=True)

            if medium_priority:
                self._add_section_header(pdf, "MEDIUM PRIORITY ALERTS")
                self._add_alerts(pdf, medium_priority, highlight=False)

        # Footer
        self._add_footer(pdf)

        return bytes(pdf.output())

    def generate_weekly_report(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        week_start: date | None = None,
        week_end: date | None = None,
        ai_synthesis: str | None = None,
    ) -> bytes:
        """Generate a weekly report PDF (comprehensive - all alerts).

        Args:
            messages: All alert messages from the week.
            portfolio: Optional portfolio for context.
            week_start: Start of week. Defaults to 7 days ago.
            week_end: End of week. Defaults to today.
            ai_synthesis: Optional AI-generated summary.

        Returns:
            PDF file contents as bytes.
        """
        if week_end is None:
            week_end = date.today()
        if week_start is None:
            week_start = week_end - timedelta(days=6)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        self._add_title(pdf, "WEEKLY INVESTMENT REPORT")
        date_range = self._format_date_range(week_start, week_end)
        self._add_subtitle(pdf, date_range)

        # AI Synthesis
        if ai_synthesis:
            self._add_section_header(pdf, "AI SYNTHESIS")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*self.COLOR_PRIMARY)
            # Handle multi-line synthesis
            pdf.multi_cell(0, 5, ai_synthesis)
            pdf.ln(4)

        # Portfolio summary
        if portfolio and portfolio.holdings:
            self._add_section_header(pdf, "PORTFOLIO OVERVIEW")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            tickers = ", ".join(h.ticker for h in portfolio.holdings[:10])
            if len(portfolio.holdings) > 10:
                tickers += f" (+{len(portfolio.holdings) - 10} more)"
            pdf.cell(0, 6, f"Holdings: {tickers}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)

        # No alerts case
        if not messages:
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            pdf.cell(0, 10, "No alerts this week.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # Summary stats
            self._add_section_header(pdf, "WEEK SUMMARY")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*self.COLOR_PRIMARY)
            pdf.cell(0, 6, f"Total alerts: {len(messages)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Count by type
            by_type: dict[str, int] = defaultdict(int)
            for m in messages:
                by_type[m.alert_type] += 1
            for alert_type, count in sorted(by_type.items()):
                pdf.cell(0, 5, f"  - {alert_type}: {count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)

            # All alerts by category
            self._add_section_header(pdf, "ALL ALERTS BY CATEGORY")
            grouped = self._group_by_type(messages)

            type_order = ["price", "volume", "insider", "news", "earnings", "dividend", "filing", "analyst", "system"]
            sorted_types = sorted(
                grouped.keys(),
                key=lambda t: (type_order.index(t.lower()) if t.lower() in type_order else len(type_order), t.lower())
            )

            for alert_type in sorted_types:
                type_messages = grouped[alert_type]
                self._add_subsection_header(pdf, alert_type.upper())
                self._add_alerts(pdf, type_messages, highlight=False)

        # Footer
        self._add_footer(pdf)

        return bytes(pdf.output())

    def _add_title(self, pdf: FPDF, title: str) -> None:
        """Add main title to PDF."""
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*self.COLOR_PRIMARY)
        pdf.cell(0, 12, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    def _add_subtitle(self, pdf: FPDF, subtitle: str) -> None:
        """Add subtitle to PDF."""
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(*self.COLOR_SECONDARY)
        pdf.cell(0, 8, subtitle, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(8)

    def _add_section_header(self, pdf: FPDF, header: str) -> None:
        """Add section header to PDF."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*self.COLOR_ACCENT)
        pdf.cell(0, 8, header, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(*self.COLOR_ACCENT)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    def _add_subsection_header(self, pdf: FPDF, header: str) -> None:
        """Add subsection header to PDF."""
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*self.COLOR_PRIMARY)
        pdf.cell(0, 6, header, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def _add_alerts(self, pdf: FPDF, messages: list[AlertMessage], highlight: bool) -> None:
        """Add alert messages to PDF."""
        for msg in messages:
            # Ticker prefix
            if msg.ticker:
                pdf.set_font("Helvetica", "B", 10)
                if highlight:
                    pdf.set_text_color(*self.COLOR_DANGER)
                else:
                    pdf.set_text_color(*self.COLOR_PRIMARY)
                pdf.cell(20, 5, f"[{msg.ticker}]", new_x=XPos.RIGHT, new_y=YPos.TOP)
            else:
                pdf.cell(20, 5, "", new_x=XPos.RIGHT, new_y=YPos.TOP)

            # Title
            pdf.set_font("Helvetica", "B" if highlight else "", 10)
            pdf.set_text_color(*self.COLOR_PRIMARY)
            # Truncate long titles
            title = msg.title[:80] + "..." if len(msg.title) > 80 else msg.title
            pdf.cell(0, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Body (truncated for daily, full for weekly could be handled differently)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            body = msg.body[:200] + "..." if len(msg.body) > 200 else msg.body
            pdf.set_x(30)
            pdf.multi_cell(0, 4, body)

            pdf.ln(2)

    def _add_footer(self, pdf: FPDF) -> None:
        """Add footer to PDF."""
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*self.COLOR_SECONDARY)
        pdf.cell(0, 5, "Generated by Investment Monitor", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    def _format_date_range(self, start: date, end: date) -> str:
        """Format date range string."""
        if start.year == end.year:
            if start.month == end.month:
                return f"{start.strftime('%B %d')} - {end.day}, {end.year}"
            return f"{start.strftime('%B %d')} - {end.strftime('%B %d')}, {end.year}"
        return f"{start.strftime('%B %d, %Y')} - {end.strftime('%B %d, %Y')}"

    def _group_by_type(self, messages: list[AlertMessage]) -> dict[str, list[AlertMessage]]:
        """Group messages by alert type."""
        grouped: dict[str, list[AlertMessage]] = defaultdict(list)
        for msg in messages:
            grouped[msg.alert_type].append(msg)
        return dict(grouped)
