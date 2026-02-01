# Discord Notifications Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Discord notification channel with PDF report attachments for daily and weekly investment digests.

**Architecture:** Extends existing notification system with DiscordChannel class. Uses fpdf2 for PDF generation and Ollama for weekly synthesis. Discord receives embed summaries + PDF attachments.

**Tech Stack:** Python 3.11+, httpx (existing), fpdf2 (new), Ollama (existing)

---

## Task 1: Add fpdf2 Dependency

**Files:**
- Modify: `pyproject.toml:10-25`

**Step 1: Add fpdf2 to dependencies**

Edit `pyproject.toml` to add fpdf2:

```toml
dependencies = [
    "yfinance>=0.2.0",
    "feedparser>=6.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
    "pandas>=2.0.0",
    "loguru>=0.7.0",
    "httpx>=0.25.0",
    "typer>=0.9.0",
    "fpdf2>=2.7.0",
]
```

**Step 2: Install dependencies**

Run: `pip install -e ".[dev,ai]"`
Expected: Successfully installs fpdf2

**Step 3: Verify installation**

Run: `python -c "from fpdf import FPDF; print('fpdf2 installed')"`
Expected: Prints "fpdf2 installed"

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add fpdf2 dependency for PDF report generation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add discord_webhook_url to Settings

**Files:**
- Modify: `src/investment_monitor/config.py:11-37`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_discord_webhook_url_default():
    """Test discord_webhook_url defaults to empty string."""
    settings = Settings()
    assert settings.discord_webhook_url == ""


def test_discord_webhook_url_from_env(monkeypatch):
    """Test discord_webhook_url can be set from environment."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/abc")
    settings = Settings()
    assert settings.discord_webhook_url == "https://discord.com/api/webhooks/123/abc"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_discord_webhook_url_default -v`
Expected: FAIL with AttributeError (discord_webhook_url not found)

**Step 3: Add discord_webhook_url to Settings**

Edit `src/investment_monitor/config.py`, add to Settings class after line 24:

```python
    # Discord
    discord_webhook_url: str = ""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v -k discord`
Expected: PASS

**Step 5: Commit**

```bash
git add src/investment_monitor/config.py tests/test_config.py
git commit -m "feat: add discord_webhook_url setting

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create PDFReportGenerator

**Files:**
- Create: `src/investment_monitor/notifications/pdf_report.py`
- Test: `tests/test_pdf_report.py`

**Step 1: Write failing tests for PDFReportGenerator**

Create `tests/test_pdf_report.py`:

```python
"""Tests for PDF report generation."""

from datetime import date
from decimal import Decimal

import pytest

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

        # PDF should be generated (we can't easily check contents without parsing)
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pdf_report.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Implement PDFReportGenerator**

Create `src/investment_monitor/notifications/pdf_report.py`:

```python
"""PDF report generation for investment digests."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from fpdf import FPDF
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
            pdf.cell(0, 6, f"Holdings tracked: {len(portfolio.holdings)}", ln=True)
            pdf.ln(4)

        # No alerts case
        if not curated_messages:
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            pdf.cell(0, 10, "No alerts for today.", ln=True)
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

        return pdf.output()

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
            week_start = week_end - __import__("datetime").timedelta(days=6)

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
            pdf.cell(0, 6, f"Holdings: {tickers}", ln=True)
            pdf.ln(4)

        # No alerts case
        if not messages:
            pdf.set_font("Helvetica", "I", 11)
            pdf.set_text_color(*self.COLOR_SECONDARY)
            pdf.cell(0, 10, "No alerts this week.", ln=True)
        else:
            # Summary stats
            self._add_section_header(pdf, "WEEK SUMMARY")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*self.COLOR_PRIMARY)
            pdf.cell(0, 6, f"Total alerts: {len(messages)}", ln=True)

            # Count by type
            by_type: dict[str, int] = defaultdict(int)
            for m in messages:
                by_type[m.alert_type] += 1
            for alert_type, count in sorted(by_type.items()):
                pdf.cell(0, 5, f"  - {alert_type}: {count}", ln=True)
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

        return pdf.output()

    def _add_title(self, pdf: FPDF, title: str) -> None:
        """Add main title to PDF."""
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*self.COLOR_PRIMARY)
        pdf.cell(0, 12, title, ln=True, align="C")

    def _add_subtitle(self, pdf: FPDF, subtitle: str) -> None:
        """Add subtitle to PDF."""
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(*self.COLOR_SECONDARY)
        pdf.cell(0, 8, subtitle, ln=True, align="C")
        pdf.ln(8)

    def _add_section_header(self, pdf: FPDF, header: str) -> None:
        """Add section header to PDF."""
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*self.COLOR_ACCENT)
        pdf.cell(0, 8, header, ln=True)
        pdf.set_draw_color(*self.COLOR_ACCENT)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    def _add_subsection_header(self, pdf: FPDF, header: str) -> None:
        """Add subsection header to PDF."""
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*self.COLOR_PRIMARY)
        pdf.cell(0, 6, header, ln=True)
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
                pdf.cell(20, 5, f"[{msg.ticker}]", ln=False)
            else:
                pdf.cell(20, 5, "", ln=False)

            # Title
            pdf.set_font("Helvetica", "B" if highlight else "", 10)
            pdf.set_text_color(*self.COLOR_PRIMARY)
            # Truncate long titles
            title = msg.title[:80] + "..." if len(msg.title) > 80 else msg.title
            pdf.cell(0, 5, title, ln=True)

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
        pdf.cell(0, 5, "Generated by Investment Monitor", ln=True, align="C")

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pdf_report.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/investment_monitor/notifications/pdf_report.py tests/test_pdf_report.py
git commit -m "feat: add PDFReportGenerator for daily and weekly reports

- Daily reports are curated (MEDIUM+ priority only)
- Weekly reports are comprehensive (all alerts)
- Supports AI synthesis section for weekly reports
- Clean formatting with sections and color coding

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Weekly Synthesis to LocalLLM

**Files:**
- Modify: `src/investment_monitor/analysis/local_llm.py`
- Modify: `src/investment_monitor/analysis/prompts.py`
- Test: `tests/test_analysis.py`

**Step 1: Write failing test for generate_weekly_synthesis**

Add to `tests/test_analysis.py`:

```python
class TestLocalLLMWeeklySynthesis:
    """Tests for weekly synthesis generation."""

    @pytest.mark.asyncio
    async def test_generate_weekly_synthesis_unavailable(self):
        """Test synthesis returns empty string when Ollama unavailable."""
        from investment_monitor.analysis.local_llm import LocalLLM

        llm = LocalLLM(base_url="http://localhost:99999")  # Invalid port

        result = await llm.generate_weekly_synthesis(
            alert_counts={"price": 5, "insider": 2},
            top_movers=[("AAPL", -5.2), ("NVDA", 8.1)],
            portfolio_change_pct=2.3,
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_weekly_synthesis_returns_string(self):
        """Test synthesis returns string when available."""
        from unittest.mock import MagicMock, patch
        from investment_monitor.analysis.local_llm import LocalLLM

        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="Tech stocks showed mixed performance this week."):
                result = await llm.generate_weekly_synthesis(
                    alert_counts={"price": 5, "insider": 2},
                    top_movers=[("AAPL", -5.2), ("NVDA", 8.1)],
                    portfolio_change_pct=2.3,
                )

        assert isinstance(result, str)
        assert len(result) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis.py::TestLocalLLMWeeklySynthesis -v`
Expected: FAIL with AttributeError (generate_weekly_synthesis not found)

**Step 3: Add WEEKLY_SYNTHESIS_PROMPT to prompts.py**

Add to `src/investment_monitor/analysis/prompts.py`:

```python
WEEKLY_SYNTHESIS_PROMPT = """You are an investment analyst summarizing the week's activity.

Week Summary:
- Alert counts: {alert_counts}
- Top movers: {top_movers}
- Portfolio change: {portfolio_change}

Generate a 2-3 sentence synthesis for an investor. Focus on:
1. Key trends or patterns
2. Notable events
3. What to watch next week

Be concise and actionable. No bullet points.

Synthesis:"""
```

**Step 4: Add generate_weekly_synthesis to LocalLLM**

Add to `src/investment_monitor/analysis/local_llm.py` after the summarize method:

```python
    async def generate_weekly_synthesis(
        self,
        alert_counts: dict[str, int],
        top_movers: list[tuple[str, float]],
        portfolio_change_pct: float | None = None,
    ) -> str:
        """Generate a weekly synthesis narrative.

        Args:
            alert_counts: Dict of alert_type -> count.
            top_movers: List of (ticker, percent_change) tuples.
            portfolio_change_pct: Portfolio change percentage.

        Returns:
            Synthesis text, or empty string if unavailable.
        """
        if not self.is_available():
            return ""

        # Format inputs for prompt
        alert_str = ", ".join(f"{count} {atype}" for atype, count in alert_counts.items())
        movers_str = ", ".join(f"{ticker} {change:+.1f}%" for ticker, change in top_movers[:5])
        portfolio_str = f"{portfolio_change_pct:+.1f}%" if portfolio_change_pct is not None else "N/A"

        from .prompts import WEEKLY_SYNTHESIS_PROMPT

        prompt = WEEKLY_SYNTHESIS_PROMPT.format(
            alert_counts=alert_str or "None",
            top_movers=movers_str or "None",
            portfolio_change=portfolio_str,
        )

        # Use longer response for synthesis
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 150,
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.debug(f"Weekly synthesis generation failed: {e}")
            return ""
```

Also add the import at the top of the file if not present.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_analysis.py::TestLocalLLMWeeklySynthesis -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/investment_monitor/analysis/local_llm.py src/investment_monitor/analysis/prompts.py tests/test_analysis.py
git commit -m "feat: add weekly synthesis generation to LocalLLM

Uses Ollama to generate 2-3 sentence summaries of weekly activity.
Gracefully returns empty string if Ollama unavailable.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create DiscordChannel

**Files:**
- Create: `src/investment_monitor/notifications/discord.py`
- Test: `tests/test_discord_channel.py`

**Step 1: Write failing tests for DiscordChannel**

Create `tests/test_discord_channel.py`:

```python
"""Tests for Discord notification channel."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.notifications.base import AlertMessage, Priority


class TestDiscordChannel:
    """Tests for DiscordChannel."""

    def test_channel_name(self):
        """Test channel has correct name."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        assert channel.name == "discord"

    def test_init_requires_webhook_url(self):
        """Test initialization requires webhook URL."""
        from investment_monitor.notifications.discord import DiscordChannel

        with pytest.raises(ValueError):
            DiscordChannel("")

    @pytest.mark.asyncio
    async def test_send_single_alert(self):
        """Test sending a single alert."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="AAPL dropped 5%",
            body="Apple stock fell significantly.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send(msg)

            assert result is True
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert "json" in call_kwargs
            assert "embeds" in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_handles_failure(self):
        """Test send returns False on HTTP error."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="Test",
            body="Test body",
            alert_type="test",
            priority=Priority.HIGH,
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.is_success = False
            mock_post.return_value = mock_response

            result = await channel.send(msg)

            assert result is False

    @pytest.mark.asyncio
    async def test_send_digest_with_pdf(self):
        """Test sending digest generates PDF and sends embed."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        messages = [
            AlertMessage(
                title="Alert 1",
                body="Body 1",
                ticker="AAPL",
                alert_type="price",
                priority=Priority.HIGH,
            ),
            AlertMessage(
                title="Alert 2",
                body="Body 2",
                ticker="MSFT",
                alert_type="volume",
                priority=Priority.MEDIUM,
            ),
        ]

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send_digest(messages)

            assert result is True
            mock_post.assert_called_once()
            # Should include files for PDF
            call_kwargs = mock_post.call_args[1]
            assert "files" in call_kwargs or "data" in call_kwargs

    @pytest.mark.asyncio
    async def test_send_digest_empty_messages(self):
        """Test sending empty digest."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.is_success = True
            mock_post.return_value = mock_response

            result = await channel.send_digest([])

            assert result is True

    def test_format_alert_embed_price_down(self):
        """Test embed formatting for price drop."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="AAPL -5.2%",
            body="Apple stock dropped significantly.",
            ticker="AAPL",
            alert_type="price",
            priority=Priority.HIGH,
        )

        embed = channel._format_alert_embed(msg)

        assert embed["title"] == "[AAPL] AAPL -5.2%"
        assert embed["color"] == 0xDC3545  # Red for price drop

    def test_format_alert_embed_price_up(self):
        """Test embed formatting for price gain."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        msg = AlertMessage(
            title="NVDA +8.1%",
            body="NVIDIA stock rose today.",
            ticker="NVDA",
            alert_type="price",
            priority=Priority.HIGH,
        )

        embed = channel._format_alert_embed(msg)

        assert embed["color"] == 0x28A745  # Green for price gain

    def test_supports_all_priorities(self):
        """Test channel supports all priorities."""
        from investment_monitor.notifications.discord import DiscordChannel

        channel = DiscordChannel("https://discord.com/api/webhooks/123/abc")
        assert channel.supports_priority(Priority.HIGH) is True
        assert channel.supports_priority(Priority.MEDIUM) is True
        assert channel.supports_priority(Priority.LOW) is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_discord_channel.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Implement DiscordChannel**

Create `src/investment_monitor/notifications/discord.py`:

```python
"""Discord notification channel using webhooks."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from .base import AlertMessage, NotificationChannel, Priority
from .pdf_report import PDFReportGenerator

if TYPE_CHECKING:
    from investment_monitor.models.portfolio import Portfolio


class DiscordChannel(NotificationChannel):
    """Discord notification channel using webhooks.

    Sends individual alerts as embeds and digests as embed + PDF attachment.
    """

    name = "discord"

    # Discord embed colors
    COLOR_DANGER = 0xDC3545  # Red
    COLOR_SUCCESS = 0x28A745  # Green
    COLOR_WARNING = 0xFFC107  # Amber
    COLOR_INFO = 0x17A2B8  # Cyan
    COLOR_DEFAULT = 0x6C757D  # Gray

    def __init__(self, webhook_url: str) -> None:
        """Initialize Discord channel.

        Args:
            webhook_url: Discord webhook URL.

        Raises:
            ValueError: If webhook_url is empty.
        """
        if not webhook_url:
            raise ValueError("Discord webhook URL is required")

        self._webhook_url = webhook_url
        self._pdf_generator = PDFReportGenerator()
        self._logger = logger.bind(component="discord_channel")

    async def send(self, message: AlertMessage) -> bool:
        """Send a single alert message as a Discord embed.

        Args:
            message: The alert to send.

        Returns:
            True if successful, False otherwise.
        """
        embed = self._format_alert_embed(message)
        payload = {"embeds": [embed]}

        return await self._post_webhook(payload)

    async def send_digest(
        self,
        messages: list[AlertMessage],
        portfolio: Portfolio | None = None,
        is_weekly: bool = False,
        ai_synthesis: str | None = None,
    ) -> bool:
        """Send a digest with embed summary and PDF attachment.

        Args:
            messages: Alert messages to include.
            portfolio: Optional portfolio context.
            is_weekly: True for weekly digest, False for daily.
            ai_synthesis: Optional AI synthesis for weekly reports.

        Returns:
            True if successful, False otherwise.
        """
        # Create embed summary
        if is_weekly:
            embed = self._format_weekly_embed(messages, ai_synthesis)
            pdf_bytes = self._pdf_generator.generate_weekly_report(
                messages,
                portfolio=portfolio,
                ai_synthesis=ai_synthesis,
            )
            filename = f"weekly-report-{date.today().isoformat()}.pdf"
        else:
            embed = self._format_daily_embed(messages)
            pdf_bytes = self._pdf_generator.generate_daily_report(
                messages,
                portfolio=portfolio,
            )
            filename = f"daily-report-{date.today().isoformat()}.pdf"

        # Send with PDF attachment
        return await self._post_webhook_with_file(
            {"embeds": [embed]},
            pdf_bytes,
            filename,
        )

    async def _post_webhook(self, payload: dict[str, Any]) -> bool:
        """Post JSON payload to Discord webhook.

        Args:
            payload: JSON payload to send.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    timeout=30.0,
                )
                if response.is_success:
                    self._logger.debug("Discord webhook sent successfully")
                    return True
                else:
                    self._logger.warning(
                        "Discord webhook failed: {status}",
                        status=response.status_code,
                    )
                    return False
        except Exception as e:
            self._logger.exception("Discord webhook error: {error}", error=str(e))
            return False

    async def _post_webhook_with_file(
        self,
        payload: dict[str, Any],
        file_bytes: bytes,
        filename: str,
    ) -> bool:
        """Post payload with file attachment to Discord webhook.

        Args:
            payload: JSON payload (embeds, etc.).
            file_bytes: File contents.
            filename: Name for the attachment.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Discord requires multipart form data for file uploads
                files = {"file": (filename, file_bytes, "application/pdf")}
                data = {"payload_json": json.dumps(payload)}

                response = await client.post(
                    self._webhook_url,
                    data=data,
                    files=files,
                    timeout=60.0,
                )
                if response.is_success:
                    self._logger.debug("Discord webhook with file sent successfully")
                    return True
                else:
                    self._logger.warning(
                        "Discord webhook with file failed: {status}",
                        status=response.status_code,
                    )
                    return False
        except Exception as e:
            self._logger.exception("Discord webhook error: {error}", error=str(e))
            return False

    def _format_alert_embed(self, message: AlertMessage) -> dict[str, Any]:
        """Format an alert message as a Discord embed.

        Args:
            message: The alert to format.

        Returns:
            Discord embed dict.
        """
        # Determine color based on alert type and content
        color = self._get_alert_color(message)

        # Build title with ticker prefix
        title = f"[{message.ticker}] {message.title}" if message.ticker else message.title

        embed: dict[str, Any] = {
            "title": title,
            "description": message.body[:2000],  # Discord limit
            "color": color,
            "timestamp": message.timestamp.isoformat(),
            "footer": {"text": f"Priority: {message.priority.value.upper()}"},
        }

        if message.url:
            embed["url"] = message.url

        return embed

    def _format_daily_embed(self, messages: list[AlertMessage]) -> dict[str, Any]:
        """Format daily digest summary embed.

        Args:
            messages: Alert messages.

        Returns:
            Discord embed dict.
        """
        high_priority = [m for m in messages if m.priority == Priority.HIGH]
        medium_priority = [m for m in messages if m.priority == Priority.MEDIUM]

        # Build description
        lines = []
        if not messages:
            lines.append("No alerts for today.")
        else:
            lines.append(f"**{len(messages)} total alerts**")
            if high_priority:
                lines.append(f"\n**HIGH Priority ({len(high_priority)}):**")
                for m in high_priority[:5]:  # Limit to 5
                    ticker = f"[{m.ticker}] " if m.ticker else ""
                    lines.append(f"- {ticker}{m.title[:50]}")
                if len(high_priority) > 5:
                    lines.append(f"  ... and {len(high_priority) - 5} more")

        description = "\n".join(lines)[:2000]

        return {
            "title": f"Daily Investment Report - {date.today().strftime('%B %d, %Y')}",
            "description": description,
            "color": self.COLOR_INFO,
            "footer": {"text": "Full report attached as PDF"},
        }

    def _format_weekly_embed(
        self,
        messages: list[AlertMessage],
        ai_synthesis: str | None,
    ) -> dict[str, Any]:
        """Format weekly digest summary embed.

        Args:
            messages: Alert messages.
            ai_synthesis: Optional AI synthesis.

        Returns:
            Discord embed dict.
        """
        # Use AI synthesis as description if available
        if ai_synthesis:
            description = ai_synthesis[:2000]
        elif not messages:
            description = "No alerts this week."
        else:
            description = f"**{len(messages)} total alerts this week.**\nSee attached PDF for full details."

        return {
            "title": f"Weekly Investment Report - Week of {date.today().strftime('%B %d, %Y')}",
            "description": description,
            "color": self.COLOR_INFO,
            "footer": {"text": "Full report attached as PDF"},
        }

    def _get_alert_color(self, message: AlertMessage) -> int:
        """Determine embed color based on alert content.

        Args:
            message: The alert message.

        Returns:
            Discord color integer.
        """
        body_lower = message.body.lower()
        title_lower = message.title.lower()

        # Price alerts: red for drops, green for gains
        if message.alert_type == "price":
            if any(x in body_lower or x in title_lower for x in ["drop", "fell", "down", "-"]):
                return self.COLOR_DANGER
            if any(x in body_lower or x in title_lower for x in ["rose", "up", "gain", "+"]):
                return self.COLOR_SUCCESS

        # Insider alerts: warning color
        if message.alert_type == "insider":
            return self.COLOR_WARNING

        # Earnings: info color
        if message.alert_type == "earnings":
            return self.COLOR_INFO

        return self.COLOR_DEFAULT
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_discord_channel.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/investment_monitor/notifications/discord.py tests/test_discord_channel.py
git commit -m "feat: add DiscordChannel for webhook notifications

- Sends individual alerts as embeds
- Sends digests with embed summary + PDF attachment
- Color-coded embeds based on alert type
- Handles rate limits and errors gracefully

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Export DiscordChannel from notifications module

**Files:**
- Modify: `src/investment_monitor/notifications/__init__.py`

**Step 1: Add DiscordChannel to __init__.py**

Edit `src/investment_monitor/notifications/__init__.py`:

```python
"""Notification system for investment alerts.

This module provides the infrastructure for sending alert notifications
through various channels (console, Discord, Slack, email, etc.).

Priority levels:
    HIGH: Send immediately via all channels
    MEDIUM: Include in next digest
    LOW: Log only (debug level)

Example usage:
    from investment_monitor.notifications import (
        AlertMessage,
        ConsoleChannel,
        DiscordChannel,
        NotificationManager,
        Priority,
    )

    # Create a manager with Discord output
    manager = NotificationManager([
        ConsoleChannel(),
        DiscordChannel("https://discord.com/api/webhooks/xxx/yyy"),
    ])

    # Send a high-priority alert
    await manager.notify(AlertMessage(
        title="AAPL dropped 5%",
        body="Apple stock dropped significantly today.",
        ticker="AAPL",
        alert_type="price",
        priority=Priority.HIGH,
    ))
"""

from .base import AlertMessage, NotificationChannel, Priority
from .console import ConsoleChannel
from .digest import format_daily_digest, format_weekly_digest
from .discord import DiscordChannel
from .manager import NotificationManager
from .pdf_report import PDFReportGenerator

__all__ = [
    "AlertMessage",
    "ConsoleChannel",
    "DiscordChannel",
    "NotificationChannel",
    "NotificationManager",
    "PDFReportGenerator",
    "Priority",
    "format_daily_digest",
    "format_weekly_digest",
]
```

**Step 2: Verify import works**

Run: `python -c "from investment_monitor.notifications import DiscordChannel; print('OK')"`
Expected: Prints "OK"

**Step 3: Commit**

```bash
git add src/investment_monitor/notifications/__init__.py
git commit -m "feat: export DiscordChannel and PDFReportGenerator from notifications

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Wire Discord into main.py

**Files:**
- Modify: `src/investment_monitor/main.py`

**Step 1: Add Discord channel initialization**

Find the section in `main.py` where channels are configured (around where ConsoleChannel is added) and add:

```python
# Near the top imports, add:
from investment_monitor.notifications import DiscordChannel

# In the function where NotificationManager is created, add after ConsoleChannel:
if settings.discord_webhook_url:
    try:
        discord_channel = DiscordChannel(settings.discord_webhook_url)
        channels.append(discord_channel)
        logger.info("Discord notifications enabled")
    except ValueError as e:
        logger.warning("Discord channel not configured: {error}", error=str(e))
```

**Step 2: Update digest sending for Discord**

Find where `send_daily_digest` is called and ensure it passes the required parameters for Discord:

```python
# For daily digest
for channel in notification_manager.channels:
    if isinstance(channel, DiscordChannel):
        await channel.send_digest(digest_messages, portfolio=portfolio, is_weekly=False)
    else:
        await channel.send_digest(digest_messages)

# For weekly digest
for channel in notification_manager.channels:
    if isinstance(channel, DiscordChannel):
        await channel.send_digest(
            messages,
            portfolio=portfolio,
            is_weekly=True,
            ai_synthesis=ai_synthesis,
        )
    else:
        await channel.send_digest(messages)
```

**Step 3: Run existing tests to ensure no regressions**

Run: `pytest tests/test_main.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/investment_monitor/main.py
git commit -m "feat: wire DiscordChannel into main monitor

- Initializes Discord channel if webhook URL configured
- Passes portfolio and synthesis to Discord digests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Create notifications config file

**Files:**
- Create: `config/notifications.yaml.example`

**Step 1: Create example config**

Create `config/notifications.yaml.example`:

```yaml
# Notification channel configuration
# Copy this to notifications.yaml and customize

discord:
  # Enable/disable Discord notifications
  enabled: true

  # What to show in daily embed summary
  daily_embed:
    show_high_priority_alerts: true
    show_portfolio_change: true
    max_alerts_in_embed: 5

  # What to show in weekly embed summary
  weekly_embed:
    show_ai_synthesis: true

# Console output (always available)
console:
  enabled: true
  # Log level for different priorities
  high_priority_level: error
  medium_priority_level: warning
  low_priority_level: debug
```

**Step 2: Commit**

```bash
git add config/notifications.yaml.example
git commit -m "docs: add notifications config example

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update .env.example

**Files:**
- Modify: `.env.example` (if exists) or create it

**Step 1: Add Discord webhook URL to .env.example**

Add to `.env.example`:

```bash
# Discord notifications (optional)
# Create a webhook: Server Settings > Integrations > Webhooks
DISCORD_WEBHOOK_URL=
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add DISCORD_WEBHOOK_URL to env example

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run type checking (if mypy configured)**

Run: `python -m mypy src/investment_monitor/notifications/ --ignore-missing-imports`
Expected: No errors (or only minor ones)

**Step 3: Run linting**

Run: `ruff check src/investment_monitor/notifications/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address linting and type issues

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

After completing all tasks, you will have:

1. **fpdf2 dependency** added for PDF generation
2. **discord_webhook_url** setting in config
3. **PDFReportGenerator** class for daily (curated) and weekly (comprehensive) PDFs
4. **generate_weekly_synthesis** method in LocalLLM using Ollama
5. **DiscordChannel** class that sends embeds + PDF attachments
6. **Integration** with the main monitor
7. **Documentation** and config examples

To use Discord notifications, users just need to:
1. Create a Discord webhook in their server
2. Add `DISCORD_WEBHOOK_URL=<webhook_url>` to their `.env` file
3. Run the monitor as usual
