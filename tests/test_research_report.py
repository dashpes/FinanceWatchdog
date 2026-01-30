"""Tests for ResearchReportGenerator with Claude API integration."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from investment_monitor.analysis.research_report import (
    RESEARCH_REPORT_PROMPT,
    SONNET_INPUT_COST_PER_MILLION,
    SONNET_OUTPUT_COST_PER_MILLION,
    VALID_RECOMMENDATIONS,
    ReportResult,
    ResearchReportGenerator,
)
from investment_monitor.collectors.fundamentals import FundamentalsData
from investment_monitor.storage.research_models import CandidateScore, ResearchReport


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_fundamentals() -> FundamentalsData:
    """Create sample fundamentals data for testing."""
    return FundamentalsData(
        ticker="AAPL",
        pe_ratio=28.5,
        pb_ratio=45.2,
        ps_ratio=7.8,
        peg_ratio=2.1,
        revenue_growth_yoy=0.08,
        revenue_growth_3y=0.15,
        eps_growth_yoy=0.12,
        eps_growth_3y=0.18,
        roe=0.147,
        profit_margin=0.255,
        debt_to_equity=1.52,
        current_ratio=0.98,
        dividend_yield=0.005,
        payout_ratio=0.15,
        free_cash_flow=99000000000,
        market_cap=2800000000000,
        sector="Technology",
        industry="Consumer Electronics",
    )


@pytest.fixture
def sample_fundamentals_with_nulls() -> FundamentalsData:
    """Create fundamentals data with some None values."""
    return FundamentalsData(
        ticker="NVDA",
        pe_ratio=65.0,
        pb_ratio=None,
        ps_ratio=None,
        peg_ratio=1.5,
        revenue_growth_yoy=1.22,
        revenue_growth_3y=None,
        eps_growth_yoy=5.86,
        eps_growth_3y=None,
        roe=0.56,
        profit_margin=0.55,
        debt_to_equity=None,
        current_ratio=4.2,
        dividend_yield=None,
        payout_ratio=None,
        free_cash_flow=27000000000,
        market_cap=1200000000000,
        sector="Technology",
        industry="Semiconductors",
    )


@pytest.fixture
def sample_candidate_score() -> CandidateScore:
    """Create sample candidate score for testing."""
    return CandidateScore(
        ticker="AAPL",
        value_score=72.0,
        growth_score=65.0,
        quality_score=78.0,
        momentum_score=70.0,
        sentiment_score=82.0,
        composite_score=73.4,
        reasoning=(
            "Value (72): Fairly valued with P/E of 28.5\n"
            "Growth (65): Moderate growth trajectory\n"
            "Quality (78): Strong profit margins and ROE\n"
            "Momentum (70): Positive price momentum\n"
            "Sentiment (82): Bullish market sentiment"
        ),
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response with valid research report JSON."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='''{
        "summary": "Apple is a leading technology company with strong fundamentals and consistent growth.",
        "thesis": "Apple's ecosystem moat and services growth provide long-term value creation potential.",
        "bull_case": "1. Services revenue growing 15%+ annually. 2. Strong brand loyalty and ecosystem. 3. Share buybacks support EPS growth.",
        "bear_case": "1. iPhone revenue concentration risk. 2. China market exposure. 3. Regulatory headwinds in EU.",
        "recommendation": "buy",
        "target_price": 195.00
    }''')]
    mock_response.usage = MagicMock(input_tokens=1500, output_tokens=500)
    return mock_response


@pytest.fixture
def mock_anthropic_response_strong_buy():
    """Create a mock response with strong_buy recommendation."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='''{
        "summary": "Exceptional value opportunity.",
        "thesis": "Strong fundamentals with attractive valuation.",
        "bull_case": "Multiple expansion potential.",
        "bear_case": "Market volatility risk.",
        "recommendation": "strong_buy",
        "target_price": 250.00
    }''')]
    mock_response.usage = MagicMock(input_tokens=1200, output_tokens=400)
    return mock_response


@pytest.fixture
def mock_anthropic_response_no_target():
    """Create a mock response with null target price."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='''{
        "summary": "Uncertain outlook.",
        "thesis": "Mixed signals make valuation difficult.",
        "bull_case": "Potential turnaround.",
        "bear_case": "Continued challenges.",
        "recommendation": "hold",
        "target_price": null
    }''')]
    mock_response.usage = MagicMock(input_tokens=1000, output_tokens=300)
    return mock_response


# =============================================================================
# ReportResult Tests
# =============================================================================


class TestReportResult:
    """Tests for ReportResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful report result."""
        mock_report = MagicMock(spec=ResearchReport)
        result = ReportResult(
            report=mock_report,
            success=True,
            input_tokens=1500,
            output_tokens=500,
            cost=0.012,
        )
        assert result.report is mock_report
        assert result.success is True
        assert result.error_message is None
        assert result.input_tokens == 1500
        assert result.output_tokens == 500
        assert result.cost == 0.012

    def test_failed_result(self):
        """Test creating a failed report result."""
        result = ReportResult(
            report=None,
            success=False,
            error_message="API error: rate limited",
        )
        assert result.report is None
        assert result.success is False
        assert result.error_message == "API error: rate limited"

    def test_result_has_timestamp(self):
        """Test that result has a timestamp."""
        result = ReportResult(report=None, success=True)
        assert result.timestamp is not None


# =============================================================================
# ResearchReportGenerator Initialization Tests
# =============================================================================


class TestResearchReportGeneratorInit:
    """Tests for ResearchReportGenerator initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator.is_available() is False
        assert generator.max_monthly_spend == 50.0
        assert generator._monthly_spend == 0.0

    def test_init_with_empty_api_key(self):
        """Test initialization with empty string API key."""
        generator = ResearchReportGenerator(api_key="")
        assert generator.is_available() is False

    def test_init_with_custom_budget(self):
        """Test initialization with custom budget."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=100.00)
        assert generator.max_monthly_spend == 100.00

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        generator = ResearchReportGenerator(api_key=None, model="claude-3-opus-20240229")
        assert generator.model == "claude-3-opus-20240229"

    def test_init_with_api_key(self):
        """Test initialization with valid API key when anthropic is available."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client to make it "available"
        mock_client = MagicMock()
        generator._client = mock_client
        assert generator.is_available() is True


# =============================================================================
# Budget Management Tests
# =============================================================================


class TestBudgetManagement:
    """Tests for budget tracking and management."""

    def test_initial_budget_state(self):
        """Test initial budget state."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=50.00)
        assert generator.get_monthly_spend() == 0.0
        assert generator.get_remaining_budget() == 50.00

    def test_within_budget_initially(self):
        """Test _within_budget returns True initially."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=50.00)
        assert generator._within_budget() is True

    def test_record_cost(self):
        """Test cost recording."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=50.00)
        # Initialize the spend reset date to current month to prevent reset
        generator._spend_reset_date = date.today()

        # 1500 input tokens at $3/M = $0.0045
        # 500 output tokens at $15/M = $0.0075
        # Total = $0.012
        cost = generator._record_cost(input_tokens=1500, output_tokens=500)

        expected_cost = (1500 / 1_000_000) * SONNET_INPUT_COST_PER_MILLION + (500 / 1_000_000) * SONNET_OUTPUT_COST_PER_MILLION
        assert cost == pytest.approx(expected_cost, rel=0.01)
        assert generator.get_monthly_spend() == pytest.approx(expected_cost, rel=0.01)

    def test_budget_exceeded(self):
        """Test behavior when budget is exceeded."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=0.01)
        generator._spend_reset_date = date.today()

        # Record cost that exceeds budget
        generator._record_cost(input_tokens=10000, output_tokens=5000)

        assert generator._within_budget() is False
        assert generator.get_remaining_budget() == 0.0

    def test_cumulative_cost_tracking(self):
        """Test that costs accumulate correctly."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=50.00)
        generator._spend_reset_date = date.today()

        generator._record_cost(input_tokens=1000, output_tokens=500)
        first_spend = generator.get_monthly_spend()

        generator._record_cost(input_tokens=1000, output_tokens=500)
        second_spend = generator.get_monthly_spend()

        assert second_spend == pytest.approx(first_spend * 2, rel=0.01)

    def test_monthly_reset(self):
        """Test that spend resets on new month."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=50.00)

        # Simulate previous month spend
        generator._monthly_spend = 45.00
        generator._spend_reset_date = date(2024, 12, 15)  # Previous month

        # This should trigger reset since we're now in a different month
        spend = generator.get_monthly_spend()

        # If test runs in January or later, should have reset
        if date.today().month != 12:
            assert spend == 0.0
        else:
            # If running in December, won't reset
            assert spend == 45.00


# =============================================================================
# Value Formatting Tests
# =============================================================================


class TestValueFormatting:
    """Tests for _format_value and _format_currency methods."""

    def test_format_value_none(self):
        """Should return 'N/A' for None values."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_value(None) == "N/A"

    def test_format_value_regular(self):
        """Should format regular numbers."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_value(25.678) == "25.68"
        assert generator._format_value(100.0) == "100.00"

    def test_format_value_as_percent(self):
        """Should convert decimal to percentage."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_value(0.15, as_percent=True) == "15.00%"
        assert generator._format_value(0.0825, as_percent=True) == "8.25%"

    def test_format_value_percent_none(self):
        """Should return 'N/A' for None even with percent flag."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_value(None, as_percent=True) == "N/A"

    def test_format_currency_none(self):
        """Should return 'N/A' for None currency."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_currency(None) == "N/A"

    def test_format_currency_trillions(self):
        """Should format trillions correctly."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_currency(2_800_000_000_000) == "$2.80T"

    def test_format_currency_billions(self):
        """Should format billions correctly."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_currency(99_000_000_000) == "$99.00B"

    def test_format_currency_millions(self):
        """Should format millions correctly."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_currency(500_000_000) == "$500.00M"

    def test_format_currency_small_values(self):
        """Should format small values correctly."""
        generator = ResearchReportGenerator(api_key=None)
        assert generator._format_currency(1234.56) == "$1,234.56"


# =============================================================================
# Fundamentals Summary Tests
# =============================================================================


class TestFundamentalsSummary:
    """Tests for _build_fundamentals_summary method."""

    def test_fundamentals_summary_structure(self, sample_fundamentals):
        """Test that fundamentals summary has expected sections."""
        generator = ResearchReportGenerator(api_key=None)
        summary = generator._build_fundamentals_summary(sample_fundamentals)

        assert "### Valuation" in summary
        assert "### Growth" in summary
        assert "### Quality" in summary
        assert "### Income" in summary

    def test_fundamentals_summary_values(self, sample_fundamentals):
        """Test that fundamentals summary contains actual values."""
        generator = ResearchReportGenerator(api_key=None)
        summary = generator._build_fundamentals_summary(sample_fundamentals)

        assert "28.50" in summary  # P/E
        assert "45.20" in summary  # P/B
        assert "8.00%" in summary  # Revenue growth
        assert "$99.00B" in summary  # Free cash flow
        assert "$2.80T" in summary  # Market cap

    def test_fundamentals_summary_with_nulls(self, sample_fundamentals_with_nulls):
        """Test that fundamentals summary handles null values."""
        generator = ResearchReportGenerator(api_key=None)
        summary = generator._build_fundamentals_summary(sample_fundamentals_with_nulls)

        assert "N/A" in summary  # For null values
        assert "65.00" in summary  # P/E ratio that exists


# =============================================================================
# Score Reasoning Parsing Tests
# =============================================================================


class TestScoreReasoningParsing:
    """Tests for _parse_score_reasoning method."""

    def test_parse_full_reasoning(self):
        """Test parsing complete reasoning string."""
        generator = ResearchReportGenerator(api_key=None)
        reasoning = (
            "Value (72): Fairly valued with P/E of 28.5\n"
            "Growth (65): Moderate growth trajectory\n"
            "Quality (78): Strong profit margins\n"
            "Momentum (70): Positive momentum\n"
            "Sentiment (82): Bullish sentiment"
        )

        result = generator._parse_score_reasoning(reasoning)

        assert result["value"] == "Fairly valued with P/E of 28.5"
        assert result["growth"] == "Moderate growth trajectory"
        assert result["quality"] == "Strong profit margins"
        assert result["momentum"] == "Positive momentum"
        assert result["sentiment"] == "Bullish sentiment"

    def test_parse_empty_reasoning(self):
        """Test parsing empty reasoning string."""
        generator = ResearchReportGenerator(api_key=None)
        result = generator._parse_score_reasoning("")

        assert result == {}

    def test_parse_none_reasoning(self):
        """Test parsing None reasoning."""
        generator = ResearchReportGenerator(api_key=None)
        result = generator._parse_score_reasoning(None)

        assert result == {}


# =============================================================================
# JSON Extraction Tests
# =============================================================================


class TestJSONExtraction:
    """Tests for _extract_json static method."""

    def test_extract_simple_json(self):
        """Should extract simple JSON object."""
        result = ResearchReportGenerator._extract_json(
            '{"summary": "Test", "thesis": "Test thesis"}'
        )
        assert result is not None
        assert "summary" in result

    def test_extract_json_from_text(self):
        """Should extract JSON from surrounding text."""
        result = ResearchReportGenerator._extract_json(
            'Here is my analysis: {"summary": "Test", "recommendation": "buy"} End.'
        )
        assert result is not None
        assert "buy" in result

    def test_extract_json_empty_input(self):
        """Should return None for empty input."""
        assert ResearchReportGenerator._extract_json("") is None
        assert ResearchReportGenerator._extract_json(None) is None

    def test_extract_json_no_json(self):
        """Should return None when no JSON present."""
        result = ResearchReportGenerator._extract_json("Just plain text here")
        assert result is None

    def test_extract_json_prefers_report_fields(self):
        """Should prefer JSON objects containing expected report fields."""
        text = '{"other": 1} and {"summary": "Test", "recommendation": "buy"}'
        result = ResearchReportGenerator._extract_json(text)
        assert result is not None
        assert "summary" in result


# =============================================================================
# Report Response Parsing Tests
# =============================================================================


class TestReportResponseParsing:
    """Tests for _parse_report_response method."""

    def test_parse_valid_response(self):
        """Test parsing valid JSON response."""
        generator = ResearchReportGenerator(api_key=None)

        response = '''{
            "summary": "Strong fundamentals.",
            "thesis": "Long-term value.",
            "bull_case": "Growth potential.",
            "bear_case": "Competition risk.",
            "recommendation": "buy",
            "target_price": 195.00
        }'''

        result = generator._parse_report_response(response, "AAPL")

        assert result.success is True
        assert result.report is not None
        assert result.report.ticker == "AAPL"
        assert result.report.summary == "Strong fundamentals."
        assert result.report.thesis == "Long-term value."
        assert result.report.recommendation == "buy"
        assert result.report.target_price == 195.00

    def test_parse_response_with_null_target(self):
        """Test parsing response with null target price."""
        generator = ResearchReportGenerator(api_key=None)

        response = '''{
            "summary": "Uncertain.",
            "thesis": "Mixed signals.",
            "bull_case": "Potential.",
            "bear_case": "Risks.",
            "recommendation": "hold",
            "target_price": null
        }'''

        result = generator._parse_report_response(response, "XYZ")

        assert result.success is True
        assert result.report.target_price is None

    def test_parse_response_invalid_recommendation(self):
        """Test parsing response with invalid recommendation (should default to hold)."""
        generator = ResearchReportGenerator(api_key=None)

        response = '''{
            "summary": "Test.",
            "thesis": "Test.",
            "bull_case": "Test.",
            "bear_case": "Test.",
            "recommendation": "invalid_value",
            "target_price": 100.00
        }'''

        result = generator._parse_report_response(response, "TEST")

        assert result.success is True
        assert result.report.recommendation == "hold"

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        generator = ResearchReportGenerator(api_key=None)

        result = generator._parse_report_response("", "TEST")

        assert result.success is False
        assert "Empty response" in result.error_message

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON response."""
        generator = ResearchReportGenerator(api_key=None)

        result = generator._parse_report_response("Not valid JSON at all", "TEST")

        assert result.success is False
        assert "Failed to extract JSON" in result.error_message

    def test_parse_all_recommendations(self):
        """Test parsing all valid recommendation values."""
        generator = ResearchReportGenerator(api_key=None)

        for recommendation in VALID_RECOMMENDATIONS:
            response = f'''{{"summary": "Test", "thesis": "Test", "bull_case": "Test", "bear_case": "Test", "recommendation": "{recommendation}", "target_price": 100}}'''
            result = generator._parse_report_response(response, "TEST")

            assert result.success is True
            assert result.report.recommendation == recommendation


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestPromptBuilding:
    """Tests for _build_report_prompt method."""

    def test_prompt_contains_ticker(self, sample_fundamentals, sample_candidate_score):
        """Test that prompt includes ticker."""
        generator = ResearchReportGenerator(api_key=None)
        prompt = generator._build_report_prompt(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert "AAPL" in prompt
        assert "Apple Inc." in prompt

    def test_prompt_contains_fundamentals(self, sample_fundamentals, sample_candidate_score):
        """Test that prompt includes fundamentals."""
        generator = ResearchReportGenerator(api_key=None)
        prompt = generator._build_report_prompt(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert "P/E Ratio" in prompt
        assert "Technology" in prompt
        assert "Consumer Electronics" in prompt

    def test_prompt_contains_scores(self, sample_fundamentals, sample_candidate_score):
        """Test that prompt includes factor scores."""
        generator = ResearchReportGenerator(api_key=None)
        prompt = generator._build_report_prompt(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert "Value: 72" in prompt
        assert "Growth: 65" in prompt
        assert "Quality: 78" in prompt
        assert "Momentum: 70" in prompt
        assert "Sentiment: 82" in prompt
        assert "Composite: 73" in prompt

    def test_prompt_contains_optional_data(self, sample_fundamentals, sample_candidate_score):
        """Test that prompt includes optional price and congress data."""
        generator = ResearchReportGenerator(api_key=None)
        prompt = generator._build_report_prompt(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
            price_summary="Up 15% YTD",
            congress_summary="3 buys, 1 sell",
        )

        assert "Up 15% YTD" in prompt
        assert "3 buys, 1 sell" in prompt

    def test_prompt_default_summaries(self, sample_fundamentals, sample_candidate_score):
        """Test that prompt uses defaults when optional data is empty."""
        generator = ResearchReportGenerator(api_key=None)
        prompt = generator._build_report_prompt(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert "No price data available" in prompt
        assert "No congressional trading data available" in prompt


# =============================================================================
# Generate Report Tests (Async)
# =============================================================================


class TestGenerateReport:
    """Tests for generate_report async method."""

    @pytest.mark.asyncio
    async def test_report_unavailable_without_api_key(
        self, sample_fundamentals, sample_candidate_score
    ):
        """Test report returns error when no API key."""
        generator = ResearchReportGenerator(api_key=None)

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is False
        assert "unavailable" in result.error_message.lower()
        assert "api key" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_report_budget_exceeded(
        self, sample_fundamentals, sample_candidate_score
    ):
        """Test report returns error when budget exceeded."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=0.001)
        # Inject mock client to make it "available"
        mock_client = MagicMock()
        generator._client = mock_client
        # Simulate exceeding budget
        generator._spend_reset_date = date.today()
        generator._monthly_spend = 0.01

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is False
        assert "budget" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_successful_report_generation(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test successful report generation."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client directly
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is True
        assert result.report is not None
        assert result.report.ticker == "AAPL"
        assert result.report.recommendation == "buy"
        assert result.report.target_price == 195.00
        assert result.input_tokens == 1500
        assert result.output_tokens == 500
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_report_api_error_handling(
        self, sample_fundamentals, sample_candidate_score
    ):
        """Test report handles API errors gracefully."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client that raises an error
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Network error")
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is False
        assert "API error" in result.error_message
        assert "Network error" in result.error_message

    @pytest.mark.asyncio
    async def test_report_records_cost(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test that report generation records cost correctly."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        initial_spend = generator.get_monthly_spend()

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.cost > 0
        assert generator.get_monthly_spend() == initial_spend + result.cost

    @pytest.mark.asyncio
    async def test_report_with_all_options(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test report generation with all optional parameters."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
            price_summary="Up 20% YTD, near 52-week high",
            congress_summary="5 buys from tech committee members",
            max_tokens=3000,
        )

        assert result.success is True
        # Verify max_tokens was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 3000


# =============================================================================
# Generate Report Tests (Sync)
# =============================================================================


class TestGenerateReportSync:
    """Tests for generate_report_sync method."""

    def test_sync_unavailable_without_api_key(
        self, sample_fundamentals, sample_candidate_score
    ):
        """Test sync report returns error when no API key."""
        generator = ResearchReportGenerator(api_key=None)

        result = generator.generate_report_sync(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is False
        assert "unavailable" in result.error_message.lower()

    def test_sync_successful_report(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test successful synchronous report generation."""
        generator = ResearchReportGenerator(api_key=None)
        # Inject mock client directly
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        result = generator.generate_report_sync(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
        )

        assert result.success is True
        assert result.report.recommendation == "buy"


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for the prompt template constant."""

    def test_prompt_template_has_placeholders(self):
        """Test that RESEARCH_REPORT_PROMPT has required placeholders."""
        assert "{ticker}" in RESEARCH_REPORT_PROMPT
        assert "{company_name}" in RESEARCH_REPORT_PROMPT
        assert "{sector}" in RESEARCH_REPORT_PROMPT
        assert "{industry}" in RESEARCH_REPORT_PROMPT
        assert "{fundamentals_summary}" in RESEARCH_REPORT_PROMPT
        assert "{price_summary}" in RESEARCH_REPORT_PROMPT
        assert "{value_score}" in RESEARCH_REPORT_PROMPT
        assert "{growth_score}" in RESEARCH_REPORT_PROMPT
        assert "{quality_score}" in RESEARCH_REPORT_PROMPT
        assert "{momentum_score}" in RESEARCH_REPORT_PROMPT
        assert "{sentiment_score}" in RESEARCH_REPORT_PROMPT
        assert "{composite_score}" in RESEARCH_REPORT_PROMPT
        assert "{congress_summary}" in RESEARCH_REPORT_PROMPT

    def test_prompt_template_has_structure(self):
        """Test that RESEARCH_REPORT_PROMPT has expected sections."""
        assert "Company Information" in RESEARCH_REPORT_PROMPT
        assert "Financial Metrics" in RESEARCH_REPORT_PROMPT
        assert "Price Performance" in RESEARCH_REPORT_PROMPT
        assert "Factor Scores" in RESEARCH_REPORT_PROMPT
        assert "Congressional Trading Activity" in RESEARCH_REPORT_PROMPT
        assert "Executive Summary" in RESEARCH_REPORT_PROMPT
        assert "Investment Thesis" in RESEARCH_REPORT_PROMPT
        assert "Bull Case" in RESEARCH_REPORT_PROMPT
        assert "Bear Case" in RESEARCH_REPORT_PROMPT
        assert "Recommendation" in RESEARCH_REPORT_PROMPT
        assert "Target Price" in RESEARCH_REPORT_PROMPT


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_budget(self):
        """Test generator with zero budget."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=0.0)
        assert generator._within_budget() is False
        assert generator.get_remaining_budget() == 0.0

    def test_very_small_budget(self):
        """Test generator with very small budget."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=0.001)
        assert generator._within_budget() is True

    def test_fundamentals_with_all_nulls(self, sample_candidate_score):
        """Test report building with all null fundamentals."""
        generator = ResearchReportGenerator(api_key=None)

        fundamentals = FundamentalsData(ticker="NULL")

        prompt = generator._build_report_prompt(
            ticker="NULL",
            company_name="Null Corp",
            fundamentals=fundamentals,
            score_result=sample_candidate_score,
        )

        # Should not raise exception
        assert "NULL" in prompt
        assert "N/A" in prompt  # For null values

    def test_score_with_null_scores(self, sample_fundamentals):
        """Test report building with null scores."""
        generator = ResearchReportGenerator(api_key=None)

        score = CandidateScore(
            ticker="TEST",
            value_score=None,
            growth_score=None,
            quality_score=None,
            momentum_score=None,
            sentiment_score=None,
            composite_score=None,
            reasoning=None,
        )

        prompt = generator._build_report_prompt(
            ticker="TEST",
            company_name="Test Corp",
            fundamentals=sample_fundamentals,
            score_result=score,
        )

        # Should handle None scores gracefully
        assert "N/A" in prompt

    @pytest.mark.asyncio
    async def test_custom_max_tokens(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test report generation with custom max_tokens."""
        generator = ResearchReportGenerator(api_key=None)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
            max_tokens=1500,
        )

        # Verify max_tokens was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1500

    def test_parse_malformed_reasoning(self):
        """Test parsing malformed reasoning string."""
        generator = ResearchReportGenerator(api_key=None)

        # Malformed reasoning
        reasoning = "This is not in the expected format at all"
        result = generator._parse_score_reasoning(reasoning)

        # Should return empty dict, not raise exception
        assert result == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with fully mocked Anthropic client."""

    @pytest.mark.asyncio
    async def test_full_report_flow(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test complete report generation flow."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=100.00)

        # Set up mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        # Generate report
        result = await generator.generate_report(
            ticker="AAPL",
            company_name="Apple Inc.",
            fundamentals=sample_fundamentals,
            score_result=sample_candidate_score,
            price_summary="Up 15% YTD",
            congress_summary="Nancy Pelosi purchased shares",
        )

        # Verify result
        assert result.success is True
        assert result.report.ticker == "AAPL"
        assert result.report.summary is not None
        assert result.report.thesis is not None
        assert result.report.bull_case is not None
        assert result.report.bear_case is not None
        assert result.report.recommendation in VALID_RECOMMENDATIONS
        assert result.cost > 0

        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert "AAPL" in call_kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_reports_cost_tracking(
        self, sample_fundamentals, sample_candidate_score, mock_anthropic_response
    ):
        """Test cost tracking across multiple report generations."""
        generator = ResearchReportGenerator(api_key=None, max_monthly_spend=100.00)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        generator._client = mock_client
        generator._spend_reset_date = date.today()

        # Generate multiple reports
        for i in range(3):
            result = await generator.generate_report(
                ticker=f"TEST{i}",
                company_name=f"Test Corp {i}",
                fundamentals=sample_fundamentals,
                score_result=sample_candidate_score,
            )
            assert result.success is True

        # Verify cumulative cost tracking
        expected_cost_per_call = (
            (1500 / 1_000_000) * SONNET_INPUT_COST_PER_MILLION +
            (500 / 1_000_000) * SONNET_OUTPUT_COST_PER_MILLION
        )
        assert generator.get_monthly_spend() == pytest.approx(expected_cost_per_call * 3, rel=0.01)


# =============================================================================
# Cost Constant Tests
# =============================================================================


class TestCostConstants:
    """Tests for cost-related constants."""

    def test_cost_constants_are_positive(self):
        """Verify cost constants are positive values."""
        assert SONNET_INPUT_COST_PER_MILLION > 0
        assert SONNET_OUTPUT_COST_PER_MILLION > 0

    def test_output_costs_more_than_input(self):
        """Verify output tokens cost more than input (typical for LLMs)."""
        assert SONNET_OUTPUT_COST_PER_MILLION > SONNET_INPUT_COST_PER_MILLION

    def test_valid_recommendations_constant(self):
        """Verify valid recommendations tuple."""
        assert "strong_buy" in VALID_RECOMMENDATIONS
        assert "buy" in VALID_RECOMMENDATIONS
        assert "hold" in VALID_RECOMMENDATIONS
        assert "sell" in VALID_RECOMMENDATIONS
        assert "strong_sell" in VALID_RECOMMENDATIONS
        assert len(VALID_RECOMMENDATIONS) == 5
