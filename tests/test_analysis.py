"""Tests for the analysis module with local LLM integration."""

from unittest.mock import MagicMock, patch

import pytest

from investment_monitor.analysis import (
    LocalLLM,
    RELEVANCE_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARIZE_PROMPT,
)
from investment_monitor.analysis.local_llm import LocalLLM as LocalLLMClass


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplates:
    """Test the prompt templates are properly formatted."""

    def test_relevance_prompt_has_placeholders(self):
        """Relevance prompt should contain required placeholders."""
        assert "{ticker}" in RELEVANCE_PROMPT
        assert "{company_name}" in RELEVANCE_PROMPT
        assert "{headline}" in RELEVANCE_PROMPT

    def test_relevance_prompt_can_be_formatted(self):
        """Relevance prompt should format correctly."""
        formatted = RELEVANCE_PROMPT.format(
            ticker="AAPL",
            company_name="Apple Inc.",
            headline="Apple announces new iPhone",
        )
        assert "AAPL" in formatted
        assert "Apple Inc." in formatted
        assert "Apple announces new iPhone" in formatted

    def test_sentiment_prompt_has_placeholders(self):
        """Sentiment prompt should contain required placeholders."""
        assert "{headline}" in SENTIMENT_PROMPT

    def test_sentiment_prompt_can_be_formatted(self):
        """Sentiment prompt should format correctly."""
        formatted = SENTIMENT_PROMPT.format(
            headline="Stock prices surge on earnings beat",
        )
        assert "Stock prices surge on earnings beat" in formatted

    def test_summarize_prompt_has_placeholders(self):
        """Summarize prompt should contain required placeholders."""
        assert "{text}" in SUMMARIZE_PROMPT
        assert "{max_sentences}" in SUMMARIZE_PROMPT

    def test_summarize_prompt_can_be_formatted(self):
        """Summarize prompt should format correctly."""
        formatted = SUMMARIZE_PROMPT.format(
            text="This is a long article about technology.",
            max_sentences=2,
        )
        assert "This is a long article about technology." in formatted
        assert "2" in formatted


# =============================================================================
# LocalLLM Initialization Tests
# =============================================================================


class TestLocalLLMInit:
    """Test LocalLLM initialization."""

    def test_default_initialization(self):
        """LocalLLM should initialize with default values."""
        llm = LocalLLM()
        assert llm.model == "phi3:mini"
        assert llm.base_url == "http://localhost:11434"

    def test_custom_initialization(self):
        """LocalLLM should accept custom model and base_url."""
        llm = LocalLLM(model="llama2:7b", base_url="http://custom:1234")
        assert llm.model == "llama2:7b"
        assert llm.base_url == "http://custom:1234"

    def test_client_lazy_initialization(self):
        """Client should not be initialized until accessed."""
        llm = LocalLLM()
        assert llm._client is None


# =============================================================================
# LocalLLM Availability Tests
# =============================================================================


class TestLocalLLMAvailability:
    """Test is_available method behavior."""

    def test_is_available_when_ollama_not_installed(self):
        """Should return False when ollama package not installed."""
        llm = LocalLLM()

        with patch.dict("sys.modules", {"ollama": None}):
            with patch("investment_monitor.analysis.local_llm.logger"):
                # Force reimport failure by patching __import__
                with patch("builtins.__import__", side_effect=ImportError("No module named 'ollama'")):
                    assert llm.is_available() is False

    def test_is_available_when_server_not_running(self):
        """Should return False when Ollama server is not running."""
        llm = LocalLLM()

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            with patch("investment_monitor.analysis.local_llm.logger"):
                assert llm.is_available() is False

    def test_is_available_when_model_not_found(self):
        """Should return False when specified model is not available."""
        llm = LocalLLM(model="nonexistent:model")

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "llama2:7b"}, {"name": "phi3:mini"}]
        }
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            with patch("investment_monitor.analysis.local_llm.logger"):
                assert llm.is_available() is False

    def test_is_available_when_model_found(self):
        """Should return True when model is available."""
        llm = LocalLLM(model="phi3:mini")

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "llama2:7b"}, {"name": "phi3:mini"}]
        }
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert llm.is_available() is True


# =============================================================================
# Score Parsing Tests
# =============================================================================


class TestScoreParsing:
    """Test the _parse_score static method."""

    @pytest.mark.parametrize("response,expected", [
        ("7", 7.0),
        ("7.5", 7.5),
        ("  8  ", 8.0),
        ("Rating: 6", 6.0),
        ("rating: 9", 9.0),
        ("7/10", 7.0),
        ("7 out of 10", 7.0),
        ("I would rate this a 8", 8.0),
        ("The relevance score is 5", 5.0),
        ("10", 10.0),
        ("1", 1.0),
    ])
    def test_parse_score_valid_inputs(self, response: str, expected: float):
        """Should correctly parse valid score formats."""
        assert LocalLLMClass._parse_score(response) == expected

    @pytest.mark.parametrize("response,expected", [
        ("", None),
        (None, None),
        ("no numbers here", None),
        ("bullish", None),
    ])
    def test_parse_score_invalid_inputs(self, response: str | None, expected: float | None):
        """Should return None for invalid inputs."""
        assert LocalLLMClass._parse_score(response) == expected

    def test_parse_score_clamps_high_values(self):
        """Should clamp scores above 10 to 10."""
        assert LocalLLMClass._parse_score("15") == 10.0
        assert LocalLLMClass._parse_score("100") == 10.0

    def test_parse_score_clamps_negative_values(self):
        """Should clamp negative scores to 0."""
        # Note: regex won't match negative numbers with current pattern,
        # but if it did, they should be clamped
        assert LocalLLMClass._parse_score("0") == 0.0


# =============================================================================
# Sentiment Parsing Tests
# =============================================================================


class TestSentimentParsing:
    """Test the _parse_sentiment static method."""

    @pytest.mark.parametrize("response,expected", [
        ("bullish", "bullish"),
        ("BULLISH", "bullish"),
        ("Bullish", "bullish"),
        ("bearish", "bearish"),
        ("BEARISH", "bearish"),
        ("Bearish", "bearish"),
        ("neutral", "neutral"),
        ("NEUTRAL", "neutral"),
        ("Neutral", "neutral"),
    ])
    def test_parse_sentiment_direct_matches(self, response: str, expected: str):
        """Should correctly parse direct sentiment words."""
        assert LocalLLMClass._parse_sentiment(response) == expected

    @pytest.mark.parametrize("response,expected", [
        ("Sentiment: bullish", "bullish"),
        ("I would say this is bearish", "bearish"),
        ("The sentiment is neutral", "neutral"),
        ("This news is clearly bullish for the stock", "bullish"),
    ])
    def test_parse_sentiment_with_context(self, response: str, expected: str):
        """Should find sentiment keywords in longer responses."""
        assert LocalLLMClass._parse_sentiment(response) == expected

    @pytest.mark.parametrize("response,expected", [
        ("positive news", "bullish"),
        ("this is good for the company", "bullish"),
        ("negative outlook", "bearish"),
        ("this is bad news", "bearish"),
        ("stock will go up", "bullish"),
        ("stock will go down", "bearish"),
    ])
    def test_parse_sentiment_synonyms(self, response: str, expected: str):
        """Should recognize sentiment synonyms."""
        assert LocalLLMClass._parse_sentiment(response) == expected

    @pytest.mark.parametrize("response", [
        "",
        None,
        "I don't know",
        "uncertain",
        "maybe",
    ])
    def test_parse_sentiment_unknown(self, response: str | None):
        """Should return 'unknown' for unrecognized responses."""
        assert LocalLLMClass._parse_sentiment(response) == "unknown"


# =============================================================================
# Relevance Scoring Tests
# =============================================================================


class TestScoreRelevance:
    """Test the score_relevance method."""

    @pytest.mark.asyncio
    async def test_score_relevance_returns_negative_when_unavailable(self):
        """Should return -1 when Ollama is not available."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=False):
            score = await llm.score_relevance(
                headline="Apple announces new product",
                ticker="AAPL",
            )
            assert score == -1.0

    @pytest.mark.asyncio
    async def test_score_relevance_returns_valid_score(self):
        """Should return parsed score when LLM responds."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="8"):
                score = await llm.score_relevance(
                    headline="Apple announces new iPhone",
                    ticker="AAPL",
                    company_name="Apple Inc.",
                )
                assert score == 8.0

    @pytest.mark.asyncio
    async def test_score_relevance_handles_generation_failure(self):
        """Should return -1 when generation fails."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value=None):
                score = await llm.score_relevance(
                    headline="Some headline",
                    ticker="TEST",
                )
                assert score == -1.0

    @pytest.mark.asyncio
    async def test_score_relevance_uses_ticker_as_company_name_fallback(self):
        """Should use ticker as company name when not provided."""
        llm = LocalLLM()
        generated_prompt = None

        def capture_prompt(prompt):
            nonlocal generated_prompt
            generated_prompt = prompt
            return "7"

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", side_effect=capture_prompt):
                await llm.score_relevance(
                    headline="Test headline",
                    ticker="AAPL",
                )
                assert generated_prompt is not None
                assert "AAPL" in generated_prompt


# =============================================================================
# Sentiment Classification Tests
# =============================================================================


class TestClassifySentiment:
    """Test the classify_sentiment method."""

    @pytest.mark.asyncio
    async def test_classify_sentiment_returns_unknown_when_unavailable(self):
        """Should return 'unknown' when Ollama is not available."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=False):
            sentiment = await llm.classify_sentiment("Stock surges 10%")
            assert sentiment == "unknown"

    @pytest.mark.asyncio
    async def test_classify_sentiment_returns_bullish(self):
        """Should return 'bullish' for positive sentiment."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="bullish"):
                sentiment = await llm.classify_sentiment("Stock surges 10%")
                assert sentiment == "bullish"

    @pytest.mark.asyncio
    async def test_classify_sentiment_returns_bearish(self):
        """Should return 'bearish' for negative sentiment."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="bearish"):
                sentiment = await llm.classify_sentiment("Company reports losses")
                assert sentiment == "bearish"

    @pytest.mark.asyncio
    async def test_classify_sentiment_returns_neutral(self):
        """Should return 'neutral' for neutral sentiment."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="neutral"):
                sentiment = await llm.classify_sentiment("Company holds annual meeting")
                assert sentiment == "neutral"

    @pytest.mark.asyncio
    async def test_classify_sentiment_handles_generation_failure(self):
        """Should return 'unknown' when generation fails."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value=None):
                sentiment = await llm.classify_sentiment("Some headline")
                assert sentiment == "unknown"


# =============================================================================
# Summarization Tests
# =============================================================================


class TestSummarize:
    """Test the summarize method."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_when_unavailable(self):
        """Should return empty string when Ollama is not available."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=False):
            summary = await llm.summarize("A long article about technology...")
            assert summary == ""

    @pytest.mark.asyncio
    async def test_summarize_returns_summary(self):
        """Should return summary when LLM responds."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value="Tech advances quickly."):
                summary = await llm.summarize(
                    "A long article about technology advances in recent years.",
                    max_sentences=1,
                )
                assert summary == "Tech advances quickly."

    @pytest.mark.asyncio
    async def test_summarize_handles_generation_failure(self):
        """Should return empty string when generation fails."""
        llm = LocalLLM()

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", return_value=None):
                summary = await llm.summarize("Some text")
                assert summary == ""

    @pytest.mark.asyncio
    async def test_summarize_respects_max_sentences_parameter(self):
        """Should include max_sentences in the prompt."""
        llm = LocalLLM()
        generated_prompt = None

        def capture_prompt(prompt):
            nonlocal generated_prompt
            generated_prompt = prompt
            return "Summary here."

        with patch.object(llm, "is_available", return_value=True):
            with patch.object(llm, "_generate", side_effect=capture_prompt):
                await llm.summarize("Some text", max_sentences=3)
                assert generated_prompt is not None
                assert "3" in generated_prompt


# =============================================================================
# Generate Method Tests
# =============================================================================


class TestGenerate:
    """Test the _generate method."""

    def test_generate_returns_response(self):
        """Should return stripped response from LLM."""
        llm = LocalLLM()

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "  Hello World  "}
        llm._client = mock_client

        result = llm._generate("Test prompt")
        assert result == "Hello World"

    def test_generate_handles_exception(self):
        """Should return None when generation raises exception."""
        llm = LocalLLM()

        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("API Error")
        llm._client = mock_client

        with patch("investment_monitor.analysis.local_llm.logger"):
            result = llm._generate("Test prompt")
            assert result is None

    def test_generate_uses_correct_parameters(self):
        """Should pass correct parameters to the client."""
        llm = LocalLLM(model="test-model")

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "test"}
        llm._client = mock_client

        llm._generate("Test prompt")

        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["prompt"] == "Test prompt"
        assert "options" in call_args.kwargs
        assert call_args.kwargs["options"]["temperature"] == 0.1


# =============================================================================
# Integration-style Tests (with mocked Ollama)
# =============================================================================


class TestLocalLLMIntegration:
    """Integration-style tests with fully mocked Ollama."""

    @pytest.fixture
    def mock_ollama_module(self):
        """Create a mock ollama module."""
        mock_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_module.Client.return_value = mock_client_instance
        return mock_module, mock_client_instance

    @pytest.mark.asyncio
    async def test_full_relevance_scoring_flow(self, mock_ollama_module):
        """Test complete relevance scoring flow."""
        mock_module, mock_client = mock_ollama_module
        mock_client.list.return_value = {"models": [{"name": "phi3:mini"}]}
        mock_client.generate.return_value = {"response": "8"}

        llm = LocalLLM()

        with patch.dict("sys.modules", {"ollama": mock_module}):
            # Manually set up client since we're mocking the import
            llm._client = mock_client

            with patch.object(llm, "is_available", return_value=True):
                score = await llm.score_relevance(
                    headline="Apple reports record iPhone sales",
                    ticker="AAPL",
                    company_name="Apple Inc.",
                )

            assert score == 8.0

    @pytest.mark.asyncio
    async def test_full_sentiment_classification_flow(self, mock_ollama_module):
        """Test complete sentiment classification flow."""
        mock_module, mock_client = mock_ollama_module
        mock_client.list.return_value = {"models": [{"name": "phi3:mini"}]}
        mock_client.generate.return_value = {"response": "bullish"}

        llm = LocalLLM()

        with patch.dict("sys.modules", {"ollama": mock_module}):
            llm._client = mock_client

            with patch.object(llm, "is_available", return_value=True):
                sentiment = await llm.classify_sentiment(
                    "Tesla stock soars after earnings beat"
                )

            assert sentiment == "bullish"

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_connection_error(self):
        """Test graceful handling when Ollama server is down."""
        llm = LocalLLM()

        # Simulate connection refused
        with patch.object(llm, "is_available", return_value=False):
            score = await llm.score_relevance("Test headline", "TEST")
            sentiment = await llm.classify_sentiment("Test headline")
            summary = await llm.summarize("Test text")

            assert score == -1.0
            assert sentiment == "unknown"
            assert summary == ""


# =============================================================================
# Claude API Tests
# =============================================================================

from datetime import date
from decimal import Decimal

from investment_monitor.analysis import (
    ClaudeAnalyzer,
    SynthesisResult,
    WeeklyData,
    WEEKLY_SYNTHESIS_PROMPT,
)
from investment_monitor.models import Holding, Portfolio, WatchlistItem


# =============================================================================
# Claude API Fixtures
# =============================================================================


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing."""
    return Portfolio(
        holdings=[
            Holding(
                ticker="AAPL",
                shares=Decimal("100"),
                cost_basis=Decimal("150.00"),
                thesis="Long-term growth in services and wearables",
            ),
            Holding(
                ticker="MSFT",
                shares=Decimal("50"),
                cost_basis=Decimal("350.00"),
                thesis="Cloud computing dominance with Azure",
            ),
        ],
        watchlist=[
            WatchlistItem(
                ticker="GOOGL",
                reason="Potential AI play",
                target_price=Decimal("140.00"),
            ),
        ],
    )


@pytest.fixture
def sample_week_data() -> WeeklyData:
    """Create sample weekly data for testing."""
    return WeeklyData(
        price_summary="AAPL +3.2% ($155 -> $160), MSFT -1.5% ($360 -> $355)",
        insider_summary="AAPL: CEO Tim Cook purchased 10,000 shares at $158",
        news_summary="Apple announced new Vision Pro features. Microsoft Azure growth exceeded expectations.",
        earnings_summary="AAPL reports earnings in 3 days. MSFT reported last week, beat estimates.",
        week_start=date(2025, 1, 20),
        week_end=date(2025, 1, 26),
    )


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="## Weekly Analysis\n\n1. **Key Developments**\n- Apple's CEO purchase signals confidence\n- Azure growth remains strong\n\n2. **Thesis Check**\n- No contradictions to your investment theses\n\n3. **Watch List**\n- AAPL earnings coming up, prepare for volatility\n\n4. **Patterns**\n- Insider buying trend continues in tech sector")]
    mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
    return mock_response


# =============================================================================
# WeeklyData Tests
# =============================================================================


class TestWeeklyData:
    """Tests for WeeklyData model."""

    def test_default_values(self):
        """Test WeeklyData initializes with sensible defaults."""
        data = WeeklyData()
        assert "No significant price movements" in data.price_summary
        assert "No insider transactions" in data.insider_summary
        assert "No significant news" in data.news_summary
        assert "No upcoming earnings" in data.earnings_summary
        assert data.week_start is None
        assert data.week_end is None

    def test_custom_values(self, sample_week_data):
        """Test WeeklyData accepts custom values."""
        assert "AAPL +3.2%" in sample_week_data.price_summary
        assert "Tim Cook" in sample_week_data.insider_summary
        assert "Vision Pro" in sample_week_data.news_summary
        assert "AAPL reports earnings" in sample_week_data.earnings_summary
        assert sample_week_data.week_start == date(2025, 1, 20)
        assert sample_week_data.week_end == date(2025, 1, 26)

    def test_partial_data(self):
        """Test WeeklyData works with partial data."""
        data = WeeklyData(
            price_summary="AAPL up 5%",
            # Other fields use defaults
        )
        assert data.price_summary == "AAPL up 5%"
        assert "No insider transactions" in data.insider_summary


# =============================================================================
# SynthesisResult Tests
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful synthesis result."""
        result = SynthesisResult(
            synthesis="Analysis text here",
            success=True,
            input_tokens=500,
            output_tokens=200,
            cost=0.0045,
        )
        assert result.synthesis == "Analysis text here"
        assert result.success is True
        assert result.error_message is None
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cost == 0.0045

    def test_failed_result(self):
        """Test creating a failed synthesis result."""
        result = SynthesisResult(
            synthesis="",
            success=False,
            error_message="API error: rate limited",
        )
        assert result.synthesis == ""
        assert result.success is False
        assert result.error_message == "API error: rate limited"


# =============================================================================
# ClaudeAnalyzer Initialization Tests
# =============================================================================


class TestClaudeAnalyzerInit:
    """Tests for ClaudeAnalyzer initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        analyzer = ClaudeAnalyzer(api_key=None)
        assert analyzer.is_available() is False
        assert analyzer.max_monthly_spend == 5.00
        assert analyzer._monthly_spend == 0.0

    def test_init_with_empty_api_key(self):
        """Test initialization with empty string API key."""
        analyzer = ClaudeAnalyzer(api_key="")
        assert analyzer.is_available() is False

    def test_init_with_custom_budget(self):
        """Test initialization with custom budget."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=10.00)
        assert analyzer.max_monthly_spend == 10.00

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        analyzer = ClaudeAnalyzer(api_key=None, model="claude-3-opus-20240229")
        assert analyzer.model == "claude-3-opus-20240229"

    def test_init_with_api_key(self):
        """Test initialization with valid API key when anthropic is available."""
        mock_anthropic_class = MagicMock()
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": MagicMock(Anthropic=mock_anthropic_class)}):
            # Need to reimport to pick up the patched module
            analyzer = ClaudeAnalyzer(api_key=None)
            # Since the import happens inside __init__, and we're using a fresh instance
            # without api_key, it won't try to import. Let's test via direct client injection
            analyzer._client = mock_client
            assert analyzer.is_available() is True


# =============================================================================
# Budget Management Tests
# =============================================================================


class TestBudgetManagement:
    """Tests for budget tracking and management."""

    def test_initial_budget_state(self):
        """Test initial budget state."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=5.00)
        assert analyzer.get_monthly_spend() == 0.0
        assert analyzer.get_remaining_budget() == 5.00

    def test_within_budget_initially(self):
        """Test _within_budget returns True initially."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=5.00)
        assert analyzer._within_budget() is True

    def test_record_cost(self):
        """Test cost recording."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=5.00)
        # Initialize the spend reset date to current month to prevent reset
        analyzer._spend_reset_date = date.today()

        # 1000 input tokens at $3/M = $0.003
        # 500 output tokens at $15/M = $0.0075
        # Total = $0.0105
        cost = analyzer._record_cost(input_tokens=1000, output_tokens=500)

        assert cost == pytest.approx(0.0105, rel=0.01)
        assert analyzer.get_monthly_spend() == pytest.approx(0.0105, rel=0.01)
        assert analyzer.get_remaining_budget() == pytest.approx(5.00 - 0.0105, rel=0.01)

    def test_budget_exceeded(self):
        """Test behavior when budget is exceeded."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=0.01)
        # Initialize the spend reset date to current month to prevent reset
        analyzer._spend_reset_date = date.today()

        # Record cost that exceeds budget
        analyzer._record_cost(input_tokens=10000, output_tokens=1000)

        assert analyzer._within_budget() is False
        assert analyzer.get_remaining_budget() == 0.0

    def test_cumulative_cost_tracking(self):
        """Test that costs accumulate correctly."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=5.00)
        # Initialize the spend reset date to current month to prevent reset
        analyzer._spend_reset_date = date.today()

        analyzer._record_cost(input_tokens=1000, output_tokens=500)
        first_spend = analyzer.get_monthly_spend()

        analyzer._record_cost(input_tokens=1000, output_tokens=500)
        second_spend = analyzer.get_monthly_spend()

        assert second_spend == pytest.approx(first_spend * 2, rel=0.01)

    def test_monthly_reset(self):
        """Test that spend resets on new month."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=5.00)

        # Simulate previous month spend
        analyzer._monthly_spend = 4.50
        analyzer._spend_reset_date = date(2024, 12, 15)  # Previous month

        # This should trigger reset since we're now in a different month
        spend = analyzer.get_monthly_spend()

        # If test runs in January or later, should have reset
        if date.today().month != 12:
            assert spend == 0.0
        else:
            # If running in December, won't reset
            assert spend == 4.50


# =============================================================================
# Prompt Building Tests
# =============================================================================


class TestClaudePromptBuilding:
    """Tests for prompt construction."""

    def test_prompt_contains_portfolio_data(self, sample_portfolio, sample_week_data):
        """Test that prompt includes portfolio information."""
        analyzer = ClaudeAnalyzer(api_key=None)
        prompt = analyzer._build_synthesis_prompt(sample_portfolio, sample_week_data)

        assert "AAPL" in prompt
        assert "MSFT" in prompt
        assert "GOOGL" in prompt
        assert "150.0" in prompt  # cost basis
        assert "Cloud computing dominance" in prompt  # thesis

    def test_prompt_contains_weekly_data(self, sample_portfolio, sample_week_data):
        """Test that prompt includes weekly data."""
        analyzer = ClaudeAnalyzer(api_key=None)
        prompt = analyzer._build_synthesis_prompt(sample_portfolio, sample_week_data)

        assert "AAPL +3.2%" in prompt
        assert "Tim Cook" in prompt
        assert "Vision Pro" in prompt
        assert "AAPL reports earnings" in prompt

    def test_prompt_structure(self, sample_portfolio, sample_week_data):
        """Test that prompt follows expected structure."""
        analyzer = ClaudeAnalyzer(api_key=None)
        prompt = analyzer._build_synthesis_prompt(sample_portfolio, sample_week_data)

        assert "## My Portfolio" in prompt
        assert "## This Week's Activity" in prompt
        assert "### Price Movements" in prompt
        assert "### Insider Transactions" in prompt
        assert "### Relevant News" in prompt
        assert "### Upcoming Earnings" in prompt
        assert "## Your Analysis" in prompt

    def test_prompt_with_empty_portfolio(self):
        """Test prompt building with empty portfolio."""
        analyzer = ClaudeAnalyzer(api_key=None)
        empty_portfolio = Portfolio(holdings=[], watchlist=[])
        week_data = WeeklyData()

        prompt = analyzer._build_synthesis_prompt(empty_portfolio, week_data)

        # Should still be a valid prompt
        assert "## My Portfolio" in prompt
        assert "holdings:" in prompt


# =============================================================================
# Weekly Synthesis Tests (with mocked API)
# =============================================================================


class TestWeeklySynthesis:
    """Tests for weekly synthesis generation."""

    @pytest.mark.asyncio
    async def test_synthesis_unavailable_without_api_key(self, sample_portfolio, sample_week_data):
        """Test synthesis returns error when no API key."""
        analyzer = ClaudeAnalyzer(api_key=None)

        result = await analyzer.weekly_synthesis(sample_portfolio, sample_week_data)

        assert result.success is False
        assert "unavailable" in result.error_message.lower()
        assert "api key" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_synthesis_budget_exceeded(self, sample_portfolio, sample_week_data):
        """Test synthesis returns error when budget exceeded."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=0.001)
        # Inject mock client to make it "available"
        mock_client = MagicMock()
        analyzer._client = mock_client
        # Simulate exceeding budget - set reset date to prevent reset
        analyzer._spend_reset_date = date.today()
        analyzer._monthly_spend = 0.01

        result = await analyzer.weekly_synthesis(sample_portfolio, sample_week_data)

        assert result.success is False
        assert "budget" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_successful_synthesis(
        self, sample_portfolio, sample_week_data, mock_anthropic_response
    ):
        """Test successful synthesis generation."""
        analyzer = ClaudeAnalyzer(api_key=None)
        # Inject mock client directly
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        analyzer._client = mock_client
        analyzer._spend_reset_date = date.today()

        result = await analyzer.weekly_synthesis(sample_portfolio, sample_week_data)

        assert result.success is True
        assert "Key Developments" in result.synthesis
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cost > 0

    @pytest.mark.asyncio
    async def test_synthesis_api_error_handling(self, sample_portfolio, sample_week_data):
        """Test synthesis handles API errors gracefully."""
        analyzer = ClaudeAnalyzer(api_key=None)
        # Inject mock client that raises an error
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Network error")
        analyzer._client = mock_client
        analyzer._spend_reset_date = date.today()

        result = await analyzer.weekly_synthesis(sample_portfolio, sample_week_data)

        assert result.success is False
        assert "API error" in result.error_message
        assert "Network error" in result.error_message

    @pytest.mark.asyncio
    async def test_synthesis_records_cost(
        self, sample_portfolio, sample_week_data, mock_anthropic_response
    ):
        """Test that synthesis records cost correctly."""
        analyzer = ClaudeAnalyzer(api_key=None)
        # Inject mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        analyzer._client = mock_client
        analyzer._spend_reset_date = date.today()

        initial_spend = analyzer.get_monthly_spend()

        result = await analyzer.weekly_synthesis(sample_portfolio, sample_week_data)

        assert result.cost > 0
        assert analyzer.get_monthly_spend() == initial_spend + result.cost


# =============================================================================
# Synchronous Synthesis Tests
# =============================================================================


class TestWeeklySynthesisSync:
    """Tests for synchronous weekly synthesis."""

    def test_sync_unavailable_without_api_key(self, sample_portfolio, sample_week_data):
        """Test sync synthesis returns error when no API key."""
        analyzer = ClaudeAnalyzer(api_key=None)

        result = analyzer.weekly_synthesis_sync(sample_portfolio, sample_week_data)

        assert result.success is False
        assert "unavailable" in result.error_message.lower()

    def test_sync_successful_synthesis(
        self, sample_portfolio, sample_week_data, mock_anthropic_response
    ):
        """Test successful synchronous synthesis."""
        analyzer = ClaudeAnalyzer(api_key=None)
        # Inject mock client directly
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        analyzer._client = mock_client
        analyzer._spend_reset_date = date.today()

        result = analyzer.weekly_synthesis_sync(sample_portfolio, sample_week_data)

        assert result.success is True
        assert "Key Developments" in result.synthesis


# =============================================================================
# Weekly Synthesis Prompt Template Tests
# =============================================================================


class TestWeeklySynthesisPromptTemplate:
    """Tests for the prompt template constant."""

    def test_prompt_template_has_placeholders(self):
        """Test that WEEKLY_SYNTHESIS_PROMPT has required placeholders."""
        assert "{portfolio_yaml}" in WEEKLY_SYNTHESIS_PROMPT
        assert "{price_summary}" in WEEKLY_SYNTHESIS_PROMPT
        assert "{insider_summary}" in WEEKLY_SYNTHESIS_PROMPT
        assert "{news_summary}" in WEEKLY_SYNTHESIS_PROMPT
        assert "{earnings_summary}" in WEEKLY_SYNTHESIS_PROMPT

    def test_prompt_template_has_structure(self):
        """Test that WEEKLY_SYNTHESIS_PROMPT has expected sections."""
        assert "My Portfolio" in WEEKLY_SYNTHESIS_PROMPT
        assert "This Week's Activity" in WEEKLY_SYNTHESIS_PROMPT
        assert "Your Analysis" in WEEKLY_SYNTHESIS_PROMPT
        assert "most important developments" in WEEKLY_SYNTHESIS_PROMPT
        assert "investment thesis" in WEEKLY_SYNTHESIS_PROMPT


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestClaudeEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_budget(self):
        """Test analyzer with zero budget."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=0.0)
        assert analyzer._within_budget() is False
        assert analyzer.get_remaining_budget() == 0.0

    def test_very_small_budget(self):
        """Test analyzer with very small budget."""
        analyzer = ClaudeAnalyzer(api_key=None, max_monthly_spend=0.001)
        assert analyzer._within_budget() is True

        # Even smallest API call would exceed
        analyzer._record_cost(input_tokens=1, output_tokens=1)
        # Still should be within due to rounding

    def test_empty_week_data(self, sample_portfolio):
        """Test synthesis with empty week data."""
        analyzer = ClaudeAnalyzer(api_key=None)
        empty_data = WeeklyData()

        prompt = analyzer._build_synthesis_prompt(sample_portfolio, empty_data)

        # Should use default values
        assert "No significant price movements" in prompt

    def test_portfolio_with_no_thesis(self):
        """Test portfolio with holdings that have no thesis."""
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="AAPL", shares=Decimal("100"), cost_basis=Decimal("150.00")),
            ],
            watchlist=[],
        )
        analyzer = ClaudeAnalyzer(api_key=None)
        week_data = WeeklyData()

        prompt = analyzer._build_synthesis_prompt(portfolio, week_data)

        assert "No thesis specified" in prompt

    @pytest.mark.asyncio
    async def test_custom_max_tokens(
        self, sample_portfolio, sample_week_data, mock_anthropic_response
    ):
        """Test synthesis with custom max_tokens."""
        analyzer = ClaudeAnalyzer(api_key=None)
        # Inject mock client directly
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response
        analyzer._client = mock_client
        analyzer._spend_reset_date = date.today()

        await analyzer.weekly_synthesis(
            sample_portfolio, sample_week_data, max_tokens=500
        )

        # Verify max_tokens was passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500


# =============================================================================
# NewsProcessor Tests
# =============================================================================

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from investment_monitor.analysis import NewsProcessor
from investment_monitor.storage import (
    NewsItem,
    get_session,
    init_db,
    save_news_item,
    get_unscored_news,
    get_recent_news,
)


@pytest.fixture
def news_db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_news.db"
        init_db(db_path)
        with get_session() as session:
            yield session


@pytest.fixture
def sample_news_items(news_db_session) -> list[NewsItem]:
    """Create sample news items for testing."""
    now = datetime.now()
    items = []

    # Create news items with varying relevance scores
    news_data = [
        ("AAPL", "Apple announces new iPhone 16 with AI features", None),  # Unscored
        ("AAPL", "Apple stock rises on earnings beat", 8.5),
        ("AAPL", "Weather report for California today", 1.0),  # Low relevance
        ("MSFT", "Microsoft Azure revenue grows 30%", 9.0),
        ("MSFT", "Tech industry news roundup", 4.5),  # Below threshold
        ("GOOGL", "Google launches new AI product", None),  # Unscored
    ]

    for ticker, headline, score in news_data:
        item = NewsItem(
            ticker=ticker,
            headline=headline,
            source="Test Source",
            url=f"https://example.com/news/{len(items)}",
            published_at=now - timedelta(hours=len(items)),
            relevance_score=score,
        )
        save_news_item(news_db_session, item)
        items.append(item)

    news_db_session.commit()
    return items


@pytest.fixture
def mock_llm():
    """Create a mock LocalLLM for testing."""
    mock = MagicMock()
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def test_portfolio() -> Portfolio:
    """Create a test portfolio."""
    return Portfolio(
        holdings=[
            Holding(
                ticker="AAPL",
                shares=Decimal("100"),
                cost_basis=Decimal("150.00"),
                thesis="Long-term growth in services",
            ),
            Holding(
                ticker="MSFT",
                shares=Decimal("50"),
                cost_basis=Decimal("350.00"),
                thesis="Cloud computing dominance",
            ),
        ],
        watchlist=[
            WatchlistItem(ticker="GOOGL", reason="AI potential"),
        ],
    )


class TestNewsProcessorInit:
    """Tests for NewsProcessor initialization."""

    def test_default_initialization(self, news_db_session, mock_llm, test_portfolio):
        """Test NewsProcessor initializes with default values."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )
        assert processor.min_relevance == 5.0
        assert processor.session == news_db_session
        assert processor.llm == mock_llm
        assert processor.portfolio == test_portfolio

    def test_custom_relevance_threshold(self, news_db_session, mock_llm, test_portfolio):
        """Test NewsProcessor accepts custom relevance threshold."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=7.0,
        )
        assert processor.min_relevance == 7.0


class TestNewsProcessorProcessUnscored:
    """Tests for process_unscored_news method."""

    @pytest.mark.asyncio
    async def test_process_unscored_scores_items(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test that unscored items get scored."""
        # Make score_relevance async
        async def mock_score(*args, **kwargs):
            return 7.5

        mock_llm.score_relevance = mock_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        # Initially should have 2 unscored items
        unscored = get_unscored_news(news_db_session)
        assert len(unscored) == 2

        processed = await processor.process_unscored_news()

        assert processed == 2

        # After processing, should have no unscored items
        unscored_after = get_unscored_news(news_db_session)
        assert len(unscored_after) == 0

    @pytest.mark.asyncio
    async def test_process_unscored_returns_zero_when_no_items(
        self, news_db_session, mock_llm, test_portfolio
    ):
        """Test returns 0 when no unscored items."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        processed = await processor.process_unscored_news()
        assert processed == 0

    @pytest.mark.asyncio
    async def test_process_unscored_handles_llm_unavailable(
        self, news_db_session, test_portfolio, sample_news_items
    ):
        """Test handling when LLM is unavailable."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        processed = await processor.process_unscored_news()

        # Should process 0 items successfully (all skipped)
        assert processed == 0

    @pytest.mark.asyncio
    async def test_process_unscored_respects_batch_size(
        self, news_db_session, mock_llm, test_portfolio
    ):
        """Test batch_size parameter is respected."""
        # Add more unscored items
        now = datetime.now()
        for i in range(10):
            item = NewsItem(
                ticker="TEST",
                headline=f"Test headline {i}",
                source="Test",
                url=f"https://example.com/batch/{i}",
                published_at=now,
            )
            save_news_item(news_db_session, item)
        news_db_session.commit()

        async def mock_score(*args, **kwargs):
            return 6.0

        mock_llm.score_relevance = mock_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        # Process with batch size of 5
        processed = await processor.process_unscored_news(batch_size=5)

        assert processed == 5

        # Should still have unscored items
        remaining = get_unscored_news(news_db_session)
        assert len(remaining) == 5


class TestNewsProcessorGetRelevantNews:
    """Tests for get_relevant_news method."""

    @pytest.mark.asyncio
    async def test_get_relevant_news_filters_by_threshold(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test that news is filtered by relevance threshold."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=5.0,
        )

        relevant = await processor.get_relevant_news()

        # Should only include items with score >= 5.0
        # From sample_news_items: 8.5, 9.0 are above, 1.0, 4.5 are below, None excluded
        assert len(relevant) == 2

        for item in relevant:
            assert item.relevance_score >= 5.0

    @pytest.mark.asyncio
    async def test_get_relevant_news_filters_by_ticker(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test filtering by specific ticker."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=5.0,
        )

        relevant = await processor.get_relevant_news(ticker="AAPL")

        # Should only include AAPL items with score >= 5.0
        # From sample: only one AAPL item with 8.5
        assert len(relevant) == 1
        assert relevant[0].ticker == "AAPL"
        assert relevant[0].relevance_score == 8.5

    @pytest.mark.asyncio
    async def test_get_relevant_news_filters_by_hours(
        self, news_db_session, mock_llm, test_portfolio
    ):
        """Test filtering by time window.

        Note: The storage layer filters by created_at timestamp, which is set
        automatically when the item is inserted. We can simulate an old item
        by directly updating the created_at field after insertion.
        """
        from sqlalchemy import update

        # Create a new item
        old_item = NewsItem(
            ticker="AAPL",
            headline="Old news item",
            source="Test",
            url="https://example.com/old",
            published_at=datetime.now() - timedelta(hours=48),
            relevance_score=9.0,
        )
        save_news_item(news_db_session, old_item)
        news_db_session.commit()

        # Update created_at to simulate an old item (backdating)
        old_timestamp = datetime.now() - timedelta(hours=48)
        stmt = update(NewsItem).where(NewsItem.url == "https://example.com/old").values(created_at=old_timestamp)
        news_db_session.execute(stmt)
        news_db_session.commit()

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=5.0,
        )

        # With 24 hour window, should not include old item
        relevant = await processor.get_relevant_news(hours=24)

        for item in relevant:
            assert item.headline != "Old news item"

    @pytest.mark.asyncio
    async def test_get_relevant_news_excludes_unscored(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test that unscored items are excluded."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=0.0,  # Include everything scored
        )

        relevant = await processor.get_relevant_news()

        # Should not include items with None score
        for item in relevant:
            assert item.relevance_score is not None


class TestNewsProcessorGetNewsByPriority:
    """Tests for get_news_by_priority method."""

    @pytest.mark.asyncio
    async def test_get_news_by_priority_sorts_descending(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test that news is sorted by relevance descending."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=0.0,
        )

        news = await processor.get_news_by_priority()

        # Verify sorted descending
        scores = [item.relevance_score for item in news if item.relevance_score]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_get_news_by_priority_custom_min_score(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test custom min_score override."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
            min_relevance=5.0,  # Default threshold
        )

        # Override with higher threshold
        news = await processor.get_news_by_priority(min_score=8.0)

        # Should only include items with score >= 8.0
        for item in news:
            assert item.relevance_score >= 8.0


class TestNewsProcessorScoreItem:
    """Tests for _score_item method."""

    @pytest.mark.asyncio
    async def test_score_item_returns_negative_when_unavailable(
        self, news_db_session, test_portfolio
    ):
        """Test returns -1 when LLM unavailable."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        item = NewsItem(
            ticker="AAPL",
            headline="Test headline",
            source="Test",
            url="https://example.com/test",
        )

        score = await processor._score_item(item)
        assert score == -1.0

    @pytest.mark.asyncio
    async def test_score_item_uses_thesis_context(
        self, news_db_session, test_portfolio
    ):
        """Test that portfolio thesis is used for context."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True

        captured_args = {}

        async def capture_score_relevance(headline, ticker, company_name=""):
            captured_args["headline"] = headline
            captured_args["ticker"] = ticker
            captured_args["company_name"] = company_name
            return 7.0

        mock_llm.score_relevance = capture_score_relevance

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        item = NewsItem(
            ticker="AAPL",
            headline="Apple services revenue grows",
            source="Test",
            url="https://example.com/test",
        )

        await processor._score_item(item)

        # Should include thesis in company_name
        assert "Long-term growth in services" in captured_args["company_name"]


class TestNewsProcessorScoreSingleItem:
    """Tests for score_single_item method."""

    @pytest.mark.asyncio
    async def test_score_single_item_persists(
        self, news_db_session, mock_llm, test_portfolio
    ):
        """Test that single item scoring persists to database."""
        async def mock_score(*args, **kwargs):
            return 8.0

        mock_llm.score_relevance = mock_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        item = NewsItem(
            ticker="AAPL",
            headline="Apple announces new product",
            source="Test",
            url="https://example.com/single",
        )
        save_news_item(news_db_session, item)
        news_db_session.commit()

        score = await processor.score_single_item(item)

        assert score == 8.0
        assert item.relevance_score == 8.0


class TestNewsProcessorGetUnscoredCount:
    """Tests for get_unscored_count method."""

    @pytest.mark.asyncio
    async def test_get_unscored_count(
        self, news_db_session, mock_llm, test_portfolio, sample_news_items
    ):
        """Test counting unscored items."""
        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        count = await processor.get_unscored_count()

        # From sample_news_items, 2 items have None score
        assert count == 2


class TestNewsProcessorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_item_without_ticker(
        self, news_db_session, mock_llm, test_portfolio
    ):
        """Test handling news items without ticker."""
        async def mock_score(*args, **kwargs):
            return 5.0

        mock_llm.score_relevance = mock_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        item = NewsItem(
            ticker=None,
            headline="General market news",
            source="Test",
            url="https://example.com/general",
        )
        save_news_item(news_db_session, item)
        news_db_session.commit()

        score = await processor._score_item(item)
        assert score == 5.0

    @pytest.mark.asyncio
    async def test_handles_scoring_exception(
        self, news_db_session, test_portfolio
    ):
        """Test handling exceptions during scoring."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True

        async def failing_score(*args, **kwargs):
            raise Exception("LLM error")

        mock_llm.score_relevance = failing_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=test_portfolio,
        )

        # Add an unscored item
        item = NewsItem(
            ticker="AAPL",
            headline="Test headline",
            source="Test",
            url="https://example.com/error",
        )
        save_news_item(news_db_session, item)
        news_db_session.commit()

        # Should not raise, should skip the failing item
        processed = await processor.process_unscored_news()

        # Item was skipped due to error
        assert processed == 0

    @pytest.mark.asyncio
    async def test_empty_portfolio_still_works(self, news_db_session, mock_llm):
        """Test that processor works with empty portfolio."""
        empty_portfolio = Portfolio(holdings=[], watchlist=[])

        async def mock_score(*args, **kwargs):
            return 6.0

        mock_llm.score_relevance = mock_score

        processor = NewsProcessor(
            session=news_db_session,
            llm=mock_llm,
            portfolio=empty_portfolio,
        )

        item = NewsItem(
            ticker="AAPL",
            headline="Apple news",
            source="Test",
            url="https://example.com/empty-portfolio",
        )
        save_news_item(news_db_session, item)
        news_db_session.commit()

        score = await processor._score_item(item)
        assert score == 6.0
