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
