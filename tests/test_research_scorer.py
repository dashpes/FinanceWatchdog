"""Tests for ResearchScorer with Ollama integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_monitor.analysis.research_scorer import (
    DEFAULT_REASONING,
    DEFAULT_SCORE,
    ResearchScorer,
    ScoreResult,
)
from investment_monitor.collectors.fundamentals import FundamentalsData
from investment_monitor.models.research import ScoringWeights


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
def default_weights() -> ScoringWeights:
    """Create default scoring weights."""
    return ScoringWeights(
        value=0.2,
        growth=0.2,
        quality=0.2,
        momentum=0.2,
        sentiment=0.2,
    )


@pytest.fixture
def custom_weights() -> ScoringWeights:
    """Create custom scoring weights with different distribution."""
    return ScoringWeights(
        value=0.3,
        growth=0.25,
        quality=0.25,
        momentum=0.1,
        sentiment=0.1,
    )


# =============================================================================
# ScoreResult Tests
# =============================================================================


class TestScoreResult:
    """Tests for ScoreResult dataclass."""

    def test_score_result_creation(self):
        """Test creating a ScoreResult."""
        result = ScoreResult(score=75.5, reasoning="Good value metrics")
        assert result.score == 75.5
        assert result.reasoning == "Good value metrics"

    def test_score_result_default_values(self):
        """Test ScoreResult with default score."""
        result = ScoreResult(score=DEFAULT_SCORE, reasoning=DEFAULT_REASONING)
        assert result.score == 50.0
        assert "unavailable" in result.reasoning.lower()


# =============================================================================
# ResearchScorer Initialization Tests
# =============================================================================


class TestResearchScorerInit:
    """Tests for ResearchScorer initialization."""

    def test_default_initialization(self):
        """ResearchScorer should initialize with default values."""
        scorer = ResearchScorer()
        assert scorer.model == "phi3:mini"
        assert scorer.base_url == "http://localhost:11434"
        assert scorer._client is None
        assert scorer._available is None

    def test_custom_initialization(self):
        """ResearchScorer should accept custom model and base_url."""
        scorer = ResearchScorer(model="llama2:7b", base_url="http://custom:1234")
        assert scorer.model == "llama2:7b"
        assert scorer.base_url == "http://custom:1234"


# =============================================================================
# Availability Tests
# =============================================================================


class TestResearchScorerAvailability:
    """Tests for is_available method behavior."""

    def test_is_available_when_ollama_not_installed(self):
        """Should return False when ollama package not installed."""
        scorer = ResearchScorer()

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            assert scorer.is_available() is False
            assert scorer._available is False

    def test_is_available_when_server_not_running(self):
        """Should return False when Ollama server is not running."""
        scorer = ResearchScorer()

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("Connection refused")
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert scorer.is_available() is False

    def test_is_available_when_model_not_found(self):
        """Should return False when specified model is not available."""
        scorer = ResearchScorer(model="nonexistent:model")

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "llama2:7b"}, {"name": "phi3:mini"}]
        }
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            with patch("investment_monitor.analysis.research_scorer.logger"):
                assert scorer.is_available() is False

    def test_is_available_when_model_found(self):
        """Should return True when model is available."""
        scorer = ResearchScorer(model="phi3:mini")

        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"name": "llama2:7b"}, {"name": "phi3:mini"}]
        }
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert scorer.is_available() is True
            assert scorer._available is True


# =============================================================================
# JSON Parsing Tests
# =============================================================================


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_valid_json(self):
        """Should correctly parse valid JSON response."""
        scorer = ResearchScorer()

        response = '{"score": 75, "reasoning": "Strong fundamentals"}'
        result = scorer._parse_json_response(response)

        assert result.score == 75.0
        assert result.reasoning == "Strong fundamentals"

    def test_parse_json_with_decimal_score(self):
        """Should correctly parse JSON with decimal score."""
        scorer = ResearchScorer()

        response = '{"score": 82.5, "reasoning": "Excellent growth"}'
        result = scorer._parse_json_response(response)

        assert result.score == 82.5
        assert result.reasoning == "Excellent growth"

    def test_parse_json_clamps_high_score(self):
        """Should clamp scores above 100 to 100."""
        scorer = ResearchScorer()

        response = '{"score": 150, "reasoning": "Test"}'
        result = scorer._parse_json_response(response)

        assert result.score == 100.0

    def test_parse_json_clamps_negative_score(self):
        """Should clamp negative scores to 0."""
        scorer = ResearchScorer()

        response = '{"score": -20, "reasoning": "Test"}'
        result = scorer._parse_json_response(response)

        assert result.score == 0.0

    def test_parse_json_with_extra_text(self):
        """Should extract JSON from response with extra text."""
        scorer = ResearchScorer()

        response = 'Here is my analysis: {"score": 65, "reasoning": "Moderate value"} Hope this helps!'
        result = scorer._parse_json_response(response)

        assert result.score == 65.0
        assert result.reasoning == "Moderate value"

    def test_parse_invalid_json_returns_default(self):
        """Should return default score for invalid JSON."""
        scorer = ResearchScorer()

        response = "This is not valid JSON at all"
        result = scorer._parse_json_response(response)

        assert result.score == DEFAULT_SCORE

    def test_parse_json_missing_score_field(self):
        """Should return default when score field is missing."""
        scorer = ResearchScorer()

        response = '{"reasoning": "No score here"}'
        result = scorer._parse_json_response(response)

        assert result.score == DEFAULT_SCORE
        assert "No score field" in result.reasoning

    def test_parse_json_invalid_score_type(self):
        """Should return default when score is not a number."""
        scorer = ResearchScorer()

        response = '{"score": "high", "reasoning": "Test"}'
        result = scorer._parse_json_response(response)

        assert result.score == DEFAULT_SCORE
        assert "Invalid score" in result.reasoning

    def test_parse_empty_response(self):
        """Should return default for empty response."""
        scorer = ResearchScorer()

        result = scorer._parse_json_response("")
        assert result.score == DEFAULT_SCORE

        result = scorer._parse_json_response(None)
        assert result.score == DEFAULT_SCORE

    def test_parse_malformed_json_recovery(self):
        """Should attempt to recover from malformed JSON."""
        scorer = ResearchScorer()

        # Missing quote
        response = '{"score": 70, "reasoning": "Test}'
        result = scorer._parse_json_response(response)

        # Should fall back to default since recovery fails
        assert result.score == DEFAULT_SCORE

    def test_parse_json_missing_reasoning(self):
        """Should handle missing reasoning field."""
        scorer = ResearchScorer()

        response = '{"score": 80}'
        result = scorer._parse_json_response(response)

        assert result.score == 80.0
        assert result.reasoning == "No reasoning provided"


# =============================================================================
# JSON Extraction Tests
# =============================================================================


class TestJSONExtraction:
    """Tests for _extract_json static method."""

    def test_extract_simple_json(self):
        """Should extract simple JSON object."""
        result = ResearchScorer._extract_json('{"score": 75, "reasoning": "Test"}')
        assert result is not None
        assert "score" in result

    def test_extract_json_from_text(self):
        """Should extract JSON from surrounding text."""
        result = ResearchScorer._extract_json(
            'Analysis: {"score": 80, "reasoning": "Good"} End.'
        )
        assert result is not None
        assert "80" in result

    def test_extract_json_empty_input(self):
        """Should return None for empty input."""
        assert ResearchScorer._extract_json("") is None
        assert ResearchScorer._extract_json(None) is None

    def test_extract_json_no_json(self):
        """Should return None when no JSON present."""
        result = ResearchScorer._extract_json("Just plain text here")
        assert result is None

    def test_extract_json_prefers_score_containing(self):
        """Should prefer JSON objects containing score field."""
        text = '{"other": 1} and {"score": 75, "reasoning": "Test"}'
        result = ResearchScorer._extract_json(text)
        assert result is not None
        assert "75" in result


# =============================================================================
# Value Formatting Tests
# =============================================================================


class TestValueFormatting:
    """Tests for _format_value method."""

    def test_format_value_none(self):
        """Should return 'N/A' for None values."""
        scorer = ResearchScorer()
        assert scorer._format_value(None) == "N/A"

    def test_format_value_regular(self):
        """Should format regular numbers."""
        scorer = ResearchScorer()
        assert scorer._format_value(25.678) == "25.68"
        assert scorer._format_value(100.0) == "100.00"

    def test_format_value_as_percent(self):
        """Should convert decimal to percentage."""
        scorer = ResearchScorer()
        assert scorer._format_value(0.15, as_percent=True) == "15.00"
        assert scorer._format_value(0.0825, as_percent=True) == "8.25"

    def test_format_value_percent_none(self):
        """Should return 'N/A' for None even with percent flag."""
        scorer = ResearchScorer()
        assert scorer._format_value(None, as_percent=True) == "N/A"


# =============================================================================
# Score Value Tests
# =============================================================================


class TestScoreValue:
    """Tests for score_value method."""

    @pytest.mark.asyncio
    async def test_score_value_unavailable(self, sample_fundamentals):
        """Should return default score when Ollama unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_value(sample_fundamentals)

        assert result.score == DEFAULT_SCORE
        assert result.reasoning == DEFAULT_REASONING

    @pytest.mark.asyncio
    async def test_score_value_success(self, sample_fundamentals):
        """Should return parsed score on success."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 72, "reasoning": "Fairly valued"}'
            ):
                result = await scorer.score_value(sample_fundamentals)

        assert result.score == 72.0
        assert result.reasoning == "Fairly valued"

    @pytest.mark.asyncio
    async def test_score_value_handles_nulls(self, sample_fundamentals_with_nulls):
        """Should handle fundamentals with null values."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 55, "reasoning": "Limited data"}'
            ):
                result = await scorer.score_value(sample_fundamentals_with_nulls)

        assert result.score == 55.0

    @pytest.mark.asyncio
    async def test_score_value_generation_failure(self, sample_fundamentals):
        """Should return default on generation failure."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(scorer, "_generate", return_value=None):
                result = await scorer.score_value(sample_fundamentals)

        assert result.score == DEFAULT_SCORE


# =============================================================================
# Score Growth Tests
# =============================================================================


class TestScoreGrowth:
    """Tests for score_growth method."""

    @pytest.mark.asyncio
    async def test_score_growth_unavailable(self, sample_fundamentals):
        """Should return default score when Ollama unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_growth(sample_fundamentals)

        assert result.score == DEFAULT_SCORE

    @pytest.mark.asyncio
    async def test_score_growth_success(self, sample_fundamentals):
        """Should return parsed score on success."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 65, "reasoning": "Moderate growth"}'
            ):
                result = await scorer.score_growth(sample_fundamentals)

        assert result.score == 65.0
        assert result.reasoning == "Moderate growth"


# =============================================================================
# Score Quality Tests
# =============================================================================


class TestScoreQuality:
    """Tests for score_quality method."""

    @pytest.mark.asyncio
    async def test_score_quality_unavailable(self, sample_fundamentals):
        """Should return default score when Ollama unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_quality(sample_fundamentals)

        assert result.score == DEFAULT_SCORE

    @pytest.mark.asyncio
    async def test_score_quality_success(self, sample_fundamentals):
        """Should return parsed score on success."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 78, "reasoning": "High quality business"}'
            ):
                result = await scorer.score_quality(sample_fundamentals)

        assert result.score == 78.0


# =============================================================================
# Score Momentum Tests
# =============================================================================


class TestScoreMomentum:
    """Tests for score_momentum method."""

    @pytest.mark.asyncio
    async def test_score_momentum_unavailable(self):
        """Should return default score when Ollama unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_momentum(
                ticker="AAPL",
                price_change_1m=5.2,
                price_change_3m=12.5,
            )

        assert result.score == DEFAULT_SCORE

    @pytest.mark.asyncio
    async def test_score_momentum_success(self):
        """Should return parsed score on success."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 70, "reasoning": "Positive momentum"}'
            ):
                result = await scorer.score_momentum(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    price_change_1m=5.2,
                    price_change_3m=12.5,
                    price_change_6m=25.0,
                    price_change_1y=45.0,
                    rsi=62.0,
                    vs_52w_high=-8.0,
                    vs_52w_low=45.0,
                    sector="Technology",
                    industry="Consumer Electronics",
                )

        assert result.score == 70.0

    @pytest.mark.asyncio
    async def test_score_momentum_with_defaults(self):
        """Should use default values for optional parameters."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 50, "reasoning": "Neutral"}'
            ):
                result = await scorer.score_momentum(ticker="TEST")

        assert result.score == 50.0


# =============================================================================
# Score Sentiment Tests
# =============================================================================


class TestScoreSentiment:
    """Tests for score_sentiment method."""

    @pytest.mark.asyncio
    async def test_score_sentiment_unavailable(self):
        """Should return default score when Ollama unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_sentiment(ticker="AAPL")

        assert result.score == DEFAULT_SCORE

    @pytest.mark.asyncio
    async def test_score_sentiment_success(self):
        """Should return parsed score on success."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 82, "reasoning": "Bullish sentiment"}'
            ):
                result = await scorer.score_sentiment(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    recent_news_summary="Positive earnings report, new product launch",
                    insider_activity="CEO bought 10,000 shares",
                    analyst_rating="Strong Buy (4.5/5)",
                    short_interest=2.5,
                    sector="Technology",
                    industry="Consumer Electronics",
                )

        assert result.score == 82.0
        assert result.reasoning == "Bullish sentiment"

    @pytest.mark.asyncio
    async def test_score_sentiment_with_defaults(self):
        """Should use default values for optional parameters."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 50, "reasoning": "No data"}'
            ):
                result = await scorer.score_sentiment(ticker="TEST")

        assert result.score == 50.0


# =============================================================================
# Composite Score Tests
# =============================================================================


class TestCompositeScore:
    """Tests for calculate_composite_score method."""

    @pytest.mark.asyncio
    async def test_composite_equal_weights(self, default_weights):
        """Should calculate correct composite with equal weights."""
        scorer = ResearchScorer()

        value_result = ScoreResult(score=80.0, reasoning="Good value")
        growth_result = ScoreResult(score=70.0, reasoning="Solid growth")
        quality_result = ScoreResult(score=75.0, reasoning="High quality")
        momentum_result = ScoreResult(score=65.0, reasoning="Positive momentum")
        sentiment_result = ScoreResult(score=60.0, reasoning="Neutral sentiment")

        candidate = await scorer.calculate_composite_score(
            value_result=value_result,
            growth_result=growth_result,
            quality_result=quality_result,
            momentum_result=momentum_result,
            sentiment_result=sentiment_result,
            weights=default_weights,
            ticker="AAPL",
        )

        # Expected: (80*0.2) + (70*0.2) + (75*0.2) + (65*0.2) + (60*0.2) = 70.0
        assert candidate.composite_score == 70.0
        assert candidate.ticker == "AAPL"
        assert candidate.value_score == 80.0
        assert candidate.growth_score == 70.0
        assert candidate.quality_score == 75.0
        assert candidate.momentum_score == 65.0
        assert candidate.sentiment_score == 60.0

    @pytest.mark.asyncio
    async def test_composite_custom_weights(self, custom_weights):
        """Should calculate correct composite with custom weights."""
        scorer = ResearchScorer()

        value_result = ScoreResult(score=80.0, reasoning="Good value")
        growth_result = ScoreResult(score=70.0, reasoning="Solid growth")
        quality_result = ScoreResult(score=75.0, reasoning="High quality")
        momentum_result = ScoreResult(score=65.0, reasoning="Positive momentum")
        sentiment_result = ScoreResult(score=60.0, reasoning="Neutral sentiment")

        candidate = await scorer.calculate_composite_score(
            value_result=value_result,
            growth_result=growth_result,
            quality_result=quality_result,
            momentum_result=momentum_result,
            sentiment_result=sentiment_result,
            weights=custom_weights,
            ticker="MSFT",
        )

        # Expected: (80*0.3) + (70*0.25) + (75*0.25) + (65*0.1) + (60*0.1) = 73.75
        expected = (80 * 0.3) + (70 * 0.25) + (75 * 0.25) + (65 * 0.1) + (60 * 0.1)
        assert candidate.composite_score == pytest.approx(expected, rel=0.001)

    @pytest.mark.asyncio
    async def test_composite_reasoning_combined(self, default_weights):
        """Should combine reasoning from all factors."""
        scorer = ResearchScorer()

        value_result = ScoreResult(score=80.0, reasoning="Value reason")
        growth_result = ScoreResult(score=70.0, reasoning="Growth reason")
        quality_result = ScoreResult(score=75.0, reasoning="Quality reason")
        momentum_result = ScoreResult(score=65.0, reasoning="Momentum reason")
        sentiment_result = ScoreResult(score=60.0, reasoning="Sentiment reason")

        candidate = await scorer.calculate_composite_score(
            value_result=value_result,
            growth_result=growth_result,
            quality_result=quality_result,
            momentum_result=momentum_result,
            sentiment_result=sentiment_result,
            weights=default_weights,
            ticker="TEST",
        )

        assert "Value reason" in candidate.reasoning
        assert "Growth reason" in candidate.reasoning
        assert "Quality reason" in candidate.reasoning
        assert "Momentum reason" in candidate.reasoning
        assert "Sentiment reason" in candidate.reasoning

    @pytest.mark.asyncio
    async def test_composite_all_zeros(self, default_weights):
        """Should handle all zero scores."""
        scorer = ResearchScorer()

        zero_result = ScoreResult(score=0.0, reasoning="Zero score")

        candidate = await scorer.calculate_composite_score(
            value_result=zero_result,
            growth_result=zero_result,
            quality_result=zero_result,
            momentum_result=zero_result,
            sentiment_result=zero_result,
            weights=default_weights,
            ticker="ZERO",
        )

        assert candidate.composite_score == 0.0

    @pytest.mark.asyncio
    async def test_composite_all_hundreds(self, default_weights):
        """Should handle all perfect scores."""
        scorer = ResearchScorer()

        perfect_result = ScoreResult(score=100.0, reasoning="Perfect")

        candidate = await scorer.calculate_composite_score(
            value_result=perfect_result,
            growth_result=perfect_result,
            quality_result=perfect_result,
            momentum_result=perfect_result,
            sentiment_result=perfect_result,
            weights=default_weights,
            ticker="PERFECT",
        )

        assert candidate.composite_score == 100.0


# =============================================================================
# Full Stock Scoring Tests
# =============================================================================


class TestScoreStock:
    """Tests for score_stock convenience method."""

    @pytest.mark.asyncio
    async def test_score_stock_integration(self, sample_fundamentals, default_weights):
        """Should score stock across all factors."""
        scorer = ResearchScorer()

        # Mock all scoring to return controlled values. Force the per-factor path
        # (batched scoring returns None) so the five mocked responses map 1:1 to the
        # five factor calls — the batched fast path would otherwise consume one.
        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(scorer, "_score_batched", new_callable=AsyncMock, return_value=None):
                with patch.object(
                    scorer,
                    "_generate",
                    side_effect=[
                        '{"score": 70, "reasoning": "Value"}',
                        '{"score": 60, "reasoning": "Growth"}',
                        '{"score": 75, "reasoning": "Quality"}',
                        '{"score": 65, "reasoning": "Momentum"}',
                        '{"score": 55, "reasoning": "Sentiment"}',
                    ],
                ):
                    candidate = await scorer.score_stock(
                    fundamentals=sample_fundamentals,
                    weights=default_weights,
                    price_change_1m=5.0,
                    price_change_3m=10.0,
                    recent_news_summary="Positive earnings",
                )

        assert candidate.ticker == "AAPL"
        assert candidate.value_score == 70.0
        assert candidate.growth_score == 60.0
        assert candidate.quality_score == 75.0
        assert candidate.momentum_score == 65.0
        assert candidate.sentiment_score == 55.0

        # Expected composite: (70+60+75+65+55) * 0.2 = 65.0
        expected = (70 + 60 + 75 + 65 + 55) * 0.2
        assert candidate.composite_score == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_score_stock_unavailable(self, sample_fundamentals, default_weights):
        """Should return default scores when unavailable."""
        scorer = ResearchScorer()

        with patch.object(scorer, "is_available", return_value=False):
            candidate = await scorer.score_stock(
                fundamentals=sample_fundamentals,
                weights=default_weights,
            )

        # All scores should be default (50)
        assert candidate.value_score == DEFAULT_SCORE
        assert candidate.growth_score == DEFAULT_SCORE
        assert candidate.quality_score == DEFAULT_SCORE
        assert candidate.momentum_score == DEFAULT_SCORE
        assert candidate.sentiment_score == DEFAULT_SCORE
        assert candidate.composite_score == DEFAULT_SCORE


# =============================================================================
# Generate Method Tests
# =============================================================================


class TestGenerate:
    """Tests for _generate method."""

    def test_generate_returns_response(self):
        """Should return stripped response from LLM."""
        scorer = ResearchScorer()

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": '  {"score": 75, "reasoning": "Test"}  '}
        scorer._client = mock_client

        result = scorer._generate("Test prompt")
        assert result == '{"score": 75, "reasoning": "Test"}'

    def test_generate_handles_exception(self):
        """Should return None when generation raises exception."""
        scorer = ResearchScorer()

        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("API Error")
        scorer._client = mock_client

        with patch("investment_monitor.analysis.research_scorer.logger"):
            result = scorer._generate("Test prompt")
            assert result is None

    def test_generate_uses_correct_parameters(self):
        """Should pass correct parameters to the client."""
        scorer = ResearchScorer(model="test-model")

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "test"}
        scorer._client = mock_client

        scorer._generate("Test prompt")

        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["prompt"] == "Test prompt"
        assert "options" in call_args.kwargs
        assert call_args.kwargs["options"]["temperature"] == 0.1


# =============================================================================
# Integration Tests (with mocked Ollama)
# =============================================================================


class TestResearchScorerIntegration:
    """Integration tests with fully mocked Ollama."""

    @pytest.fixture
    def mock_ollama_module(self):
        """Create a mock ollama module."""
        mock_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_module.Client.return_value = mock_client_instance
        return mock_module, mock_client_instance

    @pytest.mark.asyncio
    async def test_full_scoring_flow(
        self, mock_ollama_module, sample_fundamentals, default_weights
    ):
        """Test complete scoring flow from fundamentals to CandidateScore."""
        mock_module, mock_client = mock_ollama_module
        mock_client.list.return_value = {"models": [{"name": "phi3:mini"}]}
        mock_client.generate.return_value = {
            "response": '{"score": 75, "reasoning": "Good stock"}'
        }

        scorer = ResearchScorer()

        with patch.dict("sys.modules", {"ollama": mock_module}):
            scorer._client = mock_client

            with patch.object(scorer, "is_available", return_value=True):
                candidate = await scorer.score_stock(
                    fundamentals=sample_fundamentals,
                    weights=default_weights,
                    price_change_1m=5.0,
                    recent_news_summary="Positive outlook",
                )

        # All factors should have score of 75 from mocked response
        assert candidate.composite_score == 75.0
        assert candidate.value_score == 75.0
        assert candidate.growth_score == 75.0
        assert candidate.quality_score == 75.0
        assert candidate.momentum_score == 75.0
        assert candidate.sentiment_score == 75.0

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_connection_error(
        self, sample_fundamentals, default_weights
    ):
        """Test graceful handling when Ollama server is down."""
        scorer = ResearchScorer()

        # Simulate unavailable Ollama
        with patch.object(scorer, "is_available", return_value=False):
            candidate = await scorer.score_stock(
                fundamentals=sample_fundamentals,
                weights=default_weights,
            )

        # All scores should fall back to defaults
        assert candidate.composite_score == DEFAULT_SCORE
        assert "unavailable" in candidate.reasoning.lower()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_fundamentals_with_all_nulls(self, default_weights):
        """Test scoring with all null fundamentals."""
        scorer = ResearchScorer()

        fundamentals = FundamentalsData(ticker="NULL")

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 50, "reasoning": "No data"}'
            ):
                result = await scorer.score_value(fundamentals)

        assert result.score == 50.0

    @pytest.mark.asyncio
    async def test_very_large_numbers(self, default_weights):
        """Test scoring with very large metric values."""
        scorer = ResearchScorer()

        fundamentals = FundamentalsData(
            ticker="BIG",
            pe_ratio=9999.99,
            free_cash_flow=999999999999999,
            market_cap=99999999999999999,
        )

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 20, "reasoning": "Overvalued"}'
            ):
                result = await scorer.score_value(fundamentals)

        assert result.score == 20.0

    @pytest.mark.asyncio
    async def test_negative_metric_values(self, default_weights):
        """Test scoring with negative metric values (e.g., negative earnings)."""
        scorer = ResearchScorer()

        fundamentals = FundamentalsData(
            ticker="NEG",
            pe_ratio=-15.5,
            roe=-0.25,
            profit_margin=-0.15,
        )

        with patch.object(scorer, "is_available", return_value=True):
            with patch.object(
                scorer, "_generate", return_value='{"score": 15, "reasoning": "Negative earnings"}'
            ):
                result = await scorer.score_quality(fundamentals)

        assert result.score == 15.0

    @pytest.mark.asyncio
    async def test_empty_sector_and_industry(self):
        """Test formatting with empty sector/industry."""
        scorer = ResearchScorer()

        fundamentals = FundamentalsData(
            ticker="EMPTY",
            sector=None,
            industry=None,
        )

        # Just verifying no exception is raised
        with patch.object(scorer, "is_available", return_value=False):
            result = await scorer.score_value(fundamentals)

        assert result.score == DEFAULT_SCORE

    def test_special_characters_in_responses(self):
        """Test handling special characters in reasoning."""
        scorer = ResearchScorer()

        response = '{"score": 65, "reasoning": "Growth is good\\nDebt is manageable\\tOverall: positive"}'
        result = scorer._parse_json_response(response)

        assert result.score == 65.0
        assert "Growth is good" in result.reasoning

    def test_unicode_in_responses(self):
        """Test handling unicode characters in reasoning."""
        scorer = ResearchScorer()

        response = '{"score": 70, "reasoning": "Strong growth trajectory"}'
        result = scorer._parse_json_response(response)

        assert result.score == 70.0
