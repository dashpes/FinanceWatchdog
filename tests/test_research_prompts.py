"""Tests for research scoring prompts."""

import pytest

from investment_monitor.analysis.research_prompts import (
    GROWTH_SCORE_PROMPT,
    MOMENTUM_SCORE_PROMPT,
    PROMPT_PLACEHOLDERS,
    QUALITY_SCORE_PROMPT,
    RESEARCH_PROMPTS,
    SENTIMENT_SCORE_PROMPT,
    VALUE_SCORE_PROMPT,
)


class TestPromptStructure:
    """Tests to verify prompt structure and content."""

    def test_all_prompts_are_non_empty(self):
        """All prompts should be non-empty strings."""
        assert len(VALUE_SCORE_PROMPT) > 0
        assert len(GROWTH_SCORE_PROMPT) > 0
        assert len(QUALITY_SCORE_PROMPT) > 0
        assert len(MOMENTUM_SCORE_PROMPT) > 0
        assert len(SENTIMENT_SCORE_PROMPT) > 0

    def test_all_prompts_are_strings(self):
        """All prompts should be string types."""
        assert isinstance(VALUE_SCORE_PROMPT, str)
        assert isinstance(GROWTH_SCORE_PROMPT, str)
        assert isinstance(QUALITY_SCORE_PROMPT, str)
        assert isinstance(MOMENTUM_SCORE_PROMPT, str)
        assert isinstance(SENTIMENT_SCORE_PROMPT, str)

    def test_research_prompts_dict_contains_all(self):
        """RESEARCH_PROMPTS dict should contain all prompts."""
        assert "value" in RESEARCH_PROMPTS
        assert "growth" in RESEARCH_PROMPTS
        assert "quality" in RESEARCH_PROMPTS
        assert "momentum" in RESEARCH_PROMPTS
        assert "sentiment" in RESEARCH_PROMPTS
        assert len(RESEARCH_PROMPTS) == 5


class TestJSONFormatInstructions:
    """Tests to verify JSON format instructions are present."""

    @pytest.mark.parametrize(
        "prompt_name,prompt",
        [
            ("value", VALUE_SCORE_PROMPT),
            ("growth", GROWTH_SCORE_PROMPT),
            ("quality", QUALITY_SCORE_PROMPT),
            ("momentum", MOMENTUM_SCORE_PROMPT),
            ("sentiment", SENTIMENT_SCORE_PROMPT),
        ],
    )
    def test_prompt_contains_json_instructions(self, prompt_name, prompt):
        """Each prompt should contain JSON format instructions."""
        assert "JSON" in prompt, f"{prompt_name} prompt missing JSON instruction"
        assert '"score"' in prompt, f"{prompt_name} prompt missing score field"
        assert '"reasoning"' in prompt, f"{prompt_name} prompt missing reasoning field"

    @pytest.mark.parametrize(
        "prompt_name,prompt",
        [
            ("value", VALUE_SCORE_PROMPT),
            ("growth", GROWTH_SCORE_PROMPT),
            ("quality", QUALITY_SCORE_PROMPT),
            ("momentum", MOMENTUM_SCORE_PROMPT),
            ("sentiment", SENTIMENT_SCORE_PROMPT),
        ],
    )
    def test_prompt_specifies_score_range(self, prompt_name, prompt):
        """Each prompt should specify the 0-100 score range."""
        assert "0-100" in prompt, f"{prompt_name} prompt missing 0-100 score range"


class TestValueScorePromptPlaceholders:
    """Tests for VALUE_SCORE_PROMPT placeholders."""

    def test_contains_ticker_placeholder(self):
        """Should contain ticker placeholder."""
        assert "{ticker}" in VALUE_SCORE_PROMPT

    def test_contains_company_name_placeholder(self):
        """Should contain company_name placeholder."""
        assert "{company_name}" in VALUE_SCORE_PROMPT

    def test_contains_pe_ratio_placeholder(self):
        """Should contain pe_ratio placeholder."""
        assert "{pe_ratio}" in VALUE_SCORE_PROMPT

    def test_contains_pb_ratio_placeholder(self):
        """Should contain pb_ratio placeholder."""
        assert "{pb_ratio}" in VALUE_SCORE_PROMPT

    def test_contains_ps_ratio_placeholder(self):
        """Should contain ps_ratio placeholder."""
        assert "{ps_ratio}" in VALUE_SCORE_PROMPT

    def test_contains_peg_ratio_placeholder(self):
        """Should contain peg_ratio placeholder."""
        assert "{peg_ratio}" in VALUE_SCORE_PROMPT

    def test_contains_dividend_yield_placeholder(self):
        """Should contain dividend_yield placeholder."""
        assert "{dividend_yield}" in VALUE_SCORE_PROMPT

    def test_contains_free_cash_flow_placeholder(self):
        """Should contain free_cash_flow placeholder."""
        assert "{free_cash_flow}" in VALUE_SCORE_PROMPT

    def test_contains_sector_placeholder(self):
        """Should contain sector placeholder."""
        assert "{sector}" in VALUE_SCORE_PROMPT

    def test_contains_industry_placeholder(self):
        """Should contain industry placeholder."""
        assert "{industry}" in VALUE_SCORE_PROMPT

    def test_all_placeholders_documented(self):
        """All expected placeholders should be documented in PROMPT_PLACEHOLDERS."""
        expected = PROMPT_PLACEHOLDERS["value"]
        for placeholder in expected:
            assert (
                "{" + placeholder + "}" in VALUE_SCORE_PROMPT
            ), f"Missing {placeholder}"


class TestGrowthScorePromptPlaceholders:
    """Tests for GROWTH_SCORE_PROMPT placeholders."""

    def test_contains_ticker_placeholder(self):
        """Should contain ticker placeholder."""
        assert "{ticker}" in GROWTH_SCORE_PROMPT

    def test_contains_company_name_placeholder(self):
        """Should contain company_name placeholder."""
        assert "{company_name}" in GROWTH_SCORE_PROMPT

    def test_contains_revenue_growth_yoy_placeholder(self):
        """Should contain revenue_growth_yoy placeholder."""
        assert "{revenue_growth_yoy}" in GROWTH_SCORE_PROMPT

    def test_contains_revenue_growth_3y_placeholder(self):
        """Should contain revenue_growth_3y placeholder."""
        assert "{revenue_growth_3y}" in GROWTH_SCORE_PROMPT

    def test_contains_eps_growth_yoy_placeholder(self):
        """Should contain eps_growth_yoy placeholder."""
        assert "{eps_growth_yoy}" in GROWTH_SCORE_PROMPT

    def test_contains_eps_growth_3y_placeholder(self):
        """Should contain eps_growth_3y placeholder."""
        assert "{eps_growth_3y}" in GROWTH_SCORE_PROMPT

    def test_contains_sector_placeholder(self):
        """Should contain sector placeholder."""
        assert "{sector}" in GROWTH_SCORE_PROMPT

    def test_contains_industry_placeholder(self):
        """Should contain industry placeholder."""
        assert "{industry}" in GROWTH_SCORE_PROMPT

    def test_all_placeholders_documented(self):
        """All expected placeholders should be documented in PROMPT_PLACEHOLDERS."""
        expected = PROMPT_PLACEHOLDERS["growth"]
        for placeholder in expected:
            assert (
                "{" + placeholder + "}" in GROWTH_SCORE_PROMPT
            ), f"Missing {placeholder}"


class TestQualityScorePromptPlaceholders:
    """Tests for QUALITY_SCORE_PROMPT placeholders."""

    def test_contains_ticker_placeholder(self):
        """Should contain ticker placeholder."""
        assert "{ticker}" in QUALITY_SCORE_PROMPT

    def test_contains_company_name_placeholder(self):
        """Should contain company_name placeholder."""
        assert "{company_name}" in QUALITY_SCORE_PROMPT

    def test_contains_roe_placeholder(self):
        """Should contain roe placeholder."""
        assert "{roe}" in QUALITY_SCORE_PROMPT

    def test_contains_profit_margin_placeholder(self):
        """Should contain profit_margin placeholder."""
        assert "{profit_margin}" in QUALITY_SCORE_PROMPT

    def test_contains_debt_to_equity_placeholder(self):
        """Should contain debt_to_equity placeholder."""
        assert "{debt_to_equity}" in QUALITY_SCORE_PROMPT

    def test_contains_current_ratio_placeholder(self):
        """Should contain current_ratio placeholder."""
        assert "{current_ratio}" in QUALITY_SCORE_PROMPT

    def test_contains_sector_placeholder(self):
        """Should contain sector placeholder."""
        assert "{sector}" in QUALITY_SCORE_PROMPT

    def test_contains_industry_placeholder(self):
        """Should contain industry placeholder."""
        assert "{industry}" in QUALITY_SCORE_PROMPT

    def test_all_placeholders_documented(self):
        """All expected placeholders should be documented in PROMPT_PLACEHOLDERS."""
        expected = PROMPT_PLACEHOLDERS["quality"]
        for placeholder in expected:
            assert (
                "{" + placeholder + "}" in QUALITY_SCORE_PROMPT
            ), f"Missing {placeholder}"


class TestMomentumScorePromptPlaceholders:
    """Tests for MOMENTUM_SCORE_PROMPT placeholders."""

    def test_contains_ticker_placeholder(self):
        """Should contain ticker placeholder."""
        assert "{ticker}" in MOMENTUM_SCORE_PROMPT

    def test_contains_company_name_placeholder(self):
        """Should contain company_name placeholder."""
        assert "{company_name}" in MOMENTUM_SCORE_PROMPT

    def test_contains_price_change_1m_placeholder(self):
        """Should contain price_change_1m placeholder."""
        assert "{price_change_1m}" in MOMENTUM_SCORE_PROMPT

    def test_contains_price_change_3m_placeholder(self):
        """Should contain price_change_3m placeholder."""
        assert "{price_change_3m}" in MOMENTUM_SCORE_PROMPT

    def test_contains_price_change_6m_placeholder(self):
        """Should contain price_change_6m placeholder."""
        assert "{price_change_6m}" in MOMENTUM_SCORE_PROMPT

    def test_contains_price_change_1y_placeholder(self):
        """Should contain price_change_1y placeholder."""
        assert "{price_change_1y}" in MOMENTUM_SCORE_PROMPT

    def test_contains_rsi_placeholder(self):
        """Should contain rsi placeholder."""
        assert "{rsi}" in MOMENTUM_SCORE_PROMPT

    def test_contains_vs_52w_high_placeholder(self):
        """Should contain vs_52w_high placeholder."""
        assert "{vs_52w_high}" in MOMENTUM_SCORE_PROMPT

    def test_contains_vs_52w_low_placeholder(self):
        """Should contain vs_52w_low placeholder."""
        assert "{vs_52w_low}" in MOMENTUM_SCORE_PROMPT

    def test_all_placeholders_documented(self):
        """All expected placeholders should be documented in PROMPT_PLACEHOLDERS."""
        expected = PROMPT_PLACEHOLDERS["momentum"]
        for placeholder in expected:
            assert (
                "{" + placeholder + "}" in MOMENTUM_SCORE_PROMPT
            ), f"Missing {placeholder}"


class TestSentimentScorePromptPlaceholders:
    """Tests for SENTIMENT_SCORE_PROMPT placeholders."""

    def test_contains_ticker_placeholder(self):
        """Should contain ticker placeholder."""
        assert "{ticker}" in SENTIMENT_SCORE_PROMPT

    def test_contains_company_name_placeholder(self):
        """Should contain company_name placeholder."""
        assert "{company_name}" in SENTIMENT_SCORE_PROMPT

    def test_contains_recent_news_summary_placeholder(self):
        """Should contain recent_news_summary placeholder."""
        assert "{recent_news_summary}" in SENTIMENT_SCORE_PROMPT

    def test_contains_insider_activity_placeholder(self):
        """Should contain insider_activity placeholder."""
        assert "{insider_activity}" in SENTIMENT_SCORE_PROMPT

    def test_contains_analyst_rating_placeholder(self):
        """Should contain analyst_rating placeholder."""
        assert "{analyst_rating}" in SENTIMENT_SCORE_PROMPT

    def test_contains_short_interest_placeholder(self):
        """Should contain short_interest placeholder."""
        assert "{short_interest}" in SENTIMENT_SCORE_PROMPT

    def test_all_placeholders_documented(self):
        """All expected placeholders should be documented in PROMPT_PLACEHOLDERS."""
        expected = PROMPT_PLACEHOLDERS["sentiment"]
        for placeholder in expected:
            assert (
                "{" + placeholder + "}" in SENTIMENT_SCORE_PROMPT
            ), f"Missing {placeholder}"


class TestPromptFormatting:
    """Tests for prompt formatting with actual values."""

    def test_value_prompt_formats_correctly(self):
        """VALUE_SCORE_PROMPT should format with all placeholders."""
        formatted = VALUE_SCORE_PROMPT.format(
            ticker="AAPL",
            company_name="Apple Inc.",
            pe_ratio=28.5,
            pb_ratio=45.2,
            ps_ratio=7.8,
            peg_ratio=2.1,
            dividend_yield=0.5,
            free_cash_flow=99000000000,
            sector="Technology",
            industry="Consumer Electronics",
        )
        assert "AAPL" in formatted
        assert "Apple Inc." in formatted
        assert "28.5" in formatted
        assert "Technology" in formatted

    def test_growth_prompt_formats_correctly(self):
        """GROWTH_SCORE_PROMPT should format with all placeholders."""
        formatted = GROWTH_SCORE_PROMPT.format(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            revenue_growth_yoy=122.4,
            revenue_growth_3y=65.8,
            eps_growth_yoy=586.2,
            eps_growth_3y=89.3,
            sector="Technology",
            industry="Semiconductors",
        )
        assert "NVDA" in formatted
        assert "NVIDIA Corporation" in formatted
        assert "122.4" in formatted
        assert "Semiconductors" in formatted

    def test_quality_prompt_formats_correctly(self):
        """QUALITY_SCORE_PROMPT should format with all placeholders."""
        formatted = QUALITY_SCORE_PROMPT.format(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            roe=35.5,
            profit_margin=36.7,
            debt_to_equity=0.42,
            current_ratio=1.77,
            sector="Technology",
            industry="Software",
        )
        assert "MSFT" in formatted
        assert "Microsoft Corporation" in formatted
        assert "35.5" in formatted
        assert "Software" in formatted

    def test_momentum_prompt_formats_correctly(self):
        """MOMENTUM_SCORE_PROMPT should format with all placeholders."""
        formatted = MOMENTUM_SCORE_PROMPT.format(
            ticker="TSLA",
            company_name="Tesla Inc.",
            price_change_1m=15.2,
            price_change_3m=-8.5,
            price_change_6m=25.3,
            price_change_1y=102.5,
            rsi=62,
            vs_52w_high=-12.5,
            vs_52w_low=85.3,
            sector="Consumer Cyclical",
            industry="Auto Manufacturers",
        )
        assert "TSLA" in formatted
        assert "Tesla Inc." in formatted
        assert "15.2" in formatted
        assert "-12.5" in formatted
        assert "Auto Manufacturers" in formatted

    def test_sentiment_prompt_formats_correctly(self):
        """SENTIMENT_SCORE_PROMPT should format with all placeholders."""
        formatted = SENTIMENT_SCORE_PROMPT.format(
            ticker="META",
            company_name="Meta Platforms Inc.",
            recent_news_summary="Positive earnings report, AI investments highlighted",
            insider_activity="CEO sold 10,000 shares for tax purposes",
            analyst_rating="Buy (4.2/5)",
            short_interest=2.5,
            sector="Communication Services",
            industry="Internet Content & Information",
        )
        assert "META" in formatted
        assert "Meta Platforms Inc." in formatted
        assert "Positive earnings report" in formatted
        assert "Buy (4.2/5)" in formatted
        assert "Internet Content & Information" in formatted


class TestScoringGuidelines:
    """Tests to verify scoring guidelines are present."""

    @pytest.mark.parametrize(
        "prompt_name,prompt",
        [
            ("value", VALUE_SCORE_PROMPT),
            ("growth", GROWTH_SCORE_PROMPT),
            ("quality", QUALITY_SCORE_PROMPT),
            ("momentum", MOMENTUM_SCORE_PROMPT),
            ("sentiment", SENTIMENT_SCORE_PROMPT),
        ],
    )
    def test_prompt_contains_scoring_ranges(self, prompt_name, prompt):
        """Each prompt should contain all scoring ranges."""
        assert "80-100" in prompt, f"{prompt_name} missing 80-100 range"
        assert "60-79" in prompt, f"{prompt_name} missing 60-79 range"
        assert "40-59" in prompt, f"{prompt_name} missing 40-59 range"
        assert "20-39" in prompt, f"{prompt_name} missing 20-39 range"
        assert "0-19" in prompt, f"{prompt_name} missing 0-19 range"

    @pytest.mark.parametrize(
        "prompt_name,prompt",
        [
            ("value", VALUE_SCORE_PROMPT),
            ("growth", GROWTH_SCORE_PROMPT),
            ("quality", QUALITY_SCORE_PROMPT),
            ("momentum", MOMENTUM_SCORE_PROMPT),
            ("sentiment", SENTIMENT_SCORE_PROMPT),
        ],
    )
    def test_prompt_contains_scoring_guidelines_section(self, prompt_name, prompt):
        """Each prompt should have a Scoring Guidelines section."""
        assert (
            "Scoring Guidelines:" in prompt
        ), f"{prompt_name} missing Scoring Guidelines section"


class TestPromptPlaceholdersConsistency:
    """Tests to verify PROMPT_PLACEHOLDERS matches actual prompts."""

    def test_value_placeholders_match_prompt(self):
        """PROMPT_PLACEHOLDERS for value should match VALUE_SCORE_PROMPT."""
        for placeholder in PROMPT_PLACEHOLDERS["value"]:
            assert (
                "{" + placeholder + "}" in VALUE_SCORE_PROMPT
            ), f"Placeholder {placeholder} not in VALUE_SCORE_PROMPT"

    def test_growth_placeholders_match_prompt(self):
        """PROMPT_PLACEHOLDERS for growth should match GROWTH_SCORE_PROMPT."""
        for placeholder in PROMPT_PLACEHOLDERS["growth"]:
            assert (
                "{" + placeholder + "}" in GROWTH_SCORE_PROMPT
            ), f"Placeholder {placeholder} not in GROWTH_SCORE_PROMPT"

    def test_quality_placeholders_match_prompt(self):
        """PROMPT_PLACEHOLDERS for quality should match QUALITY_SCORE_PROMPT."""
        for placeholder in PROMPT_PLACEHOLDERS["quality"]:
            assert (
                "{" + placeholder + "}" in QUALITY_SCORE_PROMPT
            ), f"Placeholder {placeholder} not in QUALITY_SCORE_PROMPT"

    def test_momentum_placeholders_match_prompt(self):
        """PROMPT_PLACEHOLDERS for momentum should match MOMENTUM_SCORE_PROMPT."""
        for placeholder in PROMPT_PLACEHOLDERS["momentum"]:
            assert (
                "{" + placeholder + "}" in MOMENTUM_SCORE_PROMPT
            ), f"Placeholder {placeholder} not in MOMENTUM_SCORE_PROMPT"

    def test_sentiment_placeholders_match_prompt(self):
        """PROMPT_PLACEHOLDERS for sentiment should match SENTIMENT_SCORE_PROMPT."""
        for placeholder in PROMPT_PLACEHOLDERS["sentiment"]:
            assert (
                "{" + placeholder + "}" in SENTIMENT_SCORE_PROMPT
            ), f"Placeholder {placeholder} not in SENTIMENT_SCORE_PROMPT"
