"""Tests for data models."""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from investment_monitor.models import (
    AlertsConfig,
    EarningsAlertSettings,
    ETFAlertSettings,
    Holding,
    InsiderAlertSettings,
    NewsAlertSettings,
    Portfolio,
    PriceAlertSettings,
    VolumeAlertSettings,
    WatchlistItem,
)


def test_holding_validation():
    """Test holding validation."""
    holding = Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00"))
    assert holding.ticker == "AAPL"
    assert holding.total_cost == Decimal("8250.00")


def test_holding_invalid_ticker():
    """Test that invalid tickers are rejected."""
    with pytest.raises(ValueError):
        Holding(ticker="aapl", shares=Decimal("50"), cost_basis=Decimal("165.00"))

    with pytest.raises(ValueError):
        Holding(ticker="TOOLONG", shares=Decimal("50"), cost_basis=Decimal("165.00"))


def test_holding_invalid_shares():
    """Test that invalid shares are rejected."""
    with pytest.raises(ValueError):
        Holding(ticker="AAPL", shares=Decimal("0"), cost_basis=Decimal("165.00"))

    with pytest.raises(ValueError):
        Holding(ticker="AAPL", shares=Decimal("-10"), cost_basis=Decimal("165.00"))


def test_watchlist_item():
    """Test watchlist item."""
    item = WatchlistItem(
        ticker="GOOGL", reason="Waiting for entry", target_price=Decimal("140.00")
    )
    assert item.ticker == "GOOGL"
    assert item.target_price == Decimal("140.00")


def test_portfolio_all_tickers():
    """Test all_tickers computed property."""
    portfolio = Portfolio(
        holdings=[
            Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00")),
            Holding(ticker="MSFT", shares=Decimal("30"), cost_basis=Decimal("380.00")),
        ],
        watchlist=[
            WatchlistItem(ticker="GOOGL", reason="Watching"),
            WatchlistItem(ticker="AAPL", reason="Already own"),  # Duplicate
        ],
    )

    assert portfolio.all_tickers == ["AAPL", "GOOGL", "MSFT"]
    assert portfolio.holding_tickers == ["AAPL", "MSFT"]


def test_portfolio_get_holding():
    """Test get_holding method."""
    portfolio = Portfolio(
        holdings=[
            Holding(ticker="AAPL", shares=Decimal("50"), cost_basis=Decimal("165.00")),
        ]
    )

    holding = portfolio.get_holding("AAPL")
    assert holding is not None
    assert holding.shares == Decimal("50")

    assert portfolio.get_holding("MSFT") is None


def test_portfolio_get_thesis():
    """Test get_thesis method."""
    portfolio = Portfolio(
        holdings=[
            Holding(
                ticker="AAPL",
                shares=Decimal("50"),
                cost_basis=Decimal("165.00"),
                thesis="Services growth",
            ),
            Holding(ticker="MSFT", shares=Decimal("30"), cost_basis=Decimal("380.00")),
        ]
    )

    assert portfolio.get_thesis("AAPL") == "Services growth"
    assert portfolio.get_thesis("MSFT") is None  # Empty thesis
    assert portfolio.get_thesis("GOOGL") is None  # Not in portfolio


def test_portfolio_from_yaml():
    """Test loading portfolio from YAML."""
    yaml_content = """
holdings:
  - ticker: AAPL
    shares: 50
    cost_basis: 165.00
    thesis: "Services growth"

watchlist:
  - ticker: GOOGL
    reason: "Waiting for entry"
    target_price: 140.00
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        portfolio = Portfolio.from_yaml(Path(f.name))
        assert len(portfolio.holdings) == 1
        assert len(portfolio.watchlist) == 1
        assert portfolio.holdings[0].ticker == "AAPL"


# ============================================================================
# Alert Settings Model Tests
# ============================================================================


class TestPriceAlertSettings:
    """Tests for PriceAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = PriceAlertSettings()
        assert settings.enabled is True
        assert settings.daily_drop_pct == 3.0
        assert settings.daily_rise_pct == 5.0
        assert settings.weekly_drop_pct == 7.0
        assert settings.below_cost_basis is True

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = PriceAlertSettings(
            enabled=False,
            daily_drop_pct=5.0,
            daily_rise_pct=10.0,
            weekly_drop_pct=15.0,
            below_cost_basis=False,
        )
        assert settings.enabled is False
        assert settings.daily_drop_pct == 5.0
        assert settings.daily_rise_pct == 10.0
        assert settings.weekly_drop_pct == 15.0
        assert settings.below_cost_basis is False

    def test_invalid_negative_percentage(self):
        """Test that negative percentages are rejected."""
        with pytest.raises(ValueError):
            PriceAlertSettings(daily_drop_pct=-1.0)

    def test_invalid_percentage_over_100(self):
        """Test that percentages over 100 are rejected."""
        with pytest.raises(ValueError):
            PriceAlertSettings(daily_rise_pct=101.0)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        settings = PriceAlertSettings(
            daily_drop_pct=0.0,
            daily_rise_pct=100.0,
            weekly_drop_pct=0.0,
        )
        assert settings.daily_drop_pct == 0.0
        assert settings.daily_rise_pct == 100.0


class TestVolumeAlertSettings:
    """Tests for VolumeAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = VolumeAlertSettings()
        assert settings.enabled is True
        assert settings.lookback_days == 20
        assert settings.multiplier == 2.5

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = VolumeAlertSettings(
            enabled=False,
            lookback_days=30,
            multiplier=3.0,
        )
        assert settings.enabled is False
        assert settings.lookback_days == 30
        assert settings.multiplier == 3.0

    def test_invalid_lookback_too_low(self):
        """Test that lookback days below 5 are rejected."""
        with pytest.raises(ValueError):
            VolumeAlertSettings(lookback_days=4)

    def test_invalid_lookback_too_high(self):
        """Test that lookback days above 60 are rejected."""
        with pytest.raises(ValueError):
            VolumeAlertSettings(lookback_days=61)

    def test_invalid_multiplier_below_one(self):
        """Test that multiplier below 1.0 is rejected."""
        with pytest.raises(ValueError):
            VolumeAlertSettings(multiplier=0.5)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        settings = VolumeAlertSettings(lookback_days=5, multiplier=1.0)
        assert settings.lookback_days == 5
        assert settings.multiplier == 1.0

        settings = VolumeAlertSettings(lookback_days=60)
        assert settings.lookback_days == 60


class TestInsiderAlertSettings:
    """Tests for InsiderAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = InsiderAlertSettings()
        assert settings.enabled is True
        assert settings.min_buy_value == 100_000
        assert settings.min_sell_value == 500_000
        assert settings.alert_ceo_cfo_any is True
        assert settings.cluster_threshold == 3
        assert settings.cluster_days == 7

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = InsiderAlertSettings(
            enabled=False,
            min_buy_value=50_000,
            min_sell_value=250_000,
            alert_ceo_cfo_any=False,
            cluster_threshold=5,
            cluster_days=14,
        )
        assert settings.enabled is False
        assert settings.min_buy_value == 50_000
        assert settings.min_sell_value == 250_000
        assert settings.alert_ceo_cfo_any is False
        assert settings.cluster_threshold == 5
        assert settings.cluster_days == 14

    def test_invalid_negative_value(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            InsiderAlertSettings(min_buy_value=-1)

    def test_invalid_cluster_threshold_too_low(self):
        """Test that cluster threshold below 2 is rejected."""
        with pytest.raises(ValueError):
            InsiderAlertSettings(cluster_threshold=1)

    def test_invalid_cluster_days_zero(self):
        """Test that cluster days of 0 is rejected."""
        with pytest.raises(ValueError):
            InsiderAlertSettings(cluster_days=0)

    def test_zero_min_values_allowed(self):
        """Test that zero is allowed for min values."""
        settings = InsiderAlertSettings(min_buy_value=0, min_sell_value=0)
        assert settings.min_buy_value == 0
        assert settings.min_sell_value == 0


class TestEarningsAlertSettings:
    """Tests for EarningsAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = EarningsAlertSettings()
        assert settings.enabled is True
        assert settings.lookahead_days == 7

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = EarningsAlertSettings(enabled=False, lookahead_days=14)
        assert settings.enabled is False
        assert settings.lookahead_days == 14

    def test_invalid_lookahead_too_low(self):
        """Test that lookahead days below 1 is rejected."""
        with pytest.raises(ValueError):
            EarningsAlertSettings(lookahead_days=0)

    def test_invalid_lookahead_too_high(self):
        """Test that lookahead days above 30 is rejected."""
        with pytest.raises(ValueError):
            EarningsAlertSettings(lookahead_days=31)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        settings = EarningsAlertSettings(lookahead_days=1)
        assert settings.lookahead_days == 1

        settings = EarningsAlertSettings(lookahead_days=30)
        assert settings.lookahead_days == 30


class TestNewsAlertSettings:
    """Tests for NewsAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = NewsAlertSettings()
        assert settings.enabled is True
        assert "lawsuit" in settings.keywords
        assert "SEC" in settings.keywords
        assert "investigation" in settings.keywords
        assert "guidance" in settings.keywords
        assert "acquisition" in settings.keywords
        assert "merger" in settings.keywords
        assert "layoffs" in settings.keywords
        assert "dividend" in settings.keywords
        assert "buyback" in settings.keywords
        assert len(settings.keywords) == 9
        assert settings.min_relevance_score == 5.0

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = NewsAlertSettings(
            enabled=False,
            keywords=["bankruptcy", "fraud"],
            min_relevance_score=7.5,
        )
        assert settings.enabled is False
        assert settings.keywords == ["bankruptcy", "fraud"]
        assert settings.min_relevance_score == 7.5

    def test_invalid_relevance_score_negative(self):
        """Test that negative relevance score is rejected."""
        with pytest.raises(ValueError):
            NewsAlertSettings(min_relevance_score=-1.0)

    def test_invalid_relevance_score_too_high(self):
        """Test that relevance score above 10 is rejected."""
        with pytest.raises(ValueError):
            NewsAlertSettings(min_relevance_score=10.1)

    def test_boundary_values(self):
        """Test boundary values are accepted."""
        settings = NewsAlertSettings(min_relevance_score=0.0)
        assert settings.min_relevance_score == 0.0

        settings = NewsAlertSettings(min_relevance_score=10.0)
        assert settings.min_relevance_score == 10.0

    def test_empty_keywords_allowed(self):
        """Test that empty keywords list is allowed."""
        settings = NewsAlertSettings(keywords=[])
        assert settings.keywords == []


class TestETFAlertSettings:
    """Tests for ETFAlertSettings model."""

    def test_defaults(self):
        """Test default values are sensible."""
        settings = ETFAlertSettings()
        assert settings.enabled is True
        assert settings.holdings_change is True
        assert settings.weight_change_pct == 1.0

    def test_custom_values(self):
        """Test custom values are accepted."""
        settings = ETFAlertSettings(
            enabled=False,
            holdings_change=False,
            weight_change_pct=2.5,
        )
        assert settings.enabled is False
        assert settings.holdings_change is False
        assert settings.weight_change_pct == 2.5

    def test_invalid_weight_change_negative(self):
        """Test that negative weight change is rejected."""
        with pytest.raises(ValueError):
            ETFAlertSettings(weight_change_pct=-0.1)

    def test_zero_weight_change_allowed(self):
        """Test that zero weight change is allowed."""
        settings = ETFAlertSettings(weight_change_pct=0.0)
        assert settings.weight_change_pct == 0.0


class TestAlertsConfig:
    """Tests for AlertsConfig model."""

    def test_defaults(self):
        """Test default values create valid nested settings."""
        config = AlertsConfig()
        assert config.price.enabled is True
        assert config.volume.enabled is True
        assert config.insider.enabled is True
        assert config.earnings.enabled is True
        assert config.news.enabled is True
        assert config.etf.enabled is True

    def test_partial_override(self):
        """Test that partial configuration merges with defaults."""
        config = AlertsConfig(
            price=PriceAlertSettings(enabled=False, daily_drop_pct=5.0),
            volume=VolumeAlertSettings(multiplier=3.0),
        )
        # Overridden values
        assert config.price.enabled is False
        assert config.price.daily_drop_pct == 5.0
        assert config.volume.multiplier == 3.0

        # Default values preserved
        assert config.price.daily_rise_pct == 5.0  # default
        assert config.volume.enabled is True  # default
        assert config.insider.enabled is True  # entire section defaults

    def test_disable_all_alerts(self):
        """Test that all alert types can be disabled."""
        config = AlertsConfig(
            price=PriceAlertSettings(enabled=False),
            volume=VolumeAlertSettings(enabled=False),
            insider=InsiderAlertSettings(enabled=False),
            earnings=EarningsAlertSettings(enabled=False),
            news=NewsAlertSettings(enabled=False),
            etf=ETFAlertSettings(enabled=False),
        )
        assert config.price.enabled is False
        assert config.volume.enabled is False
        assert config.insider.enabled is False
        assert config.earnings.enabled is False
        assert config.news.enabled is False
        assert config.etf.enabled is False

    def test_from_yaml_full_config(self):
        """Test loading a full configuration from YAML."""
        yaml_content = """
price:
  enabled: true
  daily_drop_pct: 4.0
  daily_rise_pct: 6.0
  weekly_drop_pct: 10.0
  below_cost_basis: false

volume:
  enabled: true
  lookback_days: 30
  multiplier: 3.0

insider:
  enabled: false
  min_buy_value: 200000
  min_sell_value: 1000000

earnings:
  lookahead_days: 14

news:
  keywords:
    - bankruptcy
    - scandal
  min_relevance_score: 7.0

etf:
  holdings_change: false
  weight_change_pct: 2.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AlertsConfig.from_yaml(Path(f.name))

            # Price settings
            assert config.price.enabled is True
            assert config.price.daily_drop_pct == 4.0
            assert config.price.daily_rise_pct == 6.0
            assert config.price.weekly_drop_pct == 10.0
            assert config.price.below_cost_basis is False

            # Volume settings
            assert config.volume.enabled is True
            assert config.volume.lookback_days == 30
            assert config.volume.multiplier == 3.0

            # Insider settings
            assert config.insider.enabled is False
            assert config.insider.min_buy_value == 200_000
            assert config.insider.min_sell_value == 1_000_000

            # Earnings settings
            assert config.earnings.lookahead_days == 14
            assert config.earnings.enabled is True  # default

            # News settings
            assert config.news.keywords == ["bankruptcy", "scandal"]
            assert config.news.min_relevance_score == 7.0

            # ETF settings
            assert config.etf.holdings_change is False
            assert config.etf.weight_change_pct == 2.0

    def test_from_yaml_partial_config(self):
        """Test loading a partial configuration from YAML merges with defaults."""
        yaml_content = """
price:
  daily_drop_pct: 2.0

news:
  enabled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AlertsConfig.from_yaml(Path(f.name))

            # Specified values
            assert config.price.daily_drop_pct == 2.0
            assert config.news.enabled is False

            # Defaults for other price fields
            assert config.price.enabled is True
            assert config.price.daily_rise_pct == 5.0
            assert config.price.weekly_drop_pct == 7.0

            # Defaults for entire sections
            assert config.volume.enabled is True
            assert config.volume.lookback_days == 20
            assert config.insider.enabled is True
            assert config.earnings.lookahead_days == 7
            assert config.etf.enabled is True

    def test_from_yaml_empty_file(self):
        """Test loading from an empty YAML file uses all defaults."""
        yaml_content = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AlertsConfig.from_yaml(Path(f.name))

            # All defaults
            assert config.price.enabled is True
            assert config.price.daily_drop_pct == 3.0
            assert config.volume.lookback_days == 20
            assert config.insider.min_buy_value == 100_000
            assert config.earnings.lookahead_days == 7
            assert len(config.news.keywords) == 9
            assert config.etf.weight_change_pct == 1.0

    def test_from_yaml_invalid_values_rejected(self):
        """Test that invalid values in YAML are rejected."""
        yaml_content = """
price:
  daily_drop_pct: -5.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError):
                AlertsConfig.from_yaml(Path(f.name))
