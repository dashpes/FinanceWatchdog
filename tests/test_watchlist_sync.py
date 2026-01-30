"""Tests for WatchlistSync - syncs high-scoring candidates to portfolio.yaml watchlist."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from investment_monitor.storage import (
    ResearchReport,
    StockCandidate,
    get_session,
    init_db,
    save_candidate,
    save_report,
)
from investment_monitor.research import WatchlistSync


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


@pytest.fixture
def temp_portfolio_dir():
    """Create a temporary directory for portfolio.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def portfolio_yaml_path(temp_portfolio_dir):
    """Create a temporary portfolio.yaml file."""
    portfolio_path = temp_portfolio_dir / "portfolio.yaml"
    initial_data = {
        "holdings": [
            {"ticker": "AAPL", "shares": 100, "cost_basis": 150.0, "thesis": "Strong ecosystem"}
        ],
        "watchlist": [
            {"ticker": "MSFT", "reason": "Cloud growth", "target_price": 400.0}
        ],
    }
    with open(portfolio_path, "w") as f:
        yaml.safe_dump(initial_data, f)
    return portfolio_path


class TestAddCandidateToWatchlist:
    """Tests for add_candidate_to_watchlist method."""

    def test_add_candidate_to_watchlist_adds_new_ticker(
        self, db_session, portfolio_yaml_path
    ):
        """Test add_candidate_to_watchlist adds a new ticker to watchlist."""
        # Create a stock candidate
        candidate = StockCandidate(
            ticker="GOOGL",
            status="screening",
            composite_score=85.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        result = sync.add_candidate_to_watchlist(candidate)

        assert result is True
        # Verify the ticker was added to the YAML file
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        tickers = [item["ticker"] for item in data.get("watchlist", [])]
        assert "GOOGL" in tickers
        # Existing items should still be there
        assert "MSFT" in tickers

    def test_add_candidate_to_watchlist_returns_false_for_duplicate(
        self, db_session, portfolio_yaml_path
    ):
        """Test add_candidate_to_watchlist returns False when ticker already exists."""
        # MSFT is already in the watchlist
        candidate = StockCandidate(
            ticker="MSFT",
            status="screening",
            composite_score=90.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        result = sync.add_candidate_to_watchlist(candidate)

        assert result is False
        # Verify only one MSFT entry exists
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        msft_count = sum(1 for item in data.get("watchlist", []) if item["ticker"] == "MSFT")
        assert msft_count == 1

    def test_add_candidate_with_report_includes_summary(
        self, db_session, portfolio_yaml_path
    ):
        """Test add_candidate_to_watchlist stores metadata when report is provided."""
        candidate = StockCandidate(
            ticker="NVDA",
            status="researched",
            composite_score=92.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)

        report = ResearchReport(
            ticker="NVDA",
            summary="Strong AI chip demand driving growth",
            recommendation="buy",
            target_price=500.0,
        )
        save_report(db_session, report)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        result = sync.add_candidate_to_watchlist(candidate, report=report)

        assert result is True
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        nvda_items = [item for item in data.get("watchlist", []) if item["ticker"] == "NVDA"]
        assert len(nvda_items) == 1
        nvda_item = nvda_items[0]
        # Should have metadata from report
        assert "reason" in nvda_item or "summary" in nvda_item

    def test_add_candidate_stores_score_metadata(
        self, db_session, portfolio_yaml_path
    ):
        """Test add_candidate_to_watchlist stores score in metadata."""
        candidate = StockCandidate(
            ticker="AMD",
            status="screening",
            composite_score=78.5,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        result = sync.add_candidate_to_watchlist(candidate)

        assert result is True
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        amd_items = [item for item in data.get("watchlist", []) if item["ticker"] == "AMD"]
        assert len(amd_items) == 1
        amd_item = amd_items[0]
        # Should have score stored
        assert "score" in amd_item or "composite_score" in amd_item


class TestSyncFromCandidates:
    """Tests for sync_from_candidates method."""

    def test_sync_from_candidates_filters_by_score(
        self, db_session, portfolio_yaml_path
    ):
        """Test sync_from_candidates only adds candidates above min_score."""
        # Create candidates with various scores
        candidates = [
            StockCandidate(ticker="HIGH1", status="screening", composite_score=85.0),
            StockCandidate(ticker="HIGH2", status="screening", composite_score=75.0),
            StockCandidate(ticker="LOW1", status="screening", composite_score=60.0),
            StockCandidate(ticker="LOW2", status="screening", composite_score=50.0),
        ]
        for c in candidates:
            save_candidate(db_session, c)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        added_count = sync.sync_from_candidates(min_score=70.0)

        # Should add HIGH1 and HIGH2
        assert added_count == 2
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        tickers = [item["ticker"] for item in data.get("watchlist", [])]
        assert "HIGH1" in tickers
        assert "HIGH2" in tickers
        assert "LOW1" not in tickers
        assert "LOW2" not in tickers

    def test_sync_from_candidates_uses_default_min_score(
        self, db_session, portfolio_yaml_path
    ):
        """Test sync_from_candidates uses default min_score of 70.0."""
        candidates = [
            StockCandidate(ticker="ABOVE", status="screening", composite_score=71.0),
            StockCandidate(ticker="BELOW", status="screening", composite_score=69.0),
        ]
        for c in candidates:
            save_candidate(db_session, c)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        added_count = sync.sync_from_candidates()

        assert added_count == 1
        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)
        tickers = [item["ticker"] for item in data.get("watchlist", [])]
        assert "ABOVE" in tickers
        assert "BELOW" not in tickers

    def test_sync_from_candidates_skips_existing(
        self, db_session, portfolio_yaml_path
    ):
        """Test sync_from_candidates does not count already existing tickers."""
        # MSFT is already in the watchlist
        candidates = [
            StockCandidate(ticker="MSFT", status="screening", composite_score=80.0),
            StockCandidate(ticker="NEWSTOCK", status="screening", composite_score=75.0),
        ]
        for c in candidates:
            save_candidate(db_session, c)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        added_count = sync.sync_from_candidates(min_score=70.0)

        # Should only count NEWSTOCK as added
        assert added_count == 1


class TestHandleMissingPortfolio:
    """Tests for handling missing portfolio.yaml."""

    def test_handles_missing_portfolio_yaml(self, db_session, temp_portfolio_dir):
        """Test creates portfolio.yaml if it doesn't exist."""
        portfolio_path = temp_portfolio_dir / "portfolio.yaml"
        assert not portfolio_path.exists()

        candidate = StockCandidate(
            ticker="TSLA",
            status="screening",
            composite_score=82.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_path)
        result = sync.add_candidate_to_watchlist(candidate)

        assert result is True
        assert portfolio_path.exists()
        with open(portfolio_path) as f:
            data = yaml.safe_load(f)
        assert "watchlist" in data
        tickers = [item["ticker"] for item in data.get("watchlist", [])]
        assert "TSLA" in tickers

    def test_creates_minimal_structure_for_new_file(
        self, db_session, temp_portfolio_dir
    ):
        """Test newly created portfolio.yaml has minimal required structure."""
        portfolio_path = temp_portfolio_dir / "portfolio.yaml"

        candidate = StockCandidate(
            ticker="META",
            status="screening",
            composite_score=77.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_path)
        sync.add_candidate_to_watchlist(candidate)

        with open(portfolio_path) as f:
            data = yaml.safe_load(f)
        # Should have at minimum a watchlist section
        assert "watchlist" in data
        assert isinstance(data["watchlist"], list)


class TestPreservesExistingItems:
    """Tests for preserving existing watchlist items."""

    def test_preserves_existing_watchlist_items(
        self, db_session, portfolio_yaml_path
    ):
        """Test that adding new items preserves existing watchlist entries."""
        # Add a new candidate
        candidate = StockCandidate(
            ticker="NFLX",
            status="screening",
            composite_score=80.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        sync.add_candidate_to_watchlist(candidate)

        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)

        # Check original MSFT entry is preserved with all its fields
        msft_items = [item for item in data.get("watchlist", []) if item["ticker"] == "MSFT"]
        assert len(msft_items) == 1
        msft_item = msft_items[0]
        assert msft_item["reason"] == "Cloud growth"
        assert msft_item["target_price"] == 400.0

    def test_preserves_holdings_section(self, db_session, portfolio_yaml_path):
        """Test that adding watchlist items preserves holdings section."""
        candidate = StockCandidate(
            ticker="SPOT",
            status="screening",
            composite_score=72.0,
            discovery_source="test",
        )
        save_candidate(db_session, candidate)
        db_session.commit()

        sync = WatchlistSync(db_session, portfolio_path=portfolio_yaml_path)
        sync.add_candidate_to_watchlist(candidate)

        with open(portfolio_yaml_path) as f:
            data = yaml.safe_load(f)

        # Holdings should be unchanged
        assert "holdings" in data
        assert len(data["holdings"]) == 1
        assert data["holdings"][0]["ticker"] == "AAPL"
        assert data["holdings"][0]["shares"] == 100
