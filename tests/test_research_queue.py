"""Tests for ResearchQueue - manages pending deep research candidates."""

import tempfile
from pathlib import Path

import pytest

from investment_monitor.storage import (
    StockCandidate,
    get_session,
    init_db,
    save_candidate,
)
from investment_monitor.research import ResearchQueue


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        with get_session() as session:
            yield session


class TestAddToQueue:
    """Tests for add_to_queue method."""

    def test_add_to_queue_creates_candidate(self, db_session):
        """Test add_to_queue creates a new candidate with screening status."""
        queue = ResearchQueue(db_session)

        result = queue.add_to_queue("AAPL", priority=80)

        assert result is True
        # Verify candidate was created with correct status and score
        candidate = queue.get_next()
        assert candidate is not None
        assert candidate.ticker == "AAPL"
        assert candidate.status == "screening"
        assert candidate.composite_score == 80

    def test_add_to_queue_updates_existing_candidate(self, db_session):
        """Test add_to_queue updates existing candidate to screening status."""
        # Pre-create a candidate with "discovered" status
        existing = StockCandidate(
            ticker="MSFT",
            status="discovered",
            composite_score=50.0,
            discovery_source="test",
        )
        save_candidate(db_session, existing)
        db_session.commit()

        queue = ResearchQueue(db_session)
        result = queue.add_to_queue("MSFT", priority=75)

        assert result is True
        # Verify status was updated
        candidate = queue.get_next()
        assert candidate is not None
        assert candidate.ticker == "MSFT"
        assert candidate.status == "screening"
        assert candidate.composite_score == 75

    def test_add_to_queue_default_priority(self, db_session):
        """Test add_to_queue uses default priority of 0."""
        queue = ResearchQueue(db_session)

        result = queue.add_to_queue("GOOGL")

        assert result is True
        candidate = queue.get_next()
        assert candidate is not None
        assert candidate.composite_score == 0


class TestGetNext:
    """Tests for get_next method."""

    def test_get_next_returns_highest_priority(self, db_session):
        """Test get_next returns candidate with highest priority (composite_score)."""
        queue = ResearchQueue(db_session)
        queue.add_to_queue("LOW", priority=30)
        queue.add_to_queue("HIGH", priority=90)
        queue.add_to_queue("MED", priority=60)
        db_session.commit()

        next_candidate = queue.get_next()

        assert next_candidate is not None
        assert next_candidate.ticker == "HIGH"
        assert next_candidate.composite_score == 90

    def test_get_next_empty_queue_returns_none(self, db_session):
        """Test get_next returns None when queue is empty."""
        queue = ResearchQueue(db_session)

        result = queue.get_next()

        assert result is None

    def test_get_next_ignores_non_screening_status(self, db_session):
        """Test get_next only returns candidates with screening status."""
        # Create candidates with various statuses
        candidates = [
            StockCandidate(ticker="DISC", status="discovered", composite_score=95.0),
            StockCandidate(ticker="SCREEN", status="screening", composite_score=70.0),
            StockCandidate(ticker="WATCH", status="watchlist", composite_score=85.0),
            StockCandidate(ticker="REJ", status="rejected", composite_score=20.0),
        ]
        for c in candidates:
            save_candidate(db_session, c)
        db_session.commit()

        queue = ResearchQueue(db_session)
        next_candidate = queue.get_next()

        assert next_candidate is not None
        assert next_candidate.ticker == "SCREEN"


class TestGetQueue:
    """Tests for get_queue method."""

    def test_get_queue_returns_ordered_list(self, db_session):
        """Test get_queue returns candidates ordered by priority (highest first)."""
        queue = ResearchQueue(db_session)
        queue.add_to_queue("C", priority=50)
        queue.add_to_queue("A", priority=80)
        queue.add_to_queue("B", priority=65)
        db_session.commit()

        result = queue.get_queue()

        assert len(result) == 3
        assert result[0].ticker == "A"
        assert result[1].ticker == "B"
        assert result[2].ticker == "C"

    def test_get_queue_respects_limit(self, db_session):
        """Test get_queue respects the limit parameter."""
        queue = ResearchQueue(db_session)
        for i in range(25):
            queue.add_to_queue(f"T{i:02d}", priority=i)
        db_session.commit()

        result = queue.get_queue(limit=10)

        assert len(result) == 10
        # Should have highest priority items
        assert result[0].ticker == "T24"

    def test_get_queue_default_limit(self, db_session):
        """Test get_queue has default limit of 20."""
        queue = ResearchQueue(db_session)
        for i in range(30):
            queue.add_to_queue(f"T{i:02d}", priority=i)
        db_session.commit()

        result = queue.get_queue()

        assert len(result) == 20

    def test_get_queue_empty_returns_empty_list(self, db_session):
        """Test get_queue returns empty list when queue is empty."""
        queue = ResearchQueue(db_session)

        result = queue.get_queue()

        assert result == []


class TestRemoveFromQueue:
    """Tests for remove_from_queue method."""

    def test_remove_from_queue_changes_status(self, db_session):
        """Test remove_from_queue changes status from screening."""
        queue = ResearchQueue(db_session)
        queue.add_to_queue("AAPL", priority=80)
        db_session.commit()

        result = queue.remove_from_queue("AAPL")

        assert result is True
        # Verify it's no longer in the queue
        assert queue.get_next() is None

    def test_remove_from_queue_nonexistent_returns_false(self, db_session):
        """Test remove_from_queue returns False for non-existent ticker."""
        queue = ResearchQueue(db_session)

        result = queue.remove_from_queue("NOTHERE")

        assert result is False

    def test_remove_from_queue_updates_to_discovered(self, db_session):
        """Test remove_from_queue sets status back to discovered."""
        queue = ResearchQueue(db_session)
        queue.add_to_queue("AAPL", priority=80)
        db_session.commit()

        queue.remove_from_queue("AAPL")
        db_session.commit()

        # Verify status is now "discovered"
        from investment_monitor.storage import get_candidate_by_ticker

        candidate = get_candidate_by_ticker(db_session, "AAPL")
        assert candidate is not None
        assert candidate.status == "discovered"


class TestConcurrentAccess:
    """Tests for concurrent access handling."""

    def test_queue_operations_are_session_safe(self, db_session):
        """Test that queue operations use the session properly."""
        queue = ResearchQueue(db_session)

        # Add and immediately query without explicit commit
        queue.add_to_queue("AAPL", priority=80)
        db_session.flush()

        # Should be visible within same session
        candidate = queue.get_next()
        assert candidate is not None
        assert candidate.ticker == "AAPL"
