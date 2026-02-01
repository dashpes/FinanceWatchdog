"""Tests for simulation result storage."""

import json
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from investment_monitor.storage.database import Base
from investment_monitor.storage.research_models import SimulationResult


@pytest.fixture
def session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestSimulationResultModel:
    """Tests for SimulationResult ORM model."""

    def test_create_simulation_result(self, session: Session):
        result = SimulationResult(
            ticker="AAPL",
            run_date=date(2026, 1, 31),
            entry_price=178.50,
            composite_score=85.0,
            num_simulations=10000,
            lookback_days=756,
            volatility=0.25,
            drift=0.08,
            results_30d={"base_mean": 182.0, "scenarios": {}},
            results_90d={"base_mean": 189.0, "scenarios": {}},
            results_252d={"base_mean": 198.0, "scenarios": {}},
            sensitivity_analysis={"primary_driver": "volatility"},
        )
        session.add(result)
        session.commit()

        retrieved = session.query(SimulationResult).filter_by(ticker="AAPL").first()
        assert retrieved is not None
        assert retrieved.ticker == "AAPL"
        assert retrieved.entry_price == 178.50
        assert retrieved.composite_score == 85.0
        assert retrieved.results_30d["base_mean"] == 182.0

    def test_simulation_result_auto_timestamps(self, session: Session):
        result = SimulationResult(
            ticker="MSFT",
            run_date=date(2026, 1, 31),
            entry_price=400.00,
            composite_score=82.0,
            num_simulations=5000,
            lookback_days=504,
            volatility=0.22,
            drift=0.10,
            results_30d={},
            results_90d={},
            results_252d={},
            sensitivity_analysis={},
        )
        session.add(result)
        session.commit()

        assert result.created_at is not None
        assert isinstance(result.created_at, datetime)

    def test_multiple_simulations_for_ticker(self, session: Session):
        """A ticker can have multiple simulation runs over time."""
        for i in range(3):
            result = SimulationResult(
                ticker="GOOGL",
                run_date=date(2026, 1, 31 - i),
                entry_price=150.0 + i,
                composite_score=80.0 + i,
                num_simulations=1000,
                lookback_days=252,
                volatility=0.20,
                drift=0.05,
                results_30d={},
                results_90d={},
                results_252d={},
                sensitivity_analysis={},
            )
            session.add(result)
        session.commit()

        results = session.query(SimulationResult).filter_by(ticker="GOOGL").all()
        assert len(results) == 3
