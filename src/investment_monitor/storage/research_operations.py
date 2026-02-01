"""CRUD operations for research database models."""

from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .research_models import (
    CandidateScore,
    CongressionalTrade,
    PerformanceTracker,
    ResearchProfile,
    ResearchReport,
    SimulationResult,
    StockCandidate,
)


# Profile operations
def get_or_create_default_profile(session: Session) -> ResearchProfile:
    """Get the default profile or create one if none exists."""
    stmt = select(ResearchProfile).where(ResearchProfile.name == "default")
    profile = session.scalar(stmt)
    if profile is None:
        profile = ResearchProfile(name="default")
        session.add(profile)
        session.flush()
    return profile


def save_profile(session: Session, profile: ResearchProfile) -> int:
    """Save/update a profile, returning its ID."""
    session.add(profile)
    session.flush()
    return profile.id


# Candidate operations
def save_candidate(session: Session, candidate: StockCandidate) -> int:
    """Save a stock candidate, returning its ID."""
    session.add(candidate)
    session.flush()
    return candidate.id


def get_candidate_by_ticker(session: Session, ticker: str) -> StockCandidate | None:
    """Get candidate by ticker."""
    stmt = select(StockCandidate).where(StockCandidate.ticker == ticker)
    return session.scalar(stmt)


def get_candidates_by_status(
    session: Session, status: str, limit: int = 100
) -> list[StockCandidate]:
    """Get candidates with given status."""
    stmt = (
        select(StockCandidate)
        .where(StockCandidate.status == status)
        .order_by(StockCandidate.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_top_candidates(
    session: Session, limit: int = 20, min_score: float | None = None
) -> list[StockCandidate]:
    """Get top scoring candidates."""
    stmt = select(StockCandidate).where(StockCandidate.composite_score.isnot(None))
    if min_score is not None:
        stmt = stmt.where(StockCandidate.composite_score >= min_score)
    stmt = stmt.order_by(StockCandidate.composite_score.desc()).limit(limit)
    return list(session.scalars(stmt))


# Score operations
def save_score(session: Session, score: CandidateScore) -> int:
    """Save a candidate score, returning its ID."""
    session.add(score)
    session.flush()
    return score.id


def get_latest_score(session: Session, ticker: str) -> CandidateScore | None:
    """Get latest score for ticker."""
    stmt = (
        select(CandidateScore)
        .where(CandidateScore.ticker == ticker)
        .order_by(CandidateScore.created_at.desc(), CandidateScore.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def get_score_history(
    session: Session, ticker: str, limit: int = 10
) -> list[CandidateScore]:
    """Get score history for a ticker."""
    stmt = (
        select(CandidateScore)
        .where(CandidateScore.ticker == ticker)
        .order_by(CandidateScore.created_at.desc(), CandidateScore.id.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


# Report operations
def save_report(session: Session, report: ResearchReport) -> int:
    """Save a research report, returning its ID."""
    session.add(report)
    session.flush()
    return report.id


def get_latest_report(session: Session, ticker: str) -> ResearchReport | None:
    """Get latest report for ticker."""
    stmt = (
        select(ResearchReport)
        .where(ResearchReport.ticker == ticker)
        .order_by(ResearchReport.created_at.desc(), ResearchReport.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


# Performance operations
def save_performance_record(session: Session, record: PerformanceTracker) -> int:
    """Save a performance record, returning its ID."""
    session.add(record)
    session.flush()
    return record.id


def get_records_needing_update(
    session: Session, days_since_update: int = 7
) -> list[PerformanceTracker]:
    """Get records that need return updates.

    Returns records that either:
    - Have never been updated (updated_at == created_at)
    - Haven't been updated in the specified number of days
    """
    cutoff = datetime.now() - timedelta(days=days_since_update)
    stmt = (
        select(PerformanceTracker)
        .where(PerformanceTracker.updated_at <= cutoff)
        .order_by(PerformanceTracker.updated_at.asc())
    )
    return list(session.scalars(stmt))


# Congressional trade operations
def save_congressional_trade(session: Session, trade: CongressionalTrade) -> int:
    """Save a congressional trade, returning its ID."""
    session.add(trade)
    session.flush()
    return trade.id


def get_trades_for_ticker(
    session: Session, ticker: str, days: int = 90
) -> list[CongressionalTrade]:
    """Get recent trades for a ticker."""
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(CongressionalTrade)
        .where(
            CongressionalTrade.ticker == ticker,
            CongressionalTrade.trade_date >= cutoff,
        )
        .order_by(CongressionalTrade.trade_date.desc())
    )
    return list(session.scalars(stmt))


# Simulation operations
def save_simulation_result(
    session: Session, output: "SimulationOutput"
) -> SimulationResult:
    """Save a SimulationOutput to the database.

    Converts the SimulationOutput Pydantic model to a SimulationResult
    ORM model and persists it.

    Args:
        session: Database session.
        output: SimulationOutput from MonteCarloAnalyzer.

    Returns:
        The saved SimulationResult with its ID.
    """
    from investment_monitor.simulation.models import SimulationOutput

    # Convert horizon results to JSON-serializable dicts
    def horizon_to_dict(horizon_result) -> dict:
        """Convert HorizonResult to JSON-serializable dict."""
        scenarios_dict = {}
        for name, scenario in horizon_result.scenarios.items():
            scenarios_dict[name] = {
                "name": scenario.name,
                "mean": scenario.mean,
                "median": scenario.median,
                "std": scenario.std,
                "ci_80": scenario.ci_80,
                "ci_95": scenario.ci_95,
                "var_95": scenario.var_95,
                "cvar_95": scenario.cvar_95,
                "prob_loss_20pct": scenario.prob_loss_20pct,
            }

        return {
            "days": horizon_result.days,
            "base_mean": horizon_result.base_mean,
            "base_median": horizon_result.base_median,
            "base_std": horizon_result.base_std,
            "base_skewness": horizon_result.base_skewness,
            "base_percentiles": horizon_result.base_percentiles,
            "base_ci_80": horizon_result.base_ci_80,
            "base_ci_95": horizon_result.base_ci_95,
            "base_var_95": horizon_result.base_var_95,
            "base_cvar_95": horizon_result.base_cvar_95,
            "scenarios": scenarios_dict,
        }

    # Get results for each horizon (or empty dict if not present)
    results_30d = horizon_to_dict(output.results[30]) if 30 in output.results else {}
    results_90d = horizon_to_dict(output.results[90]) if 90 in output.results else {}
    results_252d = horizon_to_dict(output.results[252]) if 252 in output.results else {}

    # Convert sensitivity to dict
    sensitivity_dict = {
        "volatility_impact": output.sensitivity.volatility_impact,
        "drift_impact": output.sensitivity.drift_impact,
        "lookback_impact": output.sensitivity.lookback_impact,
        "primary_driver": output.sensitivity.primary_driver,
        "volatility_range": {
            str(k): v for k, v in output.sensitivity.volatility_range.items()
        },
        "drift_range": output.sensitivity.drift_range,
        "lookback_range": {
            str(k): v for k, v in output.sensitivity.lookback_range.items()
        },
    }

    result = SimulationResult(
        ticker=output.ticker,
        run_date=date.today(),
        entry_price=output.entry_price,
        composite_score=output.composite_score,
        num_simulations=output.num_simulations,
        lookback_days=output.lookback_days,
        volatility=output.volatility,
        drift=output.drift,
        results_30d=results_30d,
        results_90d=results_90d,
        results_252d=results_252d,
        sensitivity_analysis=sensitivity_dict,
    )

    session.add(result)
    session.flush()
    return result


def get_simulation_results(
    session: Session,
    ticker: str | None = None,
    limit: int = 10,
) -> list[SimulationResult]:
    """Get simulation results, optionally filtered by ticker.

    Args:
        session: Database session.
        ticker: Optional ticker to filter by.
        limit: Maximum number of results to return.

    Returns:
        List of SimulationResult ordered by created_at descending.
    """
    stmt = select(SimulationResult)
    if ticker is not None:
        stmt = stmt.where(SimulationResult.ticker == ticker)
    stmt = stmt.order_by(SimulationResult.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def get_high_scoring_candidates(
    session: Session,
    min_score: float = 80.0,
) -> list[StockCandidate]:
    """Get candidates with composite score >= min_score.

    Args:
        session: Database session.
        min_score: Minimum composite score threshold.

    Returns:
        List of StockCandidate meeting the threshold, ordered by score descending.
    """
    stmt = (
        select(StockCandidate)
        .where(StockCandidate.composite_score >= min_score)
        .order_by(StockCandidate.composite_score.desc())
    )
    return list(session.scalars(stmt))
