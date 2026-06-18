"""CRUD operations for robo advisor runs and orders."""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .robo_models import RoboOrder, RoboRun


def save_robo_run(session: Session, run: RoboRun) -> int:
    """Persist a run row, returning its ID."""
    session.add(run)
    session.flush()
    return run.id


def finalize_robo_run(session: Session, run_id: str, **fields) -> None:
    """Update an existing run by ``run_id`` with the provided fields."""
    stmt = select(RoboRun).where(RoboRun.run_id == run_id)
    run = session.scalar(stmt)
    if run is None:
        return
    for key, value in fields.items():
        setattr(run, key, value)
    session.flush()


def save_robo_order(session: Session, order: RoboOrder) -> int:
    """Persist an order row, returning its ID."""
    session.add(order)
    session.flush()
    return order.id


def get_recent_robo_runs(session: Session, limit: int = 10) -> list[RoboRun]:
    """Most recent runs, newest first."""
    stmt = select(RoboRun).order_by(RoboRun.started_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def get_robo_orders_for_run(session: Session, run_id: str) -> list[RoboOrder]:
    """All order rows for a given run."""
    stmt = (
        select(RoboOrder)
        .where(RoboOrder.run_id == run_id)
        .order_by(RoboOrder.created_at.asc())
    )
    return list(session.scalars(stmt))


def count_placed_orders_today(session: Session) -> int:
    """Count orders actually PLACED at the broker today (UTC) — the per-day cap.

    Only real placements count toward the cap. Simulated (dry-run), market-closed-
    deferred, and failed orders do NOT, so a day of paper runs never exhausts the
    live order budget. The cap is a real-money rate limit, so it tracks real trades.
    """
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stmt = (
        select(func.count())
        .select_from(RoboOrder)
        .where(RoboOrder.placed.is_(True))
        .where(RoboOrder.created_at >= start_of_day)
    )
    return int(session.scalar(stmt) or 0)
