"""SQLAlchemy ORM models for the robo advisor (rebalance runs and orders)."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base


class RoboRun(Base):
    """One execution of the rebalance pipeline (dry-run or live)."""

    __tablename__ = "robo_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    account_id: Mapped[str] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=True)  # llm | deterministic
    total_value: Mapped[float] = mapped_column(Float, nullable=True)
    settled_cash: Mapped[float] = mapped_column(Float, nullable=True)
    num_proposed: Mapped[int] = mapped_column(Integer, default=0)
    num_accepted: Mapped[int] = mapped_column(Integer, default=0)
    num_rejected: Mapped[int] = mapped_column(Integer, default=0)
    num_placed: Mapped[int] = mapped_column(Integer, default=0)
    # running | completed | failed | refused
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class RoboOrder(Base):
    """A single candidate order and its full lifecycle (gate -> preflight -> place)."""

    __tablename__ = "robo_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(12), nullable=False)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=True)  # llm | deterministic
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    gate_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gate_code: Mapped[str] = mapped_column(String(40), nullable=True)
    gate_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    preflight_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    preflight_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    placed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
