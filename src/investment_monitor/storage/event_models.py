"""SQLAlchemy ORM model for SEC 8-K material corporate events.

One compact row per 8-K filing market-wide: which company, which Item codes
(1.01 material agreement, 5.02 exec departure, ...), when, and the EDGAR URL.
This is the raw event stream the event-driven thesis vision needs — today the
system can only see a material event indirectly, hours later, as a volume spike.

Brand-new table (never an ``ALTER``) so ``Base.metadata.create_all`` auto-creates
it on the live DB with zero migration.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models import Base

# Item codes that mark a HIGH-SIGNAL corporate event — the subset confluence treats
# as a corroborating source. Deliberately excludes the routine broadcast items
# (2.02 earnings releases, 7.01 Reg FD, 9.01 exhibits) that fire on most filings
# and would otherwise turn every earnings season into fake cross-source agreement.
SIGNAL_ITEM_CODES = frozenset({
    "1.01",  # entry into a material definitive agreement
    "1.02",  # termination of a material definitive agreement
    "1.03",  # bankruptcy or receivership
    "2.01",  # completion of acquisition or disposition of assets
    "2.05",  # costs associated with exit or disposal activities
    "2.06",  # material impairments
    "3.01",  # notice of delisting / failure to satisfy listing rule
    "4.01",  # changes in registrant's certifying accountant
    "4.02",  # non-reliance on previously issued financials (restatement)
    "5.01",  # changes in control of registrant
    "5.02",  # departure/election of directors or certain officers
})


class MaterialEvent(Base):
    """One 8-K filing, reduced to its material facts."""

    __tablename__ = "material_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    form_type: Mapped[str] = mapped_column(String(10), nullable=False, default="8-K")
    # Item codes present in the filing, e.g. ["5.02", "9.01"].
    items: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # Raw ITEM INFORMATION description lines from the SGML header (human context).
    item_descriptions: Mapped[str | None] = mapped_column(Text, nullable=True)
    filed_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    sec_url: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_material_event_ticker_date", "ticker", "filed_date"),
    )
