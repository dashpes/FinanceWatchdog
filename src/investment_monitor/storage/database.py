"""Database engine and session management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine = None
_SessionLocal = None


def attach_sqlite_pragmas(engine) -> None:
    """WAL + busy_timeout on every connection of ``engine``.

    WAL lets readers (the dashboard) proceed while a trade/research run writes;
    busy_timeout retries briefly instead of raising ``database is locked``. The
    journal mode is persistent per DB file, so the first writer flips it once.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial glue
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


def _reconcile_schema(engine) -> None:
    """Additive, fail-open migration: ``ADD COLUMN`` for model columns missing from an
    already-existing table.

    ``Base.metadata.create_all`` only CREATEs new tables — it never ALTERs an existing
    one, so a new *column* on an old table silently never lands (this project has no
    Alembic). This reconciles only ADDITIVE, nullable, server-default-less columns
    (the safe case in SQLite) and never raises, so it can't block startup.
    """
    try:
        insp = inspect(engine)
        existing = set(insp.get_table_names())
        pending = []  # (table_name, column)
        for table in Base.metadata.sorted_tables:
            if table.name not in existing:
                continue  # create_all just made it; columns already match the model
            db_cols = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in db_cols and col.nullable and col.server_default is None:
                    pending.append((table.name, col))
        if not pending:
            return
        with engine.begin() as conn:
            for table_name, col in pending:
                coltype = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {coltype}'))
                logger.info("schema reconcile: added {t}.{c} ({ct})",
                            t=table_name, c=col.name, ct=coltype)
    except Exception as exc:  # noqa: BLE001 - a reconcile error must never block startup
        logger.warning("schema reconcile failed (continuing): {e}", e=exc)


def init_db(db_path: str | Path = "data/portfolio.db") -> None:
    """Initialize database engine, create new tables, and reconcile additive columns."""
    global _engine, _SessionLocal

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    attach_sqlite_pragmas(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

    Base.metadata.create_all(bind=_engine)
    _reconcile_schema(_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
