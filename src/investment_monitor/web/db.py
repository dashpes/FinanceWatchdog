"""Read-only database access for the dashboard.

The dashboard must never write the store and never run migrations, so it does
NOT share ``storage.database.init_db`` (which CREATEs tables and reconciles
columns on every call). It builds its own engine whose every connection is
pinned read-only with ``PRAGMA query_only=ON`` — any write attempt raises
``SQLITE_READONLY`` at the driver level.

Why ``query_only`` instead of a ``file:...?mode=ro`` URI: with WAL journaling a
reader must still be able to create/write the ``-shm`` sidecar; a strictly
read-only file handle can fail to open a WAL database that has no live writer.
``query_only`` gives the same no-writes guarantee without that failure mode.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal = None
_engine_path: Path | None = None


def init_read_only(db_path: str | Path) -> None:
    """Build (or rebuild) the read-only engine for ``db_path``. Never creates tables."""
    global _engine, _SessionLocal, _engine_path

    db_path = Path(db_path)
    if _engine is not None and _engine_path == db_path:
        return
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _pin_read_only(dbapi_conn, _record):  # pragma: no cover - trivial glue
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA query_only=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    _engine = engine
    _engine_path = db_path
    _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def read_session() -> Generator[Session, None, None]:
    """A session that can only read. No commit on exit — there is nothing to commit."""
    if _SessionLocal is None:
        raise RuntimeError("read-only DB not initialized. Call init_read_only() first.")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
