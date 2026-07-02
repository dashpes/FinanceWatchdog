"""Tests for the dashboard's read-only SQLite engine and WAL coexistence."""

import pytest
from sqlalchemy import text

from investment_monitor.storage import get_session, init_db
from investment_monitor.storage.robo_models import RoboRun
from investment_monitor.web import db as webdb


def _seed(db_path):
    init_db(db_path)
    with get_session() as s:
        s.add(RoboRun(run_id="r1", dry_run=True, status="completed"))


def test_read_session_reads(tmp_path):
    path = tmp_path / "p.db"
    _seed(path)
    webdb.init_read_only(path)
    with webdb.read_session() as s:
        runs = s.query(RoboRun).all()
        assert len(runs) == 1 and runs[0].run_id == "r1"


def test_read_session_cannot_write(tmp_path):
    path = tmp_path / "p.db"
    _seed(path)
    webdb.init_read_only(path)
    with webdb.read_session() as s, pytest.raises(Exception) as excinfo:
        s.execute(text("INSERT INTO robo_runs (run_id, dry_run, status) VALUES ('x', 1, 'y')"))
        s.commit()
    assert "readonly" in str(excinfo.value).lower()


def test_wal_mode_enabled_by_writer(tmp_path):
    path = tmp_path / "p.db"
    _seed(path)
    with get_session() as s:
        mode = s.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_reader_sees_data_while_writer_holds_transaction(tmp_path):
    path = tmp_path / "p.db"
    _seed(path)
    webdb.init_read_only(path)
    # Writer opens a transaction and inserts without committing...
    with get_session() as writer:
        writer.add(RoboRun(run_id="r2", dry_run=True, status="running"))
        writer.flush()
        # ...the WAL reader still reads the last committed state, unblocked.
        with webdb.read_session() as reader:
            runs = reader.query(RoboRun).all()
            assert [r.run_id for r in runs] == ["r1"]
