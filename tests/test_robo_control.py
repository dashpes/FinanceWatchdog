"""Tests for the operator control file (pause / web kill switch)."""

import json
from types import SimpleNamespace

from investment_monitor.robo import control
from investment_monitor.robo.rebalance import _resolve_dry_run


def _db(tmp_path):
    return tmp_path / "portfolio.db"


# --- load/save fail-open -----------------------------------------------------------

def test_missing_file_is_safe_defaults(tmp_path):
    state = control.load_control(_db(tmp_path))
    assert state.trading_paused is False
    assert state.force_dry_run is False


def test_corrupt_file_is_safe_defaults(tmp_path):
    (tmp_path / "robo_control.json").write_text("{not json")
    state = control.load_control(_db(tmp_path))
    assert state.trading_paused is False
    assert state.force_dry_run is False


def test_set_paused_roundtrip(tmp_path):
    control.set_paused(_db(tmp_path), True, reason="holiday", updated_by="dashboard")
    state = control.load_control(_db(tmp_path))
    assert state.trading_paused is True
    assert state.reason == "holiday"
    assert state.updated_by == "dashboard"
    assert state.updated_at  # stamped

    control.set_paused(_db(tmp_path), False)
    assert control.load_control(_db(tmp_path)).trading_paused is False


def test_flags_are_independent(tmp_path):
    control.set_paused(_db(tmp_path), True)
    control.set_force_dry_run(_db(tmp_path), True)
    control.set_paused(_db(tmp_path), False)
    state = control.load_control(_db(tmp_path))
    assert state.force_dry_run is True  # clearing the pause left the dry-run force alone
    assert state.trading_paused is False


def test_unknown_keys_survive_writes(tmp_path):
    path = tmp_path / "robo_control.json"
    path.write_text(json.dumps({"trading_paused": False, "future_flag": 7}))
    control.set_paused(_db(tmp_path), True)
    assert json.loads(path.read_text())["future_flag"] == 7


# --- the one-way safety invariant ----------------------------------------------------

def _settings(tmp_path, force=False):
    return SimpleNamespace(robo_force_dry_run=force, db_path=str(_db(tmp_path)))


def _config(dry_run=True):
    return SimpleNamespace(dry_run=dry_run)


def test_control_force_dry_run_forces_paper(tmp_path):
    control.set_force_dry_run(_db(tmp_path), True)
    # Even a live config + explicit live override cannot escape the control force.
    assert _resolve_dry_run(_config(dry_run=False), _settings(tmp_path), None) is True
    assert _resolve_dry_run(_config(dry_run=False), _settings(tmp_path), False) is True


def test_clearing_control_never_arms_live(tmp_path):
    control.set_force_dry_run(_db(tmp_path), False)
    # With the control layer clear, the other layers still decide.
    assert _resolve_dry_run(_config(dry_run=True), _settings(tmp_path), None) is True
    assert _resolve_dry_run(_config(dry_run=False), _settings(tmp_path, force=True), None) is True
    # Live only when env allows AND config permits.
    assert _resolve_dry_run(_config(dry_run=False), _settings(tmp_path), None) is False


# --- pause short-circuits a rebalance run --------------------------------------------

def test_paused_run_persists_and_skips_broker(tmp_path):
    from investment_monitor.robo.rebalance import rebalance_run
    from investment_monitor.storage import get_session, init_db
    from investment_monitor.storage.robo_models import RoboRun

    control.set_paused(_db(tmp_path), True, reason="testing")

    class ExplodingBroker:
        dry_run = True

        def get_account_state(self):
            raise AssertionError("broker must not be touched while paused")

    settings = SimpleNamespace(
        robo_force_dry_run=True, db_path=str(_db(tmp_path)), log_dir=str(tmp_path / "logs")
    )
    config = SimpleNamespace(dry_run=True)
    result = rebalance_run(config, settings, broker=ExplodingBroker())
    assert result.status == "paused"
    assert "paused" in result.message
    assert "testing" in result.message

    init_db(settings.db_path)
    with get_session() as session:
        runs = session.query(RoboRun).all()
        assert len(runs) == 1
        assert runs[0].status == "paused"
