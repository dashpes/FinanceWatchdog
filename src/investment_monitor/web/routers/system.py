"""System health: runs, services, timers, DB size, control state."""

from __future__ import annotations

import os
import shutil
import subprocess

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from investment_monitor.config import Settings
from investment_monitor.robo import blocklist, control
from investment_monitor.storage.robo_models import RoboRun

from ..deps import get_app_settings, get_read_session
from ._serialize import run_dict

router = APIRouter(tags=["system"])

_UNITS = (
    "financewatchdog-research.service",
    "financewatchdog-trade.timer",
    "financewatchdog-summary.timer",
    "financewatchdog-prune.timer",
    "financewatchdog-dashboard.service",
)


def _systemctl(*args: str) -> str | None:
    """Best-effort systemctl read; None off-Linux or on any failure."""
    try:
        out = subprocess.run(
            ["systemctl", *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - macOS dev / missing systemd
        return None


def _services() -> list[dict] | None:
    states = []
    for unit in _UNITS:
        active = _systemctl("is-active", unit)
        if active is None:
            return None  # no systemd here — hide the section
        entry: dict = {"unit": unit, "active": active}
        if unit.endswith(".timer"):
            show = _systemctl("show", unit, "--property=NextElapseUSecRealtime")
            if show and "=" in show:
                value = show.split("=", 1)[1].strip()
                entry["next_elapse"] = value or None
        states.append(entry)
    return states


@router.get("/system")
def system(
    session: Session = Depends(get_read_session),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    runs = (
        session.query(RoboRun).order_by(RoboRun.started_at.desc()).limit(20).all()
    )
    problems = [r for r in runs if r.status in ("failed", "refused", "errored")]

    db_size = None
    try:
        db_size = os.stat(settings.db_path).st_size
    except OSError:
        pass
    disk_free = None
    try:
        disk_free = shutil.disk_usage(settings.data_dir).free
    except OSError:
        pass

    ctl = control.load_control(settings.db_path)
    return {
        "recent_runs": [run_dict(r) for r in runs],
        "problem_runs": [run_dict(r) for r in problems],
        "db_size_bytes": db_size,
        "disk_free_bytes": disk_free,
        "services": _services(),
        "control": {
            "trading_paused": ctl.trading_paused,
            "force_dry_run": ctl.force_dry_run,
            "reason": ctl.reason,
            "updated_at": ctl.updated_at,
            "updated_by": ctl.updated_by,
        },
        "env_force_dry_run": settings.robo_force_dry_run,
        "blocklist_learned": blocklist.list_learned(str(settings.db_path)),
    }
