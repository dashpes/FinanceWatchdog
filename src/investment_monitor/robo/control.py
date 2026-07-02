"""Operator control file for the robo advisor (pause / web kill switch).

A tiny JSON file next to the SQLite DB (like the learned blocklist) that the
dashboard and CLI write and every trade run reads at start. Two flags:

  * ``trading_paused`` — the next rebalance run records a ``paused`` RoboRun and
    returns without touching the broker. Research/discovery keeps running: pausing
    trading must never blind the advisor.
  * ``force_dry_run``  — forces paper mode, exactly like ``ROBO_FORCE_DRY_RUN``.

Safety invariant (one-way): this file can only ever make the system SAFER. It can
force paper mode or stop trading, but nothing in it can arm live trading — going
live remains exclusively a console act (.env ``ROBO_FORCE_DRY_RUN`` + robo.yaml
``dry_run``). Clearing ``force_dry_run`` merely stops *this* layer from forcing
paper; the other layers still decide.

All reads are fail-open to safe defaults (missing/corrupt file → not paused, not
forced); writes are atomic (temp file + ``os.replace``) so a crashed writer can
never leave a half-written control file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class ControlState:
    """The parsed control file. Defaults are the safe, do-nothing state."""

    trading_paused: bool = False
    force_dry_run: bool = False
    reason: str = ""
    updated_at: str = ""
    updated_by: str = ""

    extra: dict = field(default_factory=dict, repr=False, compare=False)


def _path(db_path: str | Path) -> Path:
    """Location of the control file: alongside the DB (like the blocklist)."""
    return Path(db_path).expanduser().parent / "robo_control.json"


def load_control(db_path: str | Path) -> ControlState:
    """Read the control state. Fail-open: any problem returns safe defaults."""
    path = _path(db_path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return ControlState()
    except Exception as exc:  # noqa: BLE001 - a bad control file must never break a run
        logger.warning("control file read failed (ignored): {e}", e=exc)
        return ControlState()
    if not isinstance(data, dict):
        return ControlState()
    known = {"trading_paused", "force_dry_run", "reason", "updated_at", "updated_by"}
    return ControlState(
        trading_paused=bool(data.get("trading_paused", False)),
        force_dry_run=bool(data.get("force_dry_run", False)),
        reason=str(data.get("reason", "") or ""),
        updated_at=str(data.get("updated_at", "") or ""),
        updated_by=str(data.get("updated_by", "") or ""),
        extra={k: v for k, v in data.items() if k not in known},
    )


def _write(db_path: str | Path, state: ControlState) -> bool:
    """Atomically persist ``state``. Returns False (logged) on any failure."""
    path = _path(db_path)
    try:
        payload = asdict(state)
        payload.pop("extra", None)
        payload.update(state.extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("control file write failed (ignored): {e}", e=exc)
        return False


def _update(
    db_path: str | Path,
    *,
    updated_by: str,
    reason: str | None = None,
    **flags: bool,
) -> ControlState:
    current = load_control(db_path)
    new = ControlState(
        trading_paused=flags.get("trading_paused", current.trading_paused),
        force_dry_run=flags.get("force_dry_run", current.force_dry_run),
        reason=current.reason if reason is None else reason[:300],
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        updated_by=updated_by,
        extra=current.extra,
    )
    _write(db_path, new)
    return new


def set_paused(
    db_path: str | Path, paused: bool, *, reason: str = "", updated_by: str = "cli"
) -> ControlState:
    """Pause/resume trading. Research and data collection are unaffected."""
    state = _update(
        db_path, trading_paused=paused, reason=reason, updated_by=updated_by
    )
    logger.info(
        "trading {v} via control file ({by})",
        v="PAUSED" if paused else "resumed",
        by=updated_by,
    )
    return state


def set_force_dry_run(
    db_path: str | Path, forced: bool, *, reason: str = "", updated_by: str = "cli"
) -> ControlState:
    """Set/clear this layer's paper-mode force.

    Clearing it never arms live trading by itself — .env and robo.yaml still rule.
    """
    state = _update(
        db_path, force_dry_run=forced, reason=reason, updated_by=updated_by
    )
    logger.info(
        "control-file dry-run force {v} ({by})",
        v="SET" if forced else "cleared",
        by=updated_by,
    )
    return state
