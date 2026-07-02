"""Settings: the tunables catalog rendered as a form, with a guarded write path.

Reads and writes go through ``robo.tunables`` — the same validated,
comment-preserving YAML path the CLI uses — never a hand-rolled parser. Two
extra guards on top of the CLI:

  * safety keys (``dry_run``, ``mode``) need an explicit ``confirm: true``;
  * ``dry_run: false`` (arming live trading) is refused outright — going live
    remains a console act, mirroring the control file's one-way invariant.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from investment_monitor.config import Settings
from investment_monitor.robo import tunables
from investment_monitor.robo.config import ConfigError, RoboConfig

from ..deps import get_app_settings, require_token

router = APIRouter(tags=["settings"])

# Mirrors robo/cli.py's _SAFETY_KEYS: changes that affect real-money behavior.
SAFETY_KEYS = {"dry_run", "mode"}

_write_lock = threading.Lock()


def _config_path(settings: Settings) -> Path:
    return Path(settings.config_dir) / "robo.yaml"


@router.get("/settings")
def list_settings(settings: Settings = Depends(get_app_settings)) -> dict:
    try:
        cfg = RoboConfig.from_yaml(_config_path(settings))
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"robo.yaml is invalid: {exc}") from exc
    items = []
    for t in tunables.catalog():
        item = t.as_dict()
        try:
            item["current"] = tunables.get_value(cfg, t.key)
        except AttributeError:
            item["current"] = None
        item["safety"] = t.key in SAFETY_KEYS
        items.append(item)
    return {"settings": items, "note": "changes take effect on the next run"}


class SettingUpdate(BaseModel):
    value: str | int | float | bool
    confirm: bool = False


@router.put("/settings/{key:path}", dependencies=[Depends(require_token)])
def update_setting(
    key: str,
    body: SettingUpdate,
    settings: Settings = Depends(get_app_settings),
) -> dict:
    raw = str(body.value).strip()
    if key == "dry_run" and raw.lower() in ("false", "0", "no", "off"):
        raise HTTPException(
            status_code=403,
            detail="arming live trading from the dashboard is not allowed — "
            "use the console (robo.yaml + ROBO_FORCE_DRY_RUN)",
        )
    if key in SAFETY_KEYS and not body.confirm:
        raise HTTPException(
            status_code=428,
            detail=f"'{key}' affects real-money behavior — resend with confirm: true",
        )
    with _write_lock:
        try:
            value = tunables.set_value(_config_path(settings), key, raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("dashboard set {k} = {v}", k=key, v=value)
    return {"key": key, "value": value, "note": "takes effect on the next run"}
