"""Controls: pause/resume, the web kill switch, and blocklist management.

All writes go through ``robo.control`` / ``robo.blocklist`` (the same modules
the CLI and the gate use), never the DB. The one-way safety invariant lives in
the control module itself: nothing here can arm live trading.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from investment_monitor.config import Settings
from investment_monitor.robo import blocklist, control

from ..deps import get_app_settings, require_token

router = APIRouter(tags=["controls"])


def _state(settings: Settings) -> dict:
    ctl = control.load_control(settings.db_path)
    return {
        "trading_paused": ctl.trading_paused,
        "force_dry_run": ctl.force_dry_run,
        "reason": ctl.reason,
        "updated_at": ctl.updated_at,
        "updated_by": ctl.updated_by,
        "env_force_dry_run": settings.robo_force_dry_run,
        "note": "changes take effect on the next run",
    }


@router.get("/control")
def get_control(settings: Settings = Depends(get_app_settings)) -> dict:
    return _state(settings)


class ControlBody(BaseModel):
    reason: str = ""


@router.post("/control/pause", dependencies=[Depends(require_token)])
def pause(body: ControlBody, settings: Settings = Depends(get_app_settings)) -> dict:
    control.set_paused(settings.db_path, True, reason=body.reason, updated_by="dashboard")
    return _state(settings)


@router.post("/control/resume", dependencies=[Depends(require_token)])
def resume(settings: Settings = Depends(get_app_settings)) -> dict:
    control.set_paused(settings.db_path, False, updated_by="dashboard")
    return _state(settings)


@router.post("/control/kill", dependencies=[Depends(require_token)])
def kill(body: ControlBody, settings: Settings = Depends(get_app_settings)) -> dict:
    """Force paper mode from the web layer (never the reverse)."""
    control.set_force_dry_run(settings.db_path, True, reason=body.reason, updated_by="dashboard")
    return _state(settings)


@router.post("/control/unkill", dependencies=[Depends(require_token)])
def unkill(settings: Settings = Depends(get_app_settings)) -> dict:
    """Clear the web-layer force. .env and robo.yaml still decide live vs paper."""
    control.set_force_dry_run(settings.db_path, False, updated_by="dashboard")
    return _state(settings)


@router.get("/blocklist")
def get_blocklist(settings: Settings = Depends(get_app_settings)) -> dict:
    static: list[str] = []
    try:
        from investment_monitor.robo.config import RoboConfig

        cfg = RoboConfig.from_yaml(settings.config_dir / "robo.yaml")
        static = sorted(getattr(cfg, "blocklist", []) or [])
    except Exception:  # noqa: BLE001 - static list is display-only
        pass
    return {
        "learned": blocklist.list_learned(str(settings.db_path)),
        "static": static,
    }


class BlocklistBody(BaseModel):
    symbol: str
    reason: str = ""


@router.post("/blocklist", dependencies=[Depends(require_token)])
def add_symbol(body: BlocklistBody, settings: Settings = Depends(get_app_settings)) -> dict:
    symbol = body.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol is required")
    blocklist.add_learned(str(settings.db_path), symbol, reason=body.reason or "added via dashboard")
    return get_blocklist(settings)


@router.delete("/blocklist/{symbol}", dependencies=[Depends(require_token)])
def remove_symbol(symbol: str, settings: Settings = Depends(get_app_settings)) -> dict:
    if not blocklist.remove_learned(str(settings.db_path), symbol):
        raise HTTPException(status_code=404, detail=f"{symbol.upper()} is not on the learned blocklist")
    return get_blocklist(settings)
