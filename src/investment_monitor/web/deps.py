"""FastAPI dependencies: settings, read-only DB sessions, and mutation auth."""

from __future__ import annotations

import secrets
from collections.abc import Generator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from investment_monitor.config import Settings

from . import db as webdb


def get_app_settings(request: Request) -> Settings:
    """The Settings instance the app was built with (overridable in tests)."""
    return request.app.state.settings


def get_read_session() -> Generator[Session, None, None]:
    """A read-only DB session per request (PRAGMA query_only)."""
    with webdb.read_session() as session:
        yield session


def require_token(request: Request, settings: Settings = Depends(get_app_settings)) -> None:
    """Gate for every mutating endpoint: Bearer token from DASHBOARD_TOKEN.

    A blank configured token disables mutations entirely (403) rather than
    allowing them unauthenticated — the safe default for a fresh install.
    """
    configured = (settings.dashboard_token or "").strip()
    if not configured:
        raise HTTPException(
            status_code=403,
            detail="dashboard is read-only until DASHBOARD_TOKEN is set in .env",
        )
    supplied = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied:
        supplied = request.headers.get("x-dashboard-token", "").strip()
    if not supplied or not secrets.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="invalid or missing dashboard token")
