"""FastAPI application for the local dashboard.

Exposes a small JSON API consumed by a single-page dashboard:
- GET  /                 -> the dashboard page
- GET  /api/status       -> hardware/model/provider/Ollama + alert counts
- GET  /api/portfolio    -> holdings + watchlist (from config)
- PUT  /api/portfolio    -> validate and save the portfolio
- GET  /api/alerts       -> recent alerts from the database
- POST /api/run          -> trigger a monitor run (regular/digest/weekly)
- GET  /api/run          -> status of the last/current run

The monitor run executes in a background thread (a fresh thread gets its own
event loop, so run_monitor_sync's asyncio.run works safely off the request path).
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..analysis.hardware import recommend_models, total_ram_gb
from ..analysis.ollama_client import model_matches
from ..config import Settings, get_settings
from ..diagnostics import _probe_ollama
from ..main import _load_portfolio, run_monitor_sync
from ..models import AlertsConfig, Portfolio
from ..setup_wizard import _pull_model

if TYPE_CHECKING:
    from fastapi import FastAPI

_STATIC_DIR = Path(__file__).parent / "static"
VALID_RUN_TYPES = ("regular", "digest", "weekly")
VALID_PROVIDERS = ("auto", "ollama", "anthropic")
_ENV_PATH = Path(".env")

# Maps Settings field -> .env variable name for editable values.
_LLM_ENV = {
    "ollama_host": "OLLAMA_HOST",
    "ollama_model": "OLLAMA_MODEL",
    "ollama_synthesis_model": "OLLAMA_SYNTHESIS_MODEL",
    "llm_provider": "LLM_PROVIDER",
}
# Secret/token settings: only written when a non-empty value is provided.
_SECRET_ENV = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "discord_webhook_url": "DISCORD_WEBHOOK_URL",
    "discord_daily_webhook_url": "DISCORD_DAILY_WEBHOOK_URL",
    "discord_weekly_webhook_url": "DISCORD_WEEKLY_WEBHOOK_URL",
    "slack_webhook_url": "SLACK_WEBHOOK_URL",
    "sendgrid_api_key": "SENDGRID_API_KEY",
}


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Update/append KEY=VALUE pairs in a .env file, preserving other lines.

    Existing keys are updated in place; unknown keys are appended; comments and
    unrelated lines are left untouched.

    Args:
        path: Path to the .env file (created if missing).
        updates: Mapping of ENV_VAR -> value to write.
    """
    if not updates:
        return
    lines = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    path.write_text("\n".join(out) + "\n")


def _build_status(settings: Settings) -> dict[str, Any]:
    """Assemble the status payload (hardware, models, provider, Ollama, counts)."""
    ram = total_ram_gb()
    rec = recommend_models(ram)
    fast = settings.resolved_ollama_model()
    synth = settings.resolved_synthesis_model()
    reachable, installed, err = _probe_ollama(settings.ollama_host)

    alerts_7d: int | None = None
    try:
        from ..storage import get_recent_alerts, get_session, init_db

        init_db(settings.db_path)
        with get_session() as session:
            alerts_7d = len(get_recent_alerts(session, hours=168))
    except Exception:
        alerts_7d = None

    return {
        "ram_gb": round(ram, 1) if ram is not None else None,
        "tier": rec.tier,
        "fast_model": fast,
        "synthesis_model": synth,
        "provider": "anthropic" if settings.prefer_anthropic_synthesis() else "ollama",
        "ollama_host": settings.ollama_host,
        "ollama_reachable": reachable,
        "ollama_error": err,
        "installed_models": installed,
        "fast_model_ready": reachable and model_matches(installed, fast),
        "synthesis_model_ready": reachable and model_matches(installed, synth),
        "alerts_last_7d": alerts_7d,
    }


def _serialize_alert(alert: Any) -> dict[str, Any]:
    return {
        "id": alert.id,
        "alert_type": alert.alert_type,
        "ticker": alert.ticker,
        "message": alert.message,
        "priority": alert.priority,
        "sent_at": alert.sent_at.isoformat() if alert.sent_at else None,
        "channel": alert.channel,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app.

    Args:
        settings: Settings to use (defaults to loaded application settings).

    Returns:
        A configured FastAPI application.
    """
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    settings = settings or get_settings()
    app = FastAPI(title="Investment Monitor", version="0.1.0")

    # Background run state (single run at a time).
    app.state.run = {
        "status": "idle",  # idle | running | done | error
        "run_type": None,
        "started_at": None,
        "finished_at": None,
        "summary": None,
        "error": None,
    }
    app.state.run_lock = threading.Lock()

    # Background model-pull state (single pull at a time).
    app.state.pull = {"status": "idle", "model": None, "error": None}
    app.state.pull_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.exists():
            return "<h1>Investment Monitor</h1><p>Dashboard assets missing.</p>"
        return index_path.read_text()

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return _build_status(settings)

    @app.get("/api/portfolio")
    def get_portfolio() -> dict[str, Any]:
        portfolio = _load_portfolio(settings.config_dir)
        return portfolio.to_dict()

    @app.put("/api/portfolio")
    def put_portfolio(payload: dict = Body(...)) -> dict[str, Any]:
        # Validate via the Pydantic model before writing anything to disk.
        try:
            portfolio = Portfolio(**payload)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid portfolio: {e}") from e
        settings.config_dir.mkdir(parents=True, exist_ok=True)
        portfolio.to_yaml(settings.config_dir / "portfolio.yaml")
        return portfolio.to_dict()

    @app.get("/api/alerts")
    def get_alerts(hours: int = 168) -> list[dict[str, Any]]:
        try:
            from ..storage import get_recent_alerts, get_session, init_db

            init_db(settings.db_path)
            with get_session() as session:
                return [_serialize_alert(a) for a in get_recent_alerts(session, hours=hours)]
        except Exception:
            return []

    @app.get("/api/run")
    def run_status() -> dict[str, Any]:
        with app.state.run_lock:
            return dict(app.state.run)

    @app.post("/api/run")
    def trigger_run(payload: dict = Body(default={})) -> JSONResponse:
        run_type = (payload or {}).get("type", "regular")
        if run_type not in VALID_RUN_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid run type '{run_type}'. Use one of {VALID_RUN_TYPES}.",
            )

        with app.state.run_lock:
            if app.state.run["status"] == "running":
                return JSONResponse(
                    status_code=409,
                    content={"status": "running", "run_type": app.state.run["run_type"]},
                )
            app.state.run.update(
                status="running",
                run_type=run_type,
                started_at=datetime.now().isoformat(),
                finished_at=None,
                summary=None,
                error=None,
            )

        def _job() -> None:
            try:
                summary = run_monitor_sync(
                    config_path=settings.config_dir, run_type=run_type
                )
                with app.state.run_lock:
                    app.state.run.update(
                        status="done",
                        finished_at=datetime.now().isoformat(),
                        summary=str(summary),
                    )
            except Exception as e:  # noqa: BLE001 - surface any run failure to the UI
                with app.state.run_lock:
                    app.state.run.update(
                        status="error",
                        finished_at=datetime.now().isoformat(),
                        error=str(e),
                    )

        threading.Thread(target=_job, daemon=True).start()
        return JSONResponse(status_code=202, content={"status": "started", "run_type": run_type})

    # ---- Settings ---------------------------------------------------------
    @app.get("/api/settings")
    def get_settings_api() -> dict[str, Any]:
        alerts_path = settings.config_dir / "alerts.yaml"
        alerts = AlertsConfig.from_yaml(alerts_path) if alerts_path.exists() else AlertsConfig()
        return {
            "llm": {
                "ollama_host": settings.ollama_host,
                "ollama_model": settings.ollama_model,
                "ollama_synthesis_model": settings.ollama_synthesis_model,
                "llm_provider": settings.llm_provider,
                "anthropic_api_key_set": bool(settings.anthropic_api_key),
            },
            "notifications": {
                "discord_webhook_url_set": bool(settings.discord_webhook_url),
                "discord_daily_webhook_url_set": bool(settings.discord_daily_webhook_url),
                "discord_weekly_webhook_url_set": bool(settings.discord_weekly_webhook_url),
                "slack_webhook_url_set": bool(settings.slack_webhook_url),
                "sendgrid_api_key_set": bool(settings.sendgrid_api_key),
            },
            "alerts": alerts.model_dump(),
        }

    @app.put("/api/settings")
    def put_settings_api(payload: dict = Body(...)) -> dict[str, Any]:
        env_updates: dict[str, str] = {}

        # LLM (plain values: always applied when present)
        llm = payload.get("llm") or {}
        provider = llm.get("llm_provider")
        if provider is not None and provider not in VALID_PROVIDERS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid llm_provider '{provider}'. Use one of {VALID_PROVIDERS}.",
            )
        for field, env in _LLM_ENV.items():
            if field in llm and llm[field] is not None:
                val = str(llm[field]).strip()
                setattr(settings, field, val)
                env_updates[env] = val

        # Secrets/tokens (LLM + notifications): only updated when non-empty.
        secrets = {**llm, **(payload.get("notifications") or {})}
        for field, env in _SECRET_ENV.items():
            val = secrets.get(field)
            if val:  # non-empty -> update; blank/missing -> keep existing
                setattr(settings, field, val)
                env_updates[env] = val

        # Alert thresholds -> alerts.yaml
        if "alerts" in payload and payload["alerts"] is not None:
            try:
                alerts = AlertsConfig(**payload["alerts"])
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid alerts config: {e}") from e
            settings.config_dir.mkdir(parents=True, exist_ok=True)
            alerts.to_yaml(settings.config_dir / "alerts.yaml")

        update_env_file(_ENV_PATH, env_updates)
        return get_settings_api()

    # ---- Model management -------------------------------------------------
    @app.get("/api/models")
    def get_models() -> dict[str, Any]:
        reachable, installed, err = _probe_ollama(settings.ollama_host)
        with app.state.pull_lock:
            pull_state = dict(app.state.pull)
        return {
            "reachable": reachable,
            "error": err,
            "installed": installed,
            "fast_model": settings.resolved_ollama_model(),
            "synthesis_model": settings.resolved_synthesis_model(),
            "pull": pull_state,
        }

    @app.post("/api/models/pull")
    def pull_model(payload: dict = Body(...)) -> JSONResponse:
        model = (payload or {}).get("model", "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="A 'model' tag is required.")

        with app.state.pull_lock:
            if app.state.pull["status"] == "pulling":
                return JSONResponse(
                    status_code=409,
                    content={"status": "pulling", "model": app.state.pull["model"]},
                )
            app.state.pull.update(status="pulling", model=model, error=None)

        def _job() -> None:
            ok = _pull_model(model)
            with app.state.pull_lock:
                if ok:
                    app.state.pull.update(status="done", error=None)
                else:
                    app.state.pull.update(
                        status="error",
                        error=f"Pull failed. Is Ollama installed/running? Try: ollama pull {model}",
                    )

        threading.Thread(target=_job, daemon=True).start()
        return JSONResponse(status_code=202, content={"status": "started", "model": model})

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, settings: Settings | None = None) -> None:
    """Run the dashboard with uvicorn (blocking).

    Args:
        host: Interface to bind (default localhost only).
        port: Port to listen on.
        settings: Settings to use.
    """
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - exercised via the CLI message
        raise RuntimeError(
            "The web dashboard requires extra packages. Install them with:\n"
            "    pip install -e '.[dashboard]'"
        ) from e

    app = create_app(settings)
    print(f"Investment Monitor dashboard -> http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="info")
