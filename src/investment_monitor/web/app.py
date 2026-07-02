"""The dashboard application factory and console entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from investment_monitor.config import Settings, get_settings

from . import db as webdb
from .account_cache import AccountCache

_HERE = Path(__file__).parent

# Page routes -> (template, nav title). Order defines the nav.
PAGES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "Overview"),
    "/theses": ("theses.html", "Theses"),
    "/trades": ("trades.html", "Ledger"),
    "/charts": ("charts.html", "Charts"),
    "/signals": ("signals.html", "Signals"),
    "/learning": ("learning.html", "The Study"),
    "/system": ("system.html", "System"),
    "/settings": ("settings.html", "Settings"),
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Archie — Personal Private Equity", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.account_cache = AccountCache(settings)
    webdb.init_read_only(settings.db_path)

    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")

    from .routers import charts, controls, learning, overview, settings_api, signals, system, theses, trades

    for router in (
        overview.router,
        theses.router,
        trades.router,
        signals.router,
        learning.router,
        charts.router,
        system.router,
        settings_api.router,
        controls.router,
    ):
        app.include_router(router, prefix="/api")

    nav = [{"href": href, "title": title} for href, (_, title) in PAGES.items()]

    def _page(template: str, active: str):
        async def render(request: Request) -> HTMLResponse:
            return templates.TemplateResponse(
                request,
                template,
                {"nav": nav, "active": active},
            )

        return render

    for href, (template, title) in PAGES.items():
        app.get(href, response_class=HTMLResponse, include_in_schema=False)(
            _page(template, href)
        )

    # Thesis detail page (same shell, symbol read client-side from the path).
    app.get("/theses/{symbol}", response_class=HTMLResponse, include_in_schema=False)(
        _page("thesis_detail.html", "/theses")
    )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    return app


def main() -> None:
    """Console entry point: `investment-dashboard`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
        workers=1,
    )


if __name__ == "__main__":
    main()
