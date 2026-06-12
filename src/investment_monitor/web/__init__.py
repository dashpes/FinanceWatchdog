"""Local web dashboard for the investment monitor.

A FastAPI app served on localhost - no code signing, no Apple/Microsoft
developer account, identical on macOS/Windows/Linux. Launch with:

    investment-monitor --serve
"""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
