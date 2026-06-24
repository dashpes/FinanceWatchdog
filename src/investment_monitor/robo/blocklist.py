"""Smart blocklist of symbols the robo must not BUY.

Motivation: a high-conviction pick that the broker refuses to buy (e.g. Public's
"only available for trading when closing an existing position" — an IPO/halt/
delisting/low-liquidity flag) otherwise wins the open position slot every single
run, fails preflight, and strands cash that buyable alternatives could use. The
blocklist makes the system *learn* such names and skip them thereafter.

Two sources, unioned at gate time:
  * static  — ``blocklist:`` in config/robo.yaml (operator-curated).
  * learned — symbols auto-recorded here the first time the broker refuses a BUY as
    un-buyable. Persisted as JSON next to the SQLite DB.

The blocklist restricts only new BUYs — SELLs are never blocked, so a blocklisted
name that is already held can always still be exited. All I/O is fail-open: a
blocklist error is logged and ignored, never raised, so it cannot break a run.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

# Substrings in a broker preflight/placement message that mean "cannot be bought now".
_UNBUYABLE_MARKERS = (
    "only available for trading when closing an existing position",
    "only available when closing",
    "not currently tradable",
    "not tradable",
    "delist",
    "halted",
)


def _path(db_path: str) -> Path:
    """Location of the learned-blocklist file: alongside the DB."""
    return Path(db_path).expanduser().parent / "robo_blocklist.json"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - never let blocklist I/O break a run
        logger.warning("blocklist read failed (ignored): {e}", e=exc)
        return {}


def load_learned(db_path: str) -> set[str]:
    """Return the set of auto-learned blocklisted symbols (uppercased)."""
    entries = _read(_path(db_path)).get("entries", {})
    return {str(s).upper() for s in entries}


def add_learned(db_path: str, symbol: str, reason: str = "") -> bool:
    """Persist ``symbol`` to the learned blocklist. Returns True if newly added."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return False
    path = _path(db_path)
    try:
        data = _read(path)
        entries = data.get("entries", {})
        if symbol in entries:
            return False
        entries[symbol] = {"reason": (reason or "")[:300]}
        data["entries"] = entries
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        logger.info("blocklisted {s} for buys (reason: {r})", s=symbol, r=(reason or "")[:120])
        return True
    except Exception as exc:  # noqa: BLE001 - fail-open
        logger.warning("blocklist add failed for {s} (ignored): {e}", s=symbol, e=exc)
        return False


def is_unbuyable_message(message: str | None) -> bool:
    """Heuristic: does a broker preflight/place message mean the asset can't be bought?"""
    if not message:
        return False
    low = message.lower()
    return any(marker in low for marker in _UNBUYABLE_MARKERS)
