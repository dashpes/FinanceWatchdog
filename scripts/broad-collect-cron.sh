#!/bin/bash
# Wrapper for the launchd-scheduled broad (universe-independent) data collection.
#
# Pulls the market-wide event stream (SEC Form 4 insider transactions today; more
# sources later) into the DB, decoupled from the portfolio universe. Unlike the robo
# wrapper this needs NO Ollama (pure HTTP + SQLite), so it can run alongside the robo
# jobs; its own lock just prevents overlapping broad-collect runs.
#
# Usage:  broad-collect-cron.sh [--days-back N]
set -uo pipefail

# Project root: explicit FW_HOME wins, else derive from this script's dir (…/scripts/..).
PROJ="${FW_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ" || { echo "broad-collect-cron: cannot cd to $PROJ" >&2; exit 1; }
mkdir -p "$PROJ/logs"
export PYTHONPATH="$PROJ/src"
LOG="$PROJ/logs/broad-collect.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] broad-collect: $*" >> "$LOG"; }

# --- single-run lock (atomic mkdir; clear a stale lock from a crashed run) ------
LOCK="$PROJ/logs/broad-collect.lock"
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -mmin +90 2>/dev/null)" ]; then
  log "clearing stale lock (>90m old)"; rmdir "$LOCK" 2>/dev/null || true
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  log "another broad-collect run is active; skipping"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

log "investment-monitor --type collect-broad $*"
"$PROJ/.venv/bin/investment-monitor" --type collect-broad "$@" >> "$LOG" 2>&1
status=$?
log "exit $status"
exit $status
