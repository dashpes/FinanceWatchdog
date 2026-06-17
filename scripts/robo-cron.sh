#!/bin/bash
# Wrapper for launchd-scheduled robo runs.
#
# launchd starts jobs with a minimal environment, so this script establishes
# everything investment-robo needs: the project working directory (so .env and
# config/ are found), PYTHONPATH, and the venv interpreter. Adds two resilience
# guardrails: a single-run lock (overlapping runs would contend for RAM/Ollama),
# and an Ollama health check that restarts a wedged server before running.
#
# Usage:  robo-cron.sh thesis-run --discover --no-trade
set -uo pipefail

PROJ="/Users/danielashpes/Developer/FinanceWatchdog"
BREW="/opt/homebrew/bin/brew"
cd "$PROJ" || { echo "robo-cron: cannot cd to $PROJ" >&2; exit 1; }
mkdir -p "$PROJ/logs"
export PYTHONPATH="$PROJ/src"
LOG="$PROJ/logs/robo-cron.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] robo-cron: $*" >> "$LOG"; }

# --- single-run lock (atomic mkdir; clear a stale lock from a crashed run) ------
LOCK="$PROJ/logs/robo-cron.lock"
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -mmin +45 2>/dev/null)" ]; then
  log "clearing stale lock (>45m old)"; rmdir "$LOCK" 2>/dev/null || true
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  log "another run is active; skipping ($*)"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# --- Ollama health check (KeepAlive restarts crashes; this catches a wedged one) -
if ! curl -sf --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
  log "Ollama not responding; restarting"
  "$BREW" services restart ollama >> "$LOG" 2>&1 || true
  sleep 6
fi

log "investment-robo $*"
"$PROJ/.venv/bin/investment-robo" "$@" >> "$LOG" 2>&1
status=$?
log "exit $status"
exit $status
