#!/bin/bash
# Wrapper for scheduled robo runs (launchd on macOS, systemd on Linux/Raspberry Pi).
#
# Schedulers start jobs with a minimal environment, so this script establishes
# everything investment-robo needs: the project working directory (so .env and
# config/ are found), PYTHONPATH, and the venv interpreter. Adds two resilience
# guardrails: a single-run lock (overlapping runs would contend for RAM/Ollama),
# and an Ollama health check that restarts a wedged server before running.
#
# Portable: the project root is derived from this script's location (override with
# FW_HOME), and the Ollama restart uses whatever is present (brew on macOS,
# systemctl on Linux) — nothing is hardcoded to one machine.
#
# Usage:  robo-cron.sh thesis-run --discover --no-trade
set -uo pipefail

# Project root: explicit FW_HOME wins, else derive from this script's dir (…/scripts/..).
PROJ="${FW_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJ" || { echo "robo-cron: cannot cd to $PROJ" >&2; exit 1; }
mkdir -p "$PROJ/logs"
export PYTHONPATH="$PROJ/src"
LOG="$PROJ/logs/robo-cron.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] robo-cron: $*" >> "$LOG"; }

# Restart the local Ollama server using whatever supervisor is available. On the Pi
# the robo user is granted a scoped sudoers entry for `systemctl restart ollama`
# (see scripts/install.sh); on macOS Homebrew manages it.
restart_ollama() {
  if command -v brew >/dev/null 2>&1; then
    brew services restart ollama >> "$LOG" 2>&1 || true
  elif command -v systemctl >/dev/null 2>&1; then
    sudo -n systemctl restart ollama >> "$LOG" 2>&1 \
      || systemctl --user restart ollama >> "$LOG" 2>&1 || true
  else
    log "no known way to restart Ollama; relying on its own supervisor"
  fi
}

# --- single-run lock (atomic mkdir) ---------------------------------------------
# Staleness is decided by the holder's LIVENESS, not age. A qwen2.5:14b synthesis on a
# Pi can legitimately hold the lock for well over an hour, so clearing on age alone would
# let the 07:00 trade run concurrently with a still-running overnight research cycle — and
# that cycle's EXIT trap would then delete the trade's lock. We only reclaim a lock whose
# recorded PID is dead (age is just a backstop for a lock left with no PID recorded).
#
# TRADE runs WAIT for the lock instead of skipping: an overnight research cycle that
# runs long (Ollama discovery can take an hour+) used to silently swallow the 07:00
# trade slot — no run, no email, nothing. Now the trade run polls until the holder
# finishes (bounded by FW_LOCK_WAIT, default 75 min) and emails if it gives up.
# Research/maintenance invocations keep the old skip-immediately behavior: the loop
# reruns them anyway, so waiting would just queue redundant cycles.
LOCK="$PROJ/logs/robo-cron.lock"
WAIT_SECS=0
case " $* " in
  *" thesis-run "*|*" run "*)
    case " $* " in
      *" --no-trade "*) ;;                       # research-only: skip, don't queue
      *) WAIT_SECS="${FW_LOCK_WAIT:-4500}" ;;    # a real trade slot: wait for it
    esac
    ;;
esac

notify_skip() {
  PYTHONPATH="$PROJ/src" "$PROJ/.venv/bin/python" - "$1" <<'PYEOF' >> "$LOG" 2>&1 || true
import sys
from investment_monitor.config import get_settings
from investment_monitor.robo.notify import notify_error
notify_error(get_settings(), message=sys.argv[1])
PYEOF
}

deadline=$(( $(date +%s) + WAIT_SECS ))
waited=0
while :; do
  if [ -d "$LOCK" ]; then
    holder="$(cat "$LOCK/pid" 2>/dev/null || true)"
    if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
      if [ "$(date +%s)" -lt "$deadline" ] && [ "$WAIT_SECS" -gt 0 ]; then
        [ "$waited" = 0 ] && log "run (pid $holder) active; waiting up to ${WAIT_SECS}s for the lock ($*)"
        waited=1; sleep 30; continue
      fi
      if [ "$WAIT_SECS" -gt 0 ]; then
        log "gave up waiting ${WAIT_SECS}s for lock held by pid $holder; skipping ($*)"
        notify_skip "Scheduled run '$*' was SKIPPED: another run (pid $holder) held the lock for over ${WAIT_SECS}s."
      else
        log "another run (pid $holder) is active; skipping ($*)"
      fi
      exit 0
    fi
    if [ -z "$holder" ] && [ -z "$(find "$LOCK" -mmin +180 2>/dev/null)" ]; then
      log "lock present without a live holder but recent; skipping ($*)"; exit 0
    fi
    log "clearing stale lock (holder pid ${holder:-none} not alive)"; rm -rf "$LOCK" 2>/dev/null || true
  fi
  if mkdir "$LOCK" 2>/dev/null; then
    break
  fi
  # Lost a race with another starter; loop re-evaluates (waits or skips as above).
  sleep 1
done
[ "$waited" = 1 ] && log "lock acquired after waiting ($*)"
echo "$$" > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null || true' EXIT

# --- Ollama health check (supervisor restarts crashes; this catches a wedged one) -
if ! curl -sf --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  log "Ollama not responding; restarting"
  restart_ollama
  sleep 6
fi

log "investment-robo $*"
"$PROJ/.venv/bin/investment-robo" "$@" >> "$LOG" 2>&1
status=$?
log "exit $status"
exit $status
