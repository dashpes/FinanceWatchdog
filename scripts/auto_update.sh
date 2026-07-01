#!/usr/bin/env bash
# Auto-update the FinanceWatchdog robo advisor from the latest STABLE git tag.
#
# Usage: auto_update.sh [--restart]
#   --restart  After updating, restart the robo systemd units (needs the scoped
#              sudoers entry that scripts/install.sh grants the service user).
#
# Design: tracks tagged releases (v*), NOT raw origin/main, so an untested commit can
# never auto-deploy to a box that trades real money. Reinstalls from requirements.lock
# when present (reproducible on ARM). The whole body is wrapped in a `{ }` group so
# bash parses it fully before executing — a mid-run `git checkout` can't corrupt the
# script it is running from.
{
set -euo pipefail

RESTART=0
[ "${1:-}" = "--restart" ] && RESTART=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${FW_HOME:-$(dirname "$SCRIPT_DIR")}"
# Absolute path so `sudo -n <path> ...` matches the scoped sudoers entry install.sh
# writes (both resolve systemctl the same way).
SUDO_SYSTEMCTL="$(command -v systemctl || echo /usr/bin/systemctl)"
LOG_FILE="$PROJECT_DIR/logs/auto_update.log"
mkdir -p "$(dirname "$LOG_FILE")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

cd "$PROJECT_DIR"
log "Auto-update check (tagged-release channel)…"

if ! git fetch --tags --prune origin 2>&1 | tee -a "$LOG_FILE"; then
  # No repo access (e.g. a private repo the service user can't reach) is not fatal — the
  # box keeps running its current code; the operator updates by re-copying + reinstalling.
  log "WARN: git fetch failed (no repo access?); skipping this update"; exit 0
fi

# Target = newest STRICT-semver tag (pre-releases like v2.0.0-rc1 are excluded so an rc
# never auto-deploys); fall back to origin/<default branch> if no release tags exist.
TARGET_REF="$(git tag -l 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
if [ -z "$TARGET_REF" ]; then
  DEF="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#origin/##')"
  TARGET_REF="origin/${DEF:-main}"
  log "No release tag found; tracking $TARGET_REF"
fi

LOCAL="$(git rev-parse HEAD)"
TARGET="$(git rev-parse "${TARGET_REF}^{commit}")"
if [ "$LOCAL" = "$TARGET" ]; then
  log "Already on the latest release ($TARGET_REF). Nothing to do."
  exit 0
fi

log "Updating $(git rev-parse --short "$LOCAL") -> $TARGET_REF ($(git rev-parse --short "$TARGET"))"

# Auto-update applies CODE + dependency changes. The rendered units in /etc/systemd/system
# are only (re)written by install.sh, so if this release changed the unit templates, say so
# loudly rather than silently running stale schedules/units.
if git diff --name-only "$LOCAL" "$TARGET" -- systemd/pi/ | grep -q .; then
  log "NOTE: systemd unit templates changed in this release — re-run 'sudo bash $PROJECT_DIR/scripts/install.sh' to apply unit/schedule changes."
fi

git checkout -q --force "$TARGET_REF"

# Reinstall deps, capturing pip's REAL exit status (not tee's). A swallowed failure that
# then restarts the trader onto a broken venv is the "half-updated box" we must avoid.
set +e
install_ok=1
if [ -f requirements.lock ]; then
  log "Installing from requirements.lock"
  .venv/bin/pip install --quiet -r requirements.lock 2>&1 | tee -a "$LOG_FILE"
  [ "${PIPESTATUS[0]}" -eq 0 ] || install_ok=0
fi
.venv/bin/pip install --quiet -e ".[ai,notifications,robo]" 2>&1 | tee -a "$LOG_FILE"
[ "${PIPESTATUS[0]}" -eq 0 ] || install_ok=0
set -e

if [ "$install_ok" -ne 1 ]; then
  log "ERROR: dependency install failed — rolling back to $(git rev-parse --short "$LOCAL") and NOT restarting the robo"
  git checkout -q --force "$LOCAL"
  .venv/bin/pip install --quiet -e ".[ai,notifications,robo]" >>"$LOG_FILE" 2>&1 || true
  exit 1   # next run sees HEAD != latest tag again and retries; the box keeps running old, working code
fi

if [ "$RESTART" = 1 ]; then
  log "Restarting robo units"
  sudo -n "$SUDO_SYSTEMCTL" daemon-reload 2>/dev/null || true
  # Only restart units that are actually installed (keeps the sudoers surface exact).
  for unit in financewatchdog-research.service financewatchdog-trade.timer \
              financewatchdog-summary.timer financewatchdog-prune.timer \
              financewatchdog-autoupdate.timer; do
    if [ -n "$(systemctl list-unit-files "$unit" --no-legend 2>/dev/null)" ]; then
      sudo -n "$SUDO_SYSTEMCTL" restart "$unit" 2>/dev/null && log "restarted $unit" || log "WARN: could not restart $unit"
    fi
  done
fi

log "Auto-update complete (now at $TARGET_REF)."
exit 0
}
