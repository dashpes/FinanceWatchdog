#!/bin/bash
# Auto-update script for Investment Monitor
# Pulls latest changes from main branch
#
# Usage: ./auto_update.sh [--restart]
#   --restart  Restart systemd services after pulling updates
#
# Setup (run once on your server):
#   git update-index --skip-worktree docker-compose.yaml
#   git update-index --skip-worktree systemd/investment-monitor.service
#   git update-index --skip-worktree systemd/investment-digest.service
#   git update-index --skip-worktree systemd/investment-weekly.service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_DIR}/logs/auto_update.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$PROJECT_DIR"

log "Starting auto-update check..."

# Fetch latest from origin
if ! git fetch origin main 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: git fetch failed"
    exit 1
fi

# Check if we're behind origin
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date."
    exit 0
fi

# Count commits behind
BEHIND=$(git rev-list HEAD..origin/main --count)
log "Found $BEHIND new commit(s). Pulling updates..."

# Pull changes (skip-worktree files won't be touched)
if ! git pull origin main 2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: git pull failed"
    exit 1
fi

log "Successfully updated to $(git rev-parse --short HEAD)"

# Check if dependencies changed
if git diff --name-only "$LOCAL" "$REMOTE" | grep -q "pyproject.toml"; then
    log "Dependencies changed. Installing..."
    if [ -d ".venv" ]; then
        .venv/bin/pip install -e . 2>&1 | tee -a "$LOG_FILE"
    fi
fi

# Restart services if requested
if [ "$1" = "--restart" ]; then
    log "Restarting systemd services..."

    # Reload systemd in case service files changed
    systemctl daemon-reload 2>/dev/null || true

    # Restart the timer-triggered services
    for service in investment-monitor investment-digest investment-weekly; do
        if systemctl is-enabled --quiet "$service.timer" 2>/dev/null; then
            systemctl restart "$service.timer"
            log "Restarted $service.timer"
        fi
    done

    log "Services restarted."
fi

log "Auto-update complete."
exit 0
