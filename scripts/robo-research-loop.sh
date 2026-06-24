#!/bin/bash
# Continuous overnight research + data-gathering loop for the robo advisor (DAEMON mode).
#
# Overnight the Mac is idle and compute/RAM are free, so the advisor does a full
# research cycle over and over instead of a couple of fixed runs. Each cycle:
#   1. broad-collect  — gather the market-wide event stream (SEC Form 4 insider,
#                       price/volume, news), run the cross-source confluence engine,
#                       and promote the strongest findings to theses. (No Ollama.)
#   2. thesis-run --discover --no-trade — Dow-30 discovery + AI-score + maintain/
#                       re-evaluate the live theses. (Uses Ollama.) NEVER trades.
# After the close the day's data is final, so this is the right time to gather + think.
#
# It self-limits to an overnight window (default 18:00–06:00) so it stays clear of
# the 07:00 & 12:30 trade slots (research and trade share robo-cron.sh's single-run
# lock — overlap would skip a trade) and idle during the day while you use the Mac.
# Everything is env-tunable. Run as a KeepAlive LaunchDaemon (robo-schedule.sh
# daemon-install). broad-collect and the trade runs use separate locks, so the
# gather step never blocks a trade.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
WRAP="$PROJ/scripts/robo-cron.sh"
BROAD_WRAP="$PROJ/scripts/broad-collect-cron.sh"
LOG="$PROJ/logs/robo-research-loop.log"

START_HOUR="${RESEARCH_LOOP_START_HOUR:-18}"    # window opens (local hour, inclusive)
END_HOUR="${RESEARCH_LOOP_END_HOUR:-6}"         # window closes (local hour, exclusive)
COOLDOWN="${RESEARCH_LOOP_COOLDOWN:-1800}"      # seconds to pause between cycles in-window
IDLE_SLEEP="${RESEARCH_LOOP_IDLE_SLEEP:-1800}"  # seconds to wait when outside the window
DAYS_BACK="${RESEARCH_LOOP_DAYS_BACK:-3}"       # broad-collect look-back (dedup makes overlap a no-op)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] research-loop: $*" >> "$LOG"; }

# Current local hour as a base-10 integer (10# avoids octal parsing of 08/09).
current_hour() { echo $(( 10#$(date +%H) )); }

in_window() {  # supports a window that wraps past midnight (e.g. 18 -> 6)
  local h="$1"
  if [ "$START_HOUR" -le "$END_HOUR" ]; then
    [ "$h" -ge "$START_HOUR" ] && [ "$h" -lt "$END_HOUR" ]
  else
    [ "$h" -ge "$START_HOUR" ] || [ "$h" -lt "$END_HOUR" ]
  fi
}

log "starting (window ${START_HOUR}:00-${END_HOUR}:00 local, cooldown ${COOLDOWN}s)"
while true; do
  if in_window "$(current_hour)"; then
    # 1. Gather + confluence + promotion (no Ollama; own lock).
    "$BROAD_WRAP" --days-back "$DAYS_BACK" || log "broad-collect cycle exited non-zero (continuing)"
    # 2. Discovery + AI scoring + thesis maintenance (Ollama; never trades).
    "$WRAP" thesis-run --discover --no-trade || log "research cycle exited non-zero (continuing)"
    sleep "$COOLDOWN"
  else
    sleep "$IDLE_SLEEP"
  fi
done
