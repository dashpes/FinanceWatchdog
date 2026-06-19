#!/bin/bash
# Install / remove the macOS launchd schedule for the autonomous robo investor.
#
#   scripts/robo-schedule.sh install     # generate plists + load the jobs
#   scripts/robo-schedule.sh uninstall   # unload + remove the plists
#   scripts/robo-schedule.sh status      # show whether the jobs are loaded
#
# Two jobs (times are LOCAL = Pacific; US market hours 06:30-13:00 PT):
#   * robo-research  thesis-run --discover --no-trade   05:00 & 17:00 daily (never trades)
#   * robo-trade     thesis-run                         07:00 & 12:30 Mon-Fri (trades, gated)
#
# Edit the StartCalendarInterval blocks below to change the cadence, then re-run
# `install`. Whether a trade is real depends on dry_run/ROBO_FORCE_DRY_RUN — the
# schedule itself is independent of going live.
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
WRAP="$PROJ/scripts/robo-cron.sh"
BROAD_WRAP="$PROJ/scripts/broad-collect-cron.sh"
AGENTS="$HOME/Library/LaunchAgents"
RESEARCH="com.financewatchdog.robo-research"
TRADE="com.financewatchdog.robo-trade"
OLLAMA_ENV="com.financewatchdog.ollama-env"
BROAD="com.financewatchdog.broad-collect"

write_ollama_env_plist() {
  # Re-apply the Ollama RAM guardrails on every login (launchctl setenv does not
  # survive a reboot), then restart Ollama so the server inherits them.
  cat > "$AGENTS/$OLLAMA_ENV.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$OLLAMA_ENV</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string><string>-c</string>
    <string>launchctl setenv OLLAMA_MAX_LOADED_MODELS 1; launchctl setenv OLLAMA_NUM_PARALLEL 1; launchctl setenv OLLAMA_KEEP_ALIVE 5m; /opt/homebrew/bin/brew services restart ollama</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
PLIST
}

write_research_plist() {
  cat > "$AGENTS/$RESEARCH.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$RESEARCH</string>
  <key>ProgramArguments</key>
  <array>
    <string>$WRAP</string><string>thesis-run</string><string>--discover</string><string>--no-trade</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>5</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-research.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-research.log</string>
</dict></plist>
PLIST
}

write_trade_plist() {
  # Weekdays (Mon=1..Fri=5) at 07:00 and 12:30 PT — both inside market hours.
  local entries=""
  for wd in 1 2 3 4 5; do
    entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>\n"
    entries+="    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>30</integer></dict>\n"
  done
  cat > "$AGENTS/$TRADE.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$TRADE</string>
  <key>ProgramArguments</key>
  <array>
    <string>$WRAP</string><string>thesis-run</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
$(printf "%b" "$entries")  </array>
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-trade.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-trade.log</string>
</dict></plist>
PLIST
}

write_broad_collect_plist() {
  # Broad market-wide event ingestion (SEC Form 4 today; more sources later). Daily
  # at 04:30 PT so fresh data is in the DB before the 05:00 research run. No Ollama.
  # --days-back 3 catches up to ~2 missed days; dedup makes the overlap a no-op.
  cat > "$AGENTS/$BROAD.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$BROAD</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BROAD_WRAP</string><string>--days-back</string><string>3</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-broad-collect.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-broad-collect.log</string>
</dict></plist>
PLIST
}

cmd="${1:-status}"
case "$cmd" in
  install)
    mkdir -p "$AGENTS" "$PROJ/logs"
    chmod +x "$WRAP" "$BROAD_WRAP"
    write_ollama_env_plist
    write_research_plist
    write_trade_plist
    write_broad_collect_plist
    for label in "$OLLAMA_ENV" "$RESEARCH" "$TRADE" "$BROAD"; do
      launchctl unload "$AGENTS/$label.plist" 2>/dev/null || true
      launchctl load "$AGENTS/$label.plist"
      echo "loaded $label"
    done
    echo "Installed. Ollama RAM caps persist on boot. Research: 05:00 & 17:00 daily"
    echo "(no-trade). Trade: 07:00 & 12:30 Mon-Fri (gated). Broad collect: 04:30 daily."
    ;;
  uninstall)
    for label in "$OLLAMA_ENV" "$RESEARCH" "$TRADE" "$BROAD"; do
      launchctl unload "$AGENTS/$label.plist" 2>/dev/null || true
      rm -f "$AGENTS/$label.plist"
      echo "removed $label"
    done
    ;;
  status)
    launchctl list | grep -E "financewatchdog" || echo "(no robo jobs loaded)"
    ;;
  *)
    echo "usage: $0 {install|uninstall|status}" >&2; exit 2 ;;
esac
