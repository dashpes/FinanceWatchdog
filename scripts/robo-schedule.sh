#!/bin/bash
# Install / remove the launchd schedule for the autonomous robo investor.
#
# TWO DEPLOYMENT MODES:
#   * AGENT  (legacy): per-user LaunchAgents in ~/Library/LaunchAgents. These run
#     ONLY while you are logged in — a logout or auto-logout silently takes the
#     whole robo (and Ollama) offline. This bit us 2026-06-20.
#   * DAEMON (recommended): system LaunchDaemons in /Library/LaunchDaemons that run
#     as your user regardless of who (if anyone) is logged in. Survives logout; the
#     Mac stays a normal computer. Needs sudo. Also runs Ollama as a daemon so the
#     local model is available headless (the research/trade runs need it).
#
# Usage:
#   scripts/robo-schedule.sh install            # AGENT mode (logged-in only)
#   scripts/robo-schedule.sh uninstall
#   scripts/robo-schedule.sh status
#   sudo scripts/robo-schedule.sh daemon-install     # DAEMON mode (headless 24/7)
#   sudo scripts/robo-schedule.sh daemon-uninstall
#   scripts/robo-schedule.sh daemon-status
#
# Jobs (times are LOCAL = Pacific; US market hours 06:30-13:00 PT):
#   * ollama         ollama serve                        always-on (DAEMON mode only)
#   * robo-research  DAEMON: continuous overnight loop (robo-research-loop.sh, default
#                    18:00-06:00) that each cycle runs broad-collect (gather + confluence
#                    + promote) THEN thesis-run --discover --no-trade (score + maintain).
#                    Never trades. AGENT mode: discrete thesis-run at 05:00 & 17:00.
#   * robo-trade     thesis-run                          07:00 & 12:30 Mon-Fri (trades, gated)
#   * broad-collect  AGENT mode only (04:30 daily); in DAEMON mode the research loop owns it
#   * robo-summary   daily-summary                       13:15 Mon-Fri (email/iMessage recap)
#
# Whether a trade is real depends on dry_run/ROBO_FORCE_DRY_RUN — the schedule is
# independent of going live.
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
WRAP="$PROJ/scripts/robo-cron.sh"
BROAD_WRAP="$PROJ/scripts/broad-collect-cron.sh"
RESEARCH_LOOP="$PROJ/scripts/robo-research-loop.sh"
AGENTS="$HOME/Library/LaunchAgents"
DAEMONS="/Library/LaunchDaemons"
OLLAMA_BIN="/opt/homebrew/bin/ollama"
BREW="/opt/homebrew/bin/brew"

RESEARCH="com.financewatchdog.robo-research"
TRADE="com.financewatchdog.robo-trade"
OLLAMA_ENV="com.financewatchdog.ollama-env"   # legacy login agent (AGENT mode only)
OLLAMA="com.financewatchdog.ollama"           # ollama server daemon (DAEMON mode only)
BROAD="com.financewatchdog.broad-collect"
SUMMARY="com.financewatchdog.robo-summary"

# Resolve the REAL invoking user/home even under sudo, so daemons run as you (with
# access to the venv, .env, ~/.ollama, and the Public token) — never as root.
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_HOME="$(dscl . -read "/Users/$RUN_USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}' || true)"
[ -n "$RUN_HOME" ] || RUN_HOME="/Users/$RUN_USER"

# --- shared StartCalendarInterval fragments --------------------------------------

research_sched() {
  cat <<'X'
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>5</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>0</integer></dict>
  </array>
X
}

broad_sched() {
  cat <<'X'
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>30</integer></dict>
X
}

weekday_sched() {  # args: HH:MM pairs, e.g. weekday_sched 7:0 12:30
  echo '  <key>StartCalendarInterval</key>'
  echo '  <array>'
  local wd hm h m
  for wd in 1 2 3 4 5; do
    for hm in "$@"; do
      h="${hm%%:*}"; m="${hm##*:}"
      printf '    <dict><key>Weekday</key><integer>%s</integer><key>Hour</key><integer>%s</integer><key>Minute</key><integer>%s</integer></dict>\n' "$wd" "$h" "$m"
    done
  done
  echo '  </array>'
}

# ================================================================================
# AGENT mode (legacy, logged-in only) — kept for rollback.
# ================================================================================

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
    <string>launchctl setenv OLLAMA_MAX_LOADED_MODELS 1; launchctl setenv OLLAMA_NUM_PARALLEL 1; launchctl setenv OLLAMA_KEEP_ALIVE 5m; $BREW services restart ollama</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
PLIST
}

write_research_plist() {
  { cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$RESEARCH</string>
  <key>ProgramArguments</key>
  <array>
    <string>$WRAP</string><string>thesis-run</string><string>--discover</string><string>--no-trade</string>
  </array>
PLIST
    research_sched
    cat <<PLIST
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-research.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-research.log</string>
</dict></plist>
PLIST
  } > "$AGENTS/$RESEARCH.plist"
}

write_trade_plist() {
  { cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$TRADE</string>
  <key>ProgramArguments</key>
  <array>
    <string>$WRAP</string><string>thesis-run</string>
  </array>
PLIST
    weekday_sched 7:0 12:30
    cat <<PLIST
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-trade.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-trade.log</string>
</dict></plist>
PLIST
  } > "$AGENTS/$TRADE.plist"
}

write_broad_collect_plist() {
  { cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$BROAD</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BROAD_WRAP</string><string>--days-back</string><string>3</string>
  </array>
PLIST
    broad_sched
    cat <<PLIST
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-broad-collect.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-broad-collect.log</string>
</dict></plist>
PLIST
  } > "$AGENTS/$BROAD.plist"
}

write_summary_plist() {
  { cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$SUMMARY</string>
  <key>ProgramArguments</key>
  <array>
    <string>$WRAP</string><string>daily-summary</string>
  </array>
PLIST
    weekday_sched 13:15
    cat <<PLIST
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-summary.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-summary.log</string>
</dict></plist>
PLIST
  } > "$AGENTS/$SUMMARY.plist"
}

# ================================================================================
# DAEMON mode (recommended, headless 24/7).
# ================================================================================

# Emit one system-daemon plist. Args:
#   $1 dir  $2 label  $3 program-args XML  $4 schedule XML  $5 extra XML (optional)
# Every daemon runs as $RUN_USER with HOME + a sane PATH + the Ollama RAM caps
# (harmless for the non-Ollama jobs; the Ollama server reads them).
write_daemon() {
  local dir="$1" label="$2" prog="$3" sched="$4" extra="${5:-}" short="${2##*.}"
  cat > "$dir/$label.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>UserName</key><string>$RUN_USER</string>
  <key>GroupName</key><string>staff</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>$RUN_HOME</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>OLLAMA_MAX_LOADED_MODELS</key><string>1</string>
    <key>OLLAMA_NUM_PARALLEL</key><string>1</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>5m</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
$prog
  </array>
$sched$extra
  <key>StandardOutPath</key><string>$PROJ/logs/launchd-$short.log</string>
  <key>StandardErrorPath</key><string>$PROJ/logs/launchd-$short.log</string>
</dict></plist>
PLIST
}

write_daemon_plists() {  # $1 = target dir
  local d="$1"
  # Ollama server — always-on (no schedule); KeepAlive restarts it if it dies.
  write_daemon "$d" "$OLLAMA" \
"    <string>$OLLAMA_BIN</string><string>serve</string>" \
"" \
"  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
"
  # Research runs CONTINUOUSLY overnight via a KeepAlive loop (free idle compute),
  # not on a fixed schedule; the loop self-limits to its overnight window so it stays
  # clear of the 07:00 & 12:30 trade slots (shared lock) and idle during the day.
  write_daemon "$d" "$RESEARCH" \
"    <string>$RESEARCH_LOOP</string>" \
"" \
"  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
"
  write_daemon "$d" "$TRADE" \
"    <string>$WRAP</string><string>thesis-run</string>" \
"$(weekday_sched 7:0 12:30)"
  # NB: no standalone broad-collect daemon here — the overnight research loop above
  # runs broad-collect (gather + confluence + promote) at the start of every cycle,
  # so data gathering and analysis share one continuous overnight pipeline.
  write_daemon "$d" "$SUMMARY" \
"    <string>$WRAP</string><string>daily-summary</string>" \
"$(weekday_sched 13:15)"
}

DAEMON_LABELS=("$OLLAMA" "$RESEARCH" "$TRADE" "$SUMMARY")

require_root() {
  [ "$(id -u)" -eq 0 ] || { echo "This must run as root:  sudo $0 $1" >&2; exit 1; }
}

# --- commands --------------------------------------------------------------------

cmd="${1:-status}"
case "$cmd" in
  install)
    mkdir -p "$AGENTS" "$PROJ/logs"
    chmod +x "$WRAP" "$BROAD_WRAP"
    write_ollama_env_plist
    write_research_plist
    write_trade_plist
    write_broad_collect_plist
    write_summary_plist
    for label in "$OLLAMA_ENV" "$RESEARCH" "$TRADE" "$BROAD" "$SUMMARY"; do
      launchctl unload "$AGENTS/$label.plist" 2>/dev/null || true
      launchctl load "$AGENTS/$label.plist"
      echo "loaded $label"
    done
    echo "Installed AGENT mode (runs ONLY while logged in). Research 05:00 & 17:00,"
    echo "trade 07:00 & 12:30 Mon-Fri, broad-collect 04:30, summary 13:15 Mon-Fri."
    echo "For true headless 24/7 use:  sudo $0 daemon-install"
    ;;

  uninstall)
    for label in "$OLLAMA_ENV" "$RESEARCH" "$TRADE" "$BROAD" "$SUMMARY"; do
      launchctl unload "$AGENTS/$label.plist" 2>/dev/null || true
      rm -f "$AGENTS/$label.plist"
      echo "removed $label"
    done
    ;;

  daemon-install)
    require_root daemon-install
    mkdir -p "$PROJ/logs"
    chmod +x "$WRAP" "$BROAD_WRAP" "$RESEARCH_LOOP"

    # Stand down the legacy per-user agents and free Ollama's port so the daemon
    # can bind it. (Run as the real user, since these live in their GUI domain.)
    uid="$(id -u "$RUN_USER")"
    for label in "$OLLAMA_ENV" "$RESEARCH" "$TRADE" "$BROAD" "$SUMMARY"; do
      launchctl bootout "gui/$uid/$label" 2>/dev/null || true
    done
    sudo -u "$RUN_USER" "$BREW" services stop ollama 2>/dev/null || true

    write_daemon_plists "$DAEMONS"
    for label in "${DAEMON_LABELS[@]}"; do
      chown root:wheel "$DAEMONS/$label.plist"
      chmod 644 "$DAEMONS/$label.plist"
      launchctl bootout "system/$label" 2>/dev/null || true
      launchctl bootstrap system "$DAEMONS/$label.plist"
      echo "loaded $label"
    done
    echo
    echo "Installed DAEMON mode — runs as $RUN_USER, survives logout/idle."
    echo "Overnight (18:00-06:00) the research loop continuously gathers data +"
    echo "runs confluence + scores/maintains theses. Trades 07:00 & 12:30 Mon-Fri."
    echo "Ollama now runs as a daemon (KeepAlive). Verify shortly with:"
    echo "  curl -s localhost:11434/api/tags >/dev/null && echo ollama-ok"
    echo "  $0 daemon-status"
    echo "Notifications go out by email when SMTP_* + EMAIL_TO are set in .env"
    echo "(verify:  PYTHONPATH=src $PROJ/.venv/bin/investment-robo notify-test)."
    echo "NOTE: a hard reboot still needs you to unlock FileVault once before any"
    echo "daemon runs."
    ;;

  daemon-uninstall)
    require_root daemon-uninstall
    for label in "${DAEMON_LABELS[@]}"; do
      launchctl bootout "system/$label" 2>/dev/null || true
      rm -f "$DAEMONS/$label.plist"
      echo "removed $label"
    done
    echo "Daemons removed. Re-enable Ollama for interactive use with:"
    echo "  brew services start ollama"
    ;;

  status)
    launchctl list | grep -E "financewatchdog" || echo "(no robo agents loaded)"
    ;;

  daemon-status)
    found=0
    for label in "${DAEMON_LABELS[@]}"; do
      if out="$(launchctl print "system/$label" 2>/dev/null)"; then
        found=1
        printf '%-40s %s\n' "$label" "$(echo "$out" | grep -E 'state =' | head -1 | xargs)"
      fi
    done
    [ "$found" -eq 1 ] || echo "(no robo daemons loaded — run: sudo $0 daemon-install)"
    ;;

  *)
    echo "usage: $0 {install|uninstall|status|daemon-install|daemon-uninstall|daemon-status}" >&2
    exit 2 ;;
esac
