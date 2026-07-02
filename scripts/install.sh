#!/usr/bin/env bash
# One-line installer for the FinanceWatchdog robo advisor on a headless Linux box
# (built for Raspberry Pi OS / Debian on a Pi 5, works on any systemd + apt system).
#
#   curl -fsSL https://raw.githubusercontent.com/dashpes/FinanceWatchdog/main/scripts/install.sh | sudo bash
#
# It is idempotent — re-running updates an existing install in place. It:
#   1. apt-installs Python + git + a build-tool safety net (for ARM wheel gaps)
#   2. installs Ollama and applies the RAM guardrails (one model resident at a time)
#   3. pulls the models (phi3:mini, nomic-embed-text, qwen2.5:14b) IN THE BACKGROUND, so
#      the multi-GB 14B download doesn't hold up the install (services fail-open until ready)
#   4. creates a dedicated `financewatchdog` service user + /opt/financewatchdog
#   5. clones the repo at the latest RELEASE tag and builds the venv (+ a lockfile)
#   6. renders and enables the systemd bundle (research loop + LAN dashboard + trade/
#      summary/prune/auto-update timers) and sets the timezone to America/Los_Angeles
#   7. runs the interactive `investment-robo init` wizard for credentials (dry-run
#      forced on — you flip it live yourself after reviewing a dry-run cycle)
#
# Override any of these before running:
#   FW_HOME=/opt/financewatchdog  FW_USER=financewatchdog  FW_TZ=America/Los_Angeles
#   FW_REPO=https://github.com/dashpes/FinanceWatchdog.git  FW_REF=<tag/branch>  FW_SKIP_INIT=1
#   FW_NO_CLONE=1   # you already placed the code at FW_HOME (git clone or scp) — don't clone
#   FW_MODELS=""    # skip model pulls (default: phi3:mini nomic-embed-text qwen2.5:14b, in bg)
#
# PRIVATE REPO / managing the code yourself: clone or scp the repo to FW_HOME first, then run
# with FW_NO_CLONE=1 (the installer never needs its own repo credentials). For git auto-update
# to keep working afterward, run as a user that can pull — i.e. FW_USER=<your user> — since the
# default 'financewatchdog' service account has no SSH key. If it can't fetch, the installer
# leaves auto-update off and you update by re-copying the code and re-running this script.
set -euo pipefail

FW_HOME="${FW_HOME:-/opt/financewatchdog}"
FW_USER="${FW_USER:-financewatchdog}"
FW_TZ="${FW_TZ:-America/Los_Angeles}"
FW_REPO="${FW_REPO:-https://github.com/dashpes/FinanceWatchdog.git}"
FW_REF="${FW_REF:-}"   # blank => latest semver tag (falls back to default branch)
FW_MODELS="${FW_MODELS-phi3:mini nomic-embed-text qwen2.5:14b}"   # FW_MODELS="" skips pulls

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root:  curl -fsSL <url>/install.sh | sudo bash"
command -v apt-get >/dev/null   || die "this installer targets Debian/Raspberry Pi OS (needs apt-get)"
command -v systemctl >/dev/null || die "this installer needs systemd (systemctl not found)"
SYSTEMCTL="$(command -v systemctl)"

say "Installing OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# python + git + the wheel-gap safety net (rustc/cc/libxml for any dep lacking an aarch64 wheel).
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-dev git curl ca-certificates jq \
  build-essential rustc libxml2-dev libxslt1-dev

say "Setting timezone to $FW_TZ"
timedatectl set-timezone "$FW_TZ" 2>/dev/null || warn "could not set timezone (continuing)"

say "Creating service user '$FW_USER' and $FW_HOME"
# NB: no --create-home — skeleton dotfiles would leave $FW_HOME non-empty and break the
# git clone below. We create the (empty) dir ourselves and hand it to the user.
if ! id -u "$FW_USER" >/dev/null 2>&1; then
  useradd --system --home-dir "$FW_HOME" --shell /usr/sbin/nologin "$FW_USER"
fi
mkdir -p "$FW_HOME"
chown "$FW_USER:$FW_USER" "$FW_HOME"

say "Installing Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
# RAM guardrails so a 16GB Pi never holds two models resident (phi3:mini + the 14b).
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_KEEP_ALIVE=5m
EOF
"$SYSTEMCTL" daemon-reload
"$SYSTEMCTL" enable --now ollama 2>/dev/null || "$SYSTEMCTL" restart ollama || true

mkdir -p "$FW_HOME/logs"
PULL_LOG="$FW_HOME/logs/ollama-pull.log"
if [ -z "${FW_MODELS// /}" ]; then
  say "Skipping model pulls (FW_MODELS empty)"
else
  # Skip models already downloaded (normalise the implicit :latest tag), then pull any
  # missing ones in the BACKGROUND so a multi-GB download doesn't block the install.
  present="$(ollama list 2>/dev/null | awk 'NR>1{sub(/:latest$/,"",$1); print $1}')"
  to_pull=""
  for m in $FW_MODELS; do
    if printf '%s\n' "$present" | grep -qxF "${m%:latest}"; then
      say "  $m already present"
    else
      to_pull="$to_pull $m"
    fi
  done
  if [ -z "${to_pull// /}" ]; then
    say "All requested models already present"
  else
    say "Pulling models in the BACKGROUND ($to_pull ) — progress: tail -f $PULL_LOG"
    nohup bash -c "for m in$to_pull; do ollama pull \"\$m\" || echo \"WARN: could not pull \$m\"; done" \
      </dev/null >>"$PULL_LOG" 2>&1 &
    disown
  fi
fi

say "Placing the app in $FW_HOME"
# MANAGED=1 means the installer owns the git checkout (it can fetch + pin a release tag).
# HAS_GIT=1 means a .git exists at all. Neither requires the installer to have repo creds
# in the FW_NO_CLONE / existing-checkout paths.
HAS_GIT=0; MANAGED=0
if [ "${FW_NO_CLONE:-0}" = 1 ]; then
  [ -f "$FW_HOME/pyproject.toml" ] || die "FW_NO_CLONE=1 but no checkout at $FW_HOME (expected pyproject.toml). Clone or scp the repo there first."
  [ -d "$FW_HOME/.git" ] && HAS_GIT=1
  say "Using your existing checkout at $FW_HOME (installer will not touch the code)"
elif [ -d "$FW_HOME/.git" ]; then
  HAS_GIT=1
  say "Found a git checkout in $FW_HOME"
  if sudo -u "$FW_USER" git -C "$FW_HOME" fetch --tags --prune origin >/dev/null 2>&1; then
    MANAGED=1
  else
    warn "could not fetch as '$FW_USER' (no repo access for that user) — using the checkout as-is"
  fi
elif [ -f "$FW_HOME/pyproject.toml" ]; then
  say "Using your existing (non-git) checkout at $FW_HOME — auto-update disabled (update by re-copying)"
else
  say "Cloning $FW_REPO into $FW_HOME"
  sudo -u "$FW_USER" git clone "$FW_REPO" "$FW_HOME"
  HAS_GIT=1; MANAGED=1
fi

# Only when the installer owns the checkout: pin it to the release ref (else respect what
# you placed). Explicit FW_REF, else the latest strict-semver tag, else the default branch.
if [ "$MANAGED" = 1 ]; then
  if [ -z "$FW_REF" ]; then
    FW_REF="$(sudo -u "$FW_USER" git -C "$FW_HOME" tag -l 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
  fi
  if [ -z "$FW_REF" ]; then
    FW_REF="$(sudo -u "$FW_USER" git -C "$FW_HOME" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#origin/##')"
    FW_REF="${FW_REF:-main}"
    warn "no release tag found — installing the '$FW_REF' branch"
  fi
  say "Checking out $FW_REF"
  sudo -u "$FW_USER" git -C "$FW_HOME" checkout -q "$FW_REF"
fi

# The runtime user must own the tree (venv/.env/data/logs writes; git-as-owner for updates).
chown -R "$FW_USER:$FW_USER" "$FW_HOME"

say "Building the virtualenv"
if [ ! -d "$FW_HOME/.venv" ]; then
  sudo -u "$FW_USER" python3 -m venv "$FW_HOME/.venv"
fi
# Network-resilient pip: a Pi install often competes with multi-GB Ollama model
# downloads for bandwidth, which caused SSL-EOF / read-timeout failures on piwheels.
# Retry hard, extend the timeout, prefer prebuilt wheels, and never let a failed pip
# self-upgrade abort the install (the venv's bundled pip is fine).
PIP_NET="--retries 10 --timeout 120 --prefer-binary"
sudo -u "$FW_USER" "$FW_HOME/.venv/bin/pip" install --quiet $PIP_NET --upgrade pip \
  || warn "pip self-upgrade skipped (using the venv's bundled pip)"
sudo -u "$FW_USER" "$FW_HOME/.venv/bin/pip" install $PIP_NET -e "$FW_HOME[ai,notifications,robo,dashboard]"
# Pin the exact resolved set so auto-updates on this box are reproducible (solves ARM
# wheel drift: the lock is built here, against this Pi's Python/arch). --exclude-editable
# keeps the app itself out of the lock (it is reinstalled with `-e` separately).
sudo -u "$FW_USER" bash -c "'$FW_HOME/.venv/bin/pip' freeze --exclude-editable > '$FW_HOME/requirements.lock'"

say "Granting the service user a scoped sudoers entry (systemctl restart only)"
cat > /etc/sudoers.d/financewatchdog <<EOF
# Lets the auto-updater and Ollama health-check restart services without a password.
$FW_USER ALL=(root) NOPASSWD: $SYSTEMCTL daemon-reload, $SYSTEMCTL restart ollama, $SYSTEMCTL restart financewatchdog-research.service, $SYSTEMCTL restart financewatchdog-dashboard.service, $SYSTEMCTL restart financewatchdog-trade.timer, $SYSTEMCTL restart financewatchdog-summary.timer, $SYSTEMCTL restart financewatchdog-prune.timer, $SYSTEMCTL restart financewatchdog-autoupdate.timer
EOF
chmod 440 /etc/sudoers.d/financewatchdog
visudo -cf /etc/sudoers.d/financewatchdog >/dev/null || die "generated sudoers file is invalid"

say "Rendering and installing the systemd bundle"
render() { sed -e "s#@FW_USER@#$FW_USER#g" -e "s#@FW_HOME@#$FW_HOME#g" "$1"; }
for unit in "$FW_HOME"/systemd/pi/financewatchdog-*.service "$FW_HOME"/systemd/pi/financewatchdog-*.timer; do
  render "$unit" > "/etc/systemd/system/$(basename "$unit")"
done
"$SYSTEMCTL" daemon-reload

say "Configuring credentials"
INIT_HINT="sudo -u $FW_USER env FW_HOME=$FW_HOME $FW_HOME/.venv/bin/investment-robo init --config $FW_HOME/config"
if [ "${FW_SKIP_INIT:-0}" = 1 ]; then
  warn "FW_SKIP_INIT=1 — skipping the wizard. Run it later with:"
  echo "  $INIT_HINT"
elif [ -e /dev/tty ]; then
  # Run FROM $FW_HOME so .env lands there (services read $FW_HOME/.env), and read prompts
  # from the terminal even though this script itself arrived on stdin via curl|bash.
  sudo -u "$FW_USER" env "FW_HOME=$FW_HOME" "PYTHONPATH=$FW_HOME/src" \
    bash -c 'cd "$FW_HOME" && exec .venv/bin/investment-robo init --config config' < /dev/tty || \
    warn "wizard did not complete — re-run:  $INIT_HINT"
else
  warn "no terminal available for the interactive wizard. Run it later with:"
  echo "  $INIT_HINT"
fi

# Lock down .env however it arrived — a scp'd secrets file isn't 0600 (scp doesn't
# preserve mode) and must be owned by the runtime user to be readable by the services.
if [ -f "$FW_HOME/.env" ]; then
  chown "$FW_USER:$FW_USER" "$FW_HOME/.env"
  chmod 600 "$FW_HOME/.env"
fi

# Dashboard PIN: generated once, appended to .env. It gates every mutating endpoint
# (pause/kill/blocklist/settings); reads are open on the LAN. Shown once at the end.
DASH_TOKEN=""
if [ -f "$FW_HOME/.env" ] && ! grep -q '^DASHBOARD_TOKEN=' "$FW_HOME/.env"; then
  DASH_TOKEN="$(openssl rand -hex 8 2>/dev/null || head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  printf '\n# Dashboard PIN — gates pause/kill/blocklist/settings on the web GUI\nDASHBOARD_TOKEN=%s\n' "$DASH_TOKEN" >> "$FW_HOME/.env"
fi
# Default the appliance to port 80 (the unit grants CAP_NET_BIND_SERVICE), so the
# GUI is just http://<host>.local — override with DASHBOARD_PORT in .env.
if [ -f "$FW_HOME/.env" ] && ! grep -q '^DASHBOARD_PORT=' "$FW_HOME/.env"; then
  printf 'DASHBOARD_PORT=80\n' >> "$FW_HOME/.env"
fi
DASH_PORT="$(grep '^DASHBOARD_PORT=' "$FW_HOME/.env" 2>/dev/null | tail -1 | cut -d= -f2)"
DASH_PORT="${DASH_PORT:-8321}"

say "Enabling services"
"$SYSTEMCTL" enable --now financewatchdog-research.service 2>/dev/null || true
"$SYSTEMCTL" enable --now financewatchdog-dashboard.service 2>/dev/null || true
for t in trade summary prune; do
  "$SYSTEMCTL" enable --now "financewatchdog-$t.timer" 2>/dev/null || true
done
# Auto-update only makes sense if the runtime user can actually pull the repo.
if [ "$MANAGED" = 1 ]; then
  "$SYSTEMCTL" enable --now financewatchdog-autoupdate.timer 2>/dev/null || true
else
  warn "auto-update timer NOT enabled ('$FW_USER' can't fetch the repo). Update by re-copying the code + re-running this script, or reinstall with FW_USER set to a user that can pull."
fi

say "Done — FinanceWatchdog is installed at $FW_HOME (ref: $FW_REF)"
if [ "$DASH_PORT" = 80 ]; then
  echo "Dashboard (Archie's web GUI): http://$(hostname).local  (any browser on your wifi)"
else
  echo "Dashboard (Archie's web GUI): http://$(hostname).local:$DASH_PORT  (any browser on your wifi)"
fi
if [ -n "$DASH_TOKEN" ]; then
  echo "  Dashboard PIN (for pause/kill/blocklist/settings): $DASH_TOKEN"
  echo "  (also saved as DASHBOARD_TOKEN in $FW_HOME/.env)"
fi
echo "Trading stays in DRY-RUN until you set ROBO_FORCE_DRY_RUN=false in $FW_HOME/.env"
echo "AND dry_run: false in $FW_HOME/config/robo.yaml. Useful checks:"
echo "  systemctl list-timers 'financewatchdog-*'"
echo "  sudo -u $FW_USER $FW_HOME/.venv/bin/investment-robo check-safety --config $FW_HOME/config"
echo "  journalctl -u financewatchdog-research -f"
echo "Model downloads may still be finishing in the background:"
echo "  tail -f $FW_HOME/logs/ollama-pull.log   (or: ollama list)"

# If a copied .env/robo.yaml already arms live trading, warn about the two-bots-one-account trap.
if grep -qE '^[[:space:]]*ROBO_FORCE_DRY_RUN[[:space:]]*=[[:space:]]*false' "$FW_HOME/.env" 2>/dev/null; then
  warn "LIVE TRADING is armed in this .env (ROBO_FORCE_DRY_RUN=false). If ANOTHER machine (e.g. your Mac) still trades this brokerage account, both bots will place orders on the SAME account and conflict. Stand down the other deployment first, or set ROBO_FORCE_DRY_RUN=true here until you've validated a dry-run cycle on the Pi."
fi
