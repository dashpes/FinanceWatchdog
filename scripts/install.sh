#!/usr/bin/env bash
# One-line installer for the FinanceWatchdog robo advisor on a headless Linux box
# (built for Raspberry Pi OS / Debian on a Pi 5, works on any systemd + apt system).
#
#   curl -fsSL https://raw.githubusercontent.com/dashpes/FinanceWatchdog/main/scripts/install.sh | sudo bash
#
# It is idempotent — re-running updates an existing install in place. It:
#   1. apt-installs Python + git + a build-tool safety net (for ARM wheel gaps)
#   2. installs Ollama and applies the RAM guardrails (one model resident at a time)
#   3. pulls the models (phi3:mini, nomic-embed-text, qwen2.5:14b)
#   4. creates a dedicated `financewatchdog` service user + /opt/financewatchdog
#   5. clones the repo at the latest RELEASE tag and builds the venv (+ a lockfile)
#   6. renders and enables the systemd bundle (research loop + trade/summary/prune/
#      auto-update timers) and sets the timezone to America/Los_Angeles
#   7. runs the interactive `investment-robo init` wizard for credentials (dry-run
#      forced on — you flip it live yourself after reviewing a dry-run cycle)
#
# Override any of these before running:
#   FW_HOME=/opt/financewatchdog  FW_USER=financewatchdog  FW_TZ=America/Los_Angeles
#   FW_REPO=https://github.com/dashpes/FinanceWatchdog.git  FW_REF=<tag/branch>  FW_SKIP_INIT=1
set -euo pipefail

FW_HOME="${FW_HOME:-/opt/financewatchdog}"
FW_USER="${FW_USER:-financewatchdog}"
FW_TZ="${FW_TZ:-America/Los_Angeles}"
FW_REPO="${FW_REPO:-https://github.com/dashpes/FinanceWatchdog.git}"
FW_REF="${FW_REF:-}"   # blank => latest semver tag (falls back to default branch)
FW_MODELS="${FW_MODELS:-phi3:mini nomic-embed-text qwen2.5:14b}"

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

say "Pulling models ($FW_MODELS) — this can take a while on first install"
for m in $FW_MODELS; do
  say "  ollama pull $m"
  ollama pull "$m" || warn "could not pull $m (you can pull it later)"
done

say "Fetching the app into $FW_HOME"
if [ -d "$FW_HOME/.git" ]; then
  sudo -u "$FW_USER" git -C "$FW_HOME" fetch --tags --prune origin
else
  # /opt/financewatchdog exists (service-user home) but is empty; clone into it.
  sudo -u "$FW_USER" git clone "$FW_REPO" "$FW_HOME"
fi
# Resolve the ref: explicit FW_REF, else the latest semver tag, else the default branch.
if [ -z "$FW_REF" ]; then
  # Strict semver only — never auto-select a pre-release tag (v2.0.0-rc1) as "latest".
  FW_REF="$(sudo -u "$FW_USER" git -C "$FW_HOME" tag -l 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
fi
if [ -z "$FW_REF" ]; then
  FW_REF="$(sudo -u "$FW_USER" git -C "$FW_HOME" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#origin/##')"
  FW_REF="${FW_REF:-main}"
  warn "no release tag found — installing the '$FW_REF' branch"
fi
say "Checking out $FW_REF"
sudo -u "$FW_USER" git -C "$FW_HOME" checkout -q "$FW_REF"

say "Building the virtualenv"
if [ ! -d "$FW_HOME/.venv" ]; then
  sudo -u "$FW_USER" python3 -m venv "$FW_HOME/.venv"
fi
sudo -u "$FW_USER" "$FW_HOME/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$FW_USER" "$FW_HOME/.venv/bin/pip" install --quiet -e "$FW_HOME[ai,notifications,robo]"
# Pin the exact resolved set so auto-updates on this box are reproducible (solves ARM
# wheel drift: the lock is built here, against this Pi's Python/arch). --exclude-editable
# keeps the app itself out of the lock (it is reinstalled with `-e` separately).
sudo -u "$FW_USER" bash -c "'$FW_HOME/.venv/bin/pip' freeze --exclude-editable > '$FW_HOME/requirements.lock'"

say "Granting the service user a scoped sudoers entry (systemctl restart only)"
cat > /etc/sudoers.d/financewatchdog <<EOF
# Lets the auto-updater and Ollama health-check restart services without a password.
$FW_USER ALL=(root) NOPASSWD: $SYSTEMCTL daemon-reload, $SYSTEMCTL restart ollama, $SYSTEMCTL restart financewatchdog-research.service, $SYSTEMCTL restart financewatchdog-trade.timer, $SYSTEMCTL restart financewatchdog-summary.timer, $SYSTEMCTL restart financewatchdog-prune.timer, $SYSTEMCTL restart financewatchdog-autoupdate.timer
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

say "Enabling services"
"$SYSTEMCTL" enable --now financewatchdog-research.service 2>/dev/null || true
for t in trade summary prune autoupdate; do
  "$SYSTEMCTL" enable --now "financewatchdog-$t.timer" 2>/dev/null || true
done

say "Done — FinanceWatchdog is installed at $FW_HOME (ref: $FW_REF)"
echo "Trading stays in DRY-RUN until you set ROBO_FORCE_DRY_RUN=false in $FW_HOME/.env"
echo "AND dry_run: false in $FW_HOME/config/robo.yaml. Useful checks:"
echo "  systemctl list-timers 'financewatchdog-*'"
echo "  sudo -u $FW_USER $FW_HOME/.venv/bin/investment-robo check-safety --config $FW_HOME/config"
echo "  journalctl -u financewatchdog-research -f"
