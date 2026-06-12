#!/usr/bin/env bash
#
# Investment Monitor - one-command installer (macOS / Linux).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/dashpes/FinanceWatchdog/main/scripts/install.sh | bash
#
# Overridable via environment variables:
#   FW_INSTALL_DIR   target directory      (default: ~/FinanceWatchdog)
#   FW_REPO_URL      git repo to clone     (default: the public repo)
#   FW_BRANCH        branch to install     (default: main)
#   FW_PYTHON        python interpreter    (default: python3)
#
set -euo pipefail

REPO_URL="${FW_REPO_URL:-https://github.com/dashpes/FinanceWatchdog.git}"
INSTALL_DIR="${FW_INSTALL_DIR:-$HOME/FinanceWatchdog}"
BRANCH="${FW_BRANCH:-main}"
PYTHON_BIN="${FW_PYTHON:-python3}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

os="$(uname -s)"

# --- Prerequisites ----------------------------------------------------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || die "Python 3.11+ is required but '$PYTHON_BIN' was not found. Install Python first."
py_ok="$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info[:2] >= (3, 11) else 0)')"
[ "$py_ok" = "1" ] || die "Python 3.11+ required; found $("$PYTHON_BIN" -V 2>&1)."
command -v git >/dev/null 2>&1 || die "git is required but was not found."

# --- Clone or update --------------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
  info "Updating existing checkout in $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" \
    || warn "Could not fast-forward; continuing with the existing checkout."
else
  info "Cloning $REPO_URL into $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# --- Python environment -----------------------------------------------------
info "Creating virtual environment (.venv)"
"$PYTHON_BIN" -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
info "Installing investment-monitor and dependencies (this can take a minute)"
./.venv/bin/python -m pip install --quiet -e ".[ai,notifications,dashboard]"

# --- Ollama (local AI, best-effort) -----------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  info "Installing Ollama (local AI engine)"
  case "$os" in
    Linux)
      curl -fsSL https://ollama.com/install.sh | sh \
        || warn "Ollama auto-install failed; install it from https://ollama.com/download" ;;
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install ollama || warn "brew install ollama failed; see https://ollama.com/download"
      else
        warn "Homebrew not found. Install Ollama from https://ollama.com/download"
      fi ;;
    *)
      warn "Unsupported OS for automatic Ollama install ($os). See https://ollama.com/download" ;;
  esac
fi

# Make sure a server is up so setup can pull models.
if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    info "Starting Ollama server"
    (ollama serve >/dev/null 2>&1 &) || true
    for _ in $(seq 1 10); do
      curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
      sleep 1
    done
  fi
fi

# --- First-run setup --------------------------------------------------------
info "Running first-run setup (config + models)"
./.venv/bin/investment-monitor --setup --yes \
  || warn "Setup reported issues; re-run later: .venv/bin/investment-monitor --setup --yes"

# --- Done -------------------------------------------------------------------
cat <<EOF

✅ Installed in: $INSTALL_DIR

Next steps:
  cd "$INSTALL_DIR"
  \${EDITOR:-nano} config/portfolio.yaml      # add your holdings
  .venv/bin/investment-monitor --doctor       # verify everything
  .venv/bin/investment-monitor --type regular # run it

EOF
