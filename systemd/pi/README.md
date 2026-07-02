# Raspberry Pi / headless Linux deployment

The one-line installer sets all of this up for you:

```bash
curl -fsSL https://raw.githubusercontent.com/dashpes/FinanceWatchdog/main/scripts/install.sh | sudo bash
```

It provisions a dedicated `financewatchdog` service user under `/opt/financewatchdog`,
installs Ollama with the RAM guardrails, pulls the models, builds the venv, renders the
units below, and runs the `investment-robo init` credential wizard (trading stays in
**dry-run** until you flip `ROBO_FORCE_DRY_RUN=false` in `.env` **and** `dry_run: false`
in `config/robo.yaml`).

### Private repo / you place the code yourself

If the repo is private (or you'd rather `git clone`/`scp` it over SSH), put it at
`/opt/financewatchdog` first and tell the installer not to clone:

```bash
sudo git clone git@github.com:dashpes/FinanceWatchdog.git /opt/financewatchdog   # or scp it there
sudo FW_NO_CLONE=1 FW_USER=$USER bash /opt/financewatchdog/scripts/install.sh
```

The installer never needs its own GitHub credentials this way. Set `FW_USER` to a user that
can pull the repo if you want git **auto-update** to keep working (the default isolated
`financewatchdog` account has no keys, so the installer leaves the auto-update timer off and
you update by re-copying the code and re-running the installer). A plain `scp`'d copy (no
`.git`) always updates that way.

## The bundle

These files are templates — the installer substitutes `@FW_USER@` / `@FW_HOME@` and
writes the result into `/etc/systemd/system/`.

| Unit | Type | Schedule (local time) | What it runs |
|------|------|-----------------------|--------------|
| `financewatchdog-research.service` | long-running | continuous (self-limits 18:00–06:00) | overnight data gather + confluence + thesis scoring (never trades) |
| `financewatchdog-dashboard.service` | long-running | always on | Archie's web GUI on the LAN — `http://<pi>.local:8321` |
| `financewatchdog-trade.timer` | timer → oneshot | Mon–Fri 07:00 & 12:30 | `thesis-run` (gated by dry-run + kill-switch) |
| `financewatchdog-summary.timer` | timer → oneshot | Mon–Fri 13:15 | `daily-summary` email |
| `financewatchdog-prune.timer` | timer → oneshot | Sun 12:00 | retention prune + `VACUUM` |
| `financewatchdog-autoupdate.timer` | timer → oneshot | daily 06:15 | update to the latest **release tag**, restart units |
| `ollama.service.d/override.conf` | drop-in | — | `OLLAMA_MAX_LOADED_MODELS=1`, `KEEP_ALIVE=5m` (one model resident at a time) |

Timers use **local** time; the installer sets the Pi timezone to `America/Los_Angeles`
so they line up with US market hours. The market-hours gate in the app is timezone-aware
regardless, so a wrong TZ affects *when jobs fire*, not *whether trading is allowed*.

## Updates

`financewatchdog-autoupdate.timer` pulls the latest **release tag** daily and applies
**code + dependency** changes (from `requirements.lock`), rolling back and skipping the
restart if the dependency install fails (so a bad release never leaves a broken trader).
Changes to the **systemd units themselves** are *not* auto-applied — the auto-updater logs
a `NOTE` when a release touches them, and you re-render them by re-running the installer:

```bash
sudo bash /opt/financewatchdog/scripts/install.sh   # idempotent; re-renders units
```

## Operating it

```bash
systemctl list-timers 'financewatchdog-*'          # next run of each timer
journalctl -u financewatchdog-research -f          # watch the research loop
journalctl -u financewatchdog-trade --since today  # today's trade runs
sudo -u financewatchdog /opt/financewatchdog/.venv/bin/investment-robo check-safety \
  --config /opt/financewatchdog/config             # confirm cash-only account
```

Config (`.env`, `config/robo.yaml`) is git-ignored and survives updates. Point the DB at
a USB SSD instead of the SD card by setting `DATA_DIR`/`DB_PATH` in `.env`.

## The dashboard

`financewatchdog-dashboard.service` serves Archie's web GUI to your local network at
`http://<pi>.local:8321` (host/port via `DASHBOARD_HOST`/`DASHBOARD_PORT` in `.env`).
Pages: Overview (letter + equity curve), Theses (full narratives + monitoring),
Ledger (orders with gate verdicts, realized P&L), Charts (candles with Archie's fills),
Signals (confluence + insiders), The Study (calibration), System, and Settings.

Reads are open on the LAN. **Mutations** — pause/resume, the web kill switch,
blocklist edits, and settings writes — require the `DASHBOARD_TOKEN` PIN from `.env`
(the installer generates one and prints it once). Two invariants hold regardless of
the PIN: the web layer can force paper mode but can never arm live trading, and
setting `dry_run: false` is refused outright — going live stays a console act.
The dashboard opens the SQLite store read-only (`PRAGMA query_only`); every write it
can make goes through the same `control`/`blocklist`/`tunables` modules the CLI uses.
