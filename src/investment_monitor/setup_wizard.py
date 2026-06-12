"""First-run setup for new users.

`investment-monitor --setup` bootstraps everything a fresh checkout needs to run:
config files (from the bundled ``*.example`` templates), a ``.env``, and the
local Ollama models that fit this machine. It is safe to re-run - existing files
are never overwritten unless ``force`` is set.

This is intentionally non-interactive and idempotent so the curl|bash installer
can call it unattended (`--setup --yes`), while a human can run it directly to
see what is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .analysis.hardware import recommend_models, total_ram_gb
from .analysis.ollama_client import model_matches
from .config import Settings, get_settings
from .diagnostics import _probe_ollama

# Minimal fallback used only when no portfolio example is available (e.g. a bare
# pipx install with no repo checkout). Keeps `--setup` working everywhere.
_FALLBACK_PORTFOLIO = """\
# Add your holdings and watchlist here. See the README for all options.
holdings:
  - ticker: AAPL
    shares: 10
    cost_basis: 165.00
    thesis: "Example holding - replace with your own"

watchlist:
  - ticker: GOOGL
    reason: "Example watchlist entry"
    target_price: 140.00
"""


@dataclass
class SetupAction:
    """One bootstrapping action and its outcome (for reporting/testing)."""

    name: str
    status: str  # "created" | "exists" | "skipped"
    detail: str = ""


def bootstrap_config(
    config_dir: Path,
    env_example: Path,
    env_target: Path,
    *,
    force: bool = False,
) -> list[SetupAction]:
    """Create config files and ``.env`` from templates when missing.

    Args:
        config_dir: Directory holding ``*.yaml.example`` templates (and where the
            real ``*.yaml`` files are written).
        env_example: Path to ``.env.example``.
        env_target: Path to the ``.env`` to create.
        force: Overwrite existing files instead of leaving them untouched.

    Returns:
        A list of SetupAction describing what happened to each file.
    """
    actions: list[SetupAction] = []

    config_dir.mkdir(parents=True, exist_ok=True)

    examples = sorted(config_dir.glob("*.yaml.example"))
    for example in examples:
        target = example.with_suffix("")  # "portfolio.yaml.example" -> "portfolio.yaml"
        if target.exists() and not force:
            actions.append(SetupAction(target.name, "exists"))
        else:
            shutil.copyfile(example, target)
            actions.append(SetupAction(target.name, "created", f"from {example.name}"))

    # Guarantee a portfolio.yaml even when no example template is present.
    portfolio = config_dir / "portfolio.yaml"
    if not portfolio.exists():
        portfolio.write_text(_FALLBACK_PORTFOLIO)
        actions.append(SetupAction(portfolio.name, "created", "built-in template"))

    # .env
    if env_target.exists() and not force:
        actions.append(SetupAction(env_target.name, "exists"))
    elif env_example.exists():
        shutil.copyfile(env_example, env_target)
        actions.append(SetupAction(env_target.name, "created", f"from {env_example.name}"))
    else:
        actions.append(SetupAction(env_target.name, "skipped", "no .env.example found"))

    return actions


def _pull_model(model: str) -> bool:
    """Pull an Ollama model via the CLI. Returns True on success."""
    if shutil.which("ollama") is None:
        return False
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def run_setup(
    settings: Settings | None = None,
    *,
    assume_yes: bool = False,
    pull_models: bool = True,
    config_dir: Path | None = None,
) -> int:
    """Run first-run setup: bootstrap config, then check/pull local models.

    Args:
        settings: Application settings (defaults to loaded settings).
        assume_yes: Pull missing models without prompting (for unattended installs).
        pull_models: Whether to attempt pulling missing models at all.
        config_dir: Override the config directory (defaults to settings.config_dir).

    Returns:
        Process exit code (0 on success).
    """
    settings = settings or get_settings()
    cfg_dir = config_dir or settings.config_dir

    print("Investment Monitor - setup")
    print("=" * 44)

    # 1) Config files
    print("\nConfiguration files")
    actions = bootstrap_config(cfg_dir, Path(".env.example"), Path(".env"))
    for a in actions:
        mark = {"created": "+", "exists": "=", "skipped": "!"}.get(a.status, "?")
        detail = f"  ({a.detail})" if a.detail else ""
        print(f"  [{mark}] {a.name}: {a.status}{detail}")

    # 2) Local models
    fast = settings.resolved_ollama_model()
    synth = settings.resolved_synthesis_model()
    wanted = [fast] + ([synth] if synth != fast else [])
    ram = total_ram_gb()
    rec = recommend_models(ram)
    ram_str = f"{ram:.0f} GiB" if ram is not None else "unknown"
    print(f"\nLocal models (detected RAM: {ram_str}, tier: {rec.tier})")

    reachable, installed, err = _probe_ollama(settings.ollama_host)
    if not reachable:
        print(f"  [!] Ollama not reachable: {err}")
        print("      Install it from https://ollama.com/download, run `ollama serve`,")
        print("      then re-run `investment-monitor --setup`. Needed models:")
        for m in wanted:
            print(f"        ollama pull {m}")
    else:
        missing = [m for m in wanted if not model_matches(installed, m)]
        for m in wanted:
            present = model_matches(installed, m)
            print(f"  [{'=' if present else ' '}] {m}: {'installed' if present else 'missing'}")
        if missing and pull_models:
            if assume_yes:
                for m in missing:
                    print(f"  -> pulling {m} ...")
                    ok = _pull_model(m)
                    print(f"     {'done' if ok else 'FAILED (run: ollama pull ' + m + ')'}")
            else:
                print("\n  To install the missing models, run:")
                for m in missing:
                    print(f"        ollama pull {m}")
                print("  (or re-run with `investment-monitor --setup --yes` to pull automatically)")

    # 3) Next steps
    print("\nNext steps")
    print(f"  1. Edit your holdings:   {cfg_dir / 'portfolio.yaml'}")
    print("  2. (Optional) edit .env for notifications / Claude")
    print("  3. Verify setup:         investment-monitor --doctor")
    print("  4. Run it:               investment-monitor --type regular")
    return 0
