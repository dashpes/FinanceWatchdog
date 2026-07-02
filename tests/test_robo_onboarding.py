"""Tests for the onboarding wizard's pure text helpers and the `init`/`prune` CLI."""

from __future__ import annotations

import stat

from typer.testing import CliRunner

from investment_monitor.robo import cli
from investment_monitor.robo.onboarding import parse_env, set_yaml_scalar, upsert_env


# --------------------------------------------------------------------------- #
# parse_env
# --------------------------------------------------------------------------- #
def test_parse_env_reads_values_and_ignores_comments_and_blanks():
    env = "# a comment\nA=1\n\nB=two words\n# C=commented-out\n"
    parsed = parse_env(env)
    assert parsed == {"A": "1", "B": "two words"}
    assert "C" not in parsed  # a commented-out key is documentation, not a value


# --------------------------------------------------------------------------- #
# upsert_env
# --------------------------------------------------------------------------- #
def test_upsert_env_updates_in_place_preserving_everything_else():
    env = "# header\nA=1\nB=2\n"
    out = upsert_env(env, {"A": "9"})
    assert out == "# header\nA=9\nB=2\n"  # comment + B untouched, trailing newline kept


def test_upsert_env_appends_missing_keys_after_a_blank():
    assert upsert_env("A=1\n", {"B": "2"}) == "A=1\n\nB=2\n"


def test_upsert_env_leaves_commented_example_line_and_appends_real_key():
    # A `# KEY=` documentation line must not be treated as the assignment.
    out = upsert_env("# TOKEN=your-token-here\n", {"TOKEN": "real"})
    assert "# TOKEN=your-token-here" in out
    assert parse_env(out)["TOKEN"] == "real"


# --------------------------------------------------------------------------- #
# set_yaml_scalar
# --------------------------------------------------------------------------- #
def test_set_yaml_scalar_replaces_and_keeps_trailing_comment():
    src = "dry_run: false  # hard kill-switch\nmode: autonomous\n"
    out = set_yaml_scalar(src, "dry_run", "true")
    assert out == "dry_run: true  # hard kill-switch\nmode: autonomous\n"


def test_set_yaml_scalar_appends_when_absent():
    assert set_yaml_scalar("mode: x\n", "account_id", '"5OL21018"') == 'mode: x\naccount_id: "5OL21018"\n'


def test_set_yaml_scalar_replaces_quoted_value():
    assert set_yaml_scalar('account_id: ""\n', "account_id", '"5OL"') == 'account_id: "5OL"\n'


# --------------------------------------------------------------------------- #
# `investment-robo init` — scaffolds config, forces dry-run, writes .env at 0600
# --------------------------------------------------------------------------- #
def test_init_non_interactive_scaffolds_and_forces_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("FW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text(
        "PUBLIC_API_TOKEN=\nSMTP_HOST=\nROBO_FORCE_DRY_RUN=true\n"
    )
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "robo.yaml.example").write_text('mode: advisory\ndry_run: false\naccount_id: ""\n')

    result = CliRunner().invoke(cli.app, ["init", "--non-interactive", "--config", str(cfg)])
    assert result.exit_code == 0, result.output

    env_path = tmp_path / ".env"
    assert env_path.exists()
    # Secrets file must be born owner-only.
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert parse_env(env_path.read_text())["ROBO_FORCE_DRY_RUN"] == "true"

    # robo.yaml scaffolded with dry_run forced on (never armed at init).
    assert "dry_run: true" in (cfg / "robo.yaml").read_text()


def test_init_is_idempotent_and_keeps_existing_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("FW_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("PUBLIC_API_TOKEN=\nROBO_FORCE_DRY_RUN=true\n")
    (tmp_path / ".env").write_text("PUBLIC_API_TOKEN=keepme\nROBO_FORCE_DRY_RUN=false\n")
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "robo.yaml.example").write_text('dry_run: false\naccount_id: ""\n')

    result = CliRunner().invoke(cli.app, ["init", "--non-interactive", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    parsed = parse_env((tmp_path / ".env").read_text())
    assert parsed["PUBLIC_API_TOKEN"] == "keepme"       # existing secret preserved
    assert parsed["ROBO_FORCE_DRY_RUN"] == "true"        # re-forced to safe


def test_init_writes_env_to_fw_home_not_cwd(tmp_path, monkeypatch):
    # Regression: the installer runs the wizard via `sudo -u` which does NOT change
    # directory, so init must anchor .env to $FW_HOME (where the services read it),
    # not the caller's CWD. Run from an unrelated CWD with FW_HOME set elsewhere.
    fw_home = tmp_path / "opt" / "financewatchdog"
    fw_home.mkdir(parents=True)
    (fw_home / ".env.example").write_text("PUBLIC_API_TOKEN=\nROBO_FORCE_DRY_RUN=true\n")
    cfg = fw_home / "config"
    cfg.mkdir()
    (cfg / "robo.yaml.example").write_text('dry_run: false\naccount_id: ""\n')
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("FW_HOME", str(fw_home))

    result = CliRunner().invoke(cli.app, ["init", "--non-interactive", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert (fw_home / ".env").exists()          # landed in FW_HOME
    assert not (cwd / ".env").exists()          # NOT in the caller's CWD
    assert stat.S_IMODE((fw_home / ".env").stat().st_mode) == 0o600


# --------------------------------------------------------------------------- #
# `investment-robo prune`
# --------------------------------------------------------------------------- #
def test_prune_reports_disabled_when_all_windows_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("INSIDER", "NEWS", "PRICE", "FINDINGS", "EVENTS"):
        monkeypatch.setenv(f"RETENTION_{k}_DAYS", "0")
    result = CliRunner().invoke(cli.app, ["prune"])
    assert result.exit_code == 0, result.output
    assert "Retention disabled" in result.output
