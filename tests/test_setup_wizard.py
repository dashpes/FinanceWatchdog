"""Tests for first-run setup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from investment_monitor.config import Settings
from investment_monitor.setup_wizard import bootstrap_config, run_setup


def _write_examples(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "portfolio.yaml.example").write_text("holdings: []\n")
    (config_dir / "alerts.yaml.example").write_text("price:\n  enabled: true\n")


class TestBootstrapConfig:
    def test_creates_yaml_from_examples(self, tmp_path):
        cfg = tmp_path / "config"
        _write_examples(cfg)
        env_example = tmp_path / ".env.example"
        env_example.write_text("OLLAMA_MODEL=auto\n")
        env_target = tmp_path / ".env"

        actions = bootstrap_config(cfg, env_example, env_target)

        assert (cfg / "portfolio.yaml").exists()
        assert (cfg / "alerts.yaml").exists()
        assert env_target.read_text() == "OLLAMA_MODEL=auto\n"
        created = {a.name for a in actions if a.status == "created"}
        assert {"portfolio.yaml", "alerts.yaml", ".env"} <= created

    def test_does_not_overwrite_existing(self, tmp_path):
        cfg = tmp_path / "config"
        _write_examples(cfg)
        (cfg / "portfolio.yaml").write_text("holdings:\n  - ticker: NVDA\n")
        env_example = tmp_path / ".env.example"
        env_example.write_text("X=1\n")
        env_target = tmp_path / ".env"

        actions = bootstrap_config(cfg, env_example, env_target)

        # User's file is preserved.
        assert "NVDA" in (cfg / "portfolio.yaml").read_text()
        statuses = {a.name: a.status for a in actions}
        assert statuses["portfolio.yaml"] == "exists"

    def test_force_overwrites(self, tmp_path):
        cfg = tmp_path / "config"
        _write_examples(cfg)
        (cfg / "portfolio.yaml").write_text("holdings:\n  - ticker: NVDA\n")
        env_example = tmp_path / ".env.example"
        env_example.write_text("X=1\n")

        bootstrap_config(cfg, env_example, tmp_path / ".env", force=True)

        assert "NVDA" not in (cfg / "portfolio.yaml").read_text()

    def test_fallback_portfolio_when_no_example(self, tmp_path):
        cfg = tmp_path / "config"
        cfg.mkdir()
        actions = bootstrap_config(cfg, tmp_path / ".env.example", tmp_path / ".env")

        assert (cfg / "portfolio.yaml").exists()
        assert "ticker" in (cfg / "portfolio.yaml").read_text()
        env_action = next(a for a in actions if a.name == ".env")
        assert env_action.status == "skipped"  # no .env.example present


class TestRunSetup:
    def test_run_setup_succeeds_when_ollama_down(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)  # .env written relative to cwd
        cfg = tmp_path / "config"
        _write_examples(cfg)
        settings = Settings(ollama_model="qwen2.5:7b", ollama_synthesis_model="qwen2.5:32b")

        with patch(
            "investment_monitor.setup_wizard._probe_ollama",
            return_value=(False, [], "refused"),
        ):
            rc = run_setup(settings, assume_yes=False, config_dir=cfg)

        assert rc == 0
        assert (cfg / "portfolio.yaml").exists()
        out = capsys.readouterr().out
        assert "Ollama not reachable" in out
        assert "ollama pull qwen2.5:7b" in out

    def test_run_setup_pulls_missing_models_when_yes(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "config"
        _write_examples(cfg)
        settings = Settings(ollama_model="qwen2.5:7b", ollama_synthesis_model="qwen2.5:7b")

        with patch(
            "investment_monitor.setup_wizard._probe_ollama",
            return_value=(True, [], None),
        ), patch(
            "investment_monitor.setup_wizard._pull_model", return_value=True
        ) as mock_pull:
            rc = run_setup(settings, assume_yes=True, config_dir=cfg)

        assert rc == 0
        mock_pull.assert_called_once_with("qwen2.5:7b")
