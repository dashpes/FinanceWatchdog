"""Tests for the tunable-settings catalog, validation, and YAML writer."""

from __future__ import annotations

import pytest

from investment_monitor.robo import tunables
from investment_monitor.robo.config import ConfigError, RoboConfig

SAMPLE = """\
# Robo advisor config (annotated)
mode: autonomous            # conviction-driven; target_allocation ignored

# Rate / size caps enforced by the gate.
caps:
  max_order_pct: 0.5        # up to 50% per order so positions can be built
  max_positions: 4          # small account: at most 4 names
  max_per_name_weight: 0.35 # no single name above 35%

dry_run: false              # LIVE
"""


def _write(tmp_path, text=SAMPLE):
    p = tmp_path / "robo.yaml"
    p.write_text(text)
    return p


# --- catalog ----------------------------------------------------------------------

def test_catalog_includes_expected_keys():
    keys = {t.key for t in tunables.catalog()}
    assert {"caps.max_positions", "caps.max_order_pct", "mode", "dry_run",
            "sizing.kelly_fraction", "sizing.max_position_weight"} <= keys


def test_catalog_entry_metadata():
    by = {t.key: t for t in tunables.catalog()}
    mp = by["caps.max_positions"]
    assert mp.type == "integer" and mp.group == "Risk" and mp.control == "stepper"
    mode = by["mode"]
    assert mode.type == "enum" and mode.choices == ["rebalance", "autonomous"]
    assert by["dry_run"].type == "boolean"


def test_catalog_sorted_by_group_then_key():
    cat = tunables.catalog()
    assert cat == sorted(cat, key=lambda t: (t.group, t.key))


# --- get_value ---------------------------------------------------------------------

def test_get_value_reads_nested_and_enum_as_string():
    cfg = RoboConfig(mode="autonomous", caps={"max_positions": 7})
    assert tunables.get_value(cfg, "caps.max_positions") == 7
    assert tunables.get_value(cfg, "mode") == "autonomous"  # enum -> plain string


# --- coerce ------------------------------------------------------------------------

def test_coerce_integer_ok_and_bounds():
    assert tunables.coerce("caps.max_positions", "8") == 8
    with pytest.raises(ValueError):
        tunables.coerce("caps.max_positions", "-1")     # below minimum 0
    with pytest.raises(ValueError):
        tunables.coerce("caps.max_positions", "999")    # above hard maximum 50
    with pytest.raises(ValueError):
        tunables.coerce("caps.max_positions", "1.5")    # not an integer


def test_coerce_boolean_variants():
    assert tunables.coerce("dry_run", "true") is True
    assert tunables.coerce("dry_run", "off") is False
    with pytest.raises(ValueError):
        tunables.coerce("dry_run", "maybe")


def test_coerce_enum():
    assert tunables.coerce("mode", "rebalance") == "rebalance"
    with pytest.raises(ValueError):
        tunables.coerce("mode", "yolo")


def test_coerce_unknown_key():
    with pytest.raises(ValueError):
        tunables.coerce("caps.does_not_exist", "1")


def test_coerce_rejects_value_equal_to_exclusive_minimum():
    # caps.max_order_pct is gt=0 -> 0 must be rejected by the standalone pre-check
    with pytest.raises(ValueError):
        tunables.coerce("caps.max_order_pct", "0")
    with pytest.raises(ValueError):
        tunables.coerce("sizing.kelly_fraction", "0")        # gt=0
    with pytest.raises(ValueError):
        tunables.coerce("sizing.max_position_weight", "0")   # gt=0
    # a value just above the exclusive bound is fine
    assert tunables.coerce("caps.max_order_pct", "0.01") == pytest.approx(0.01)


def test_coerce_allows_value_equal_to_inclusive_minimum():
    # caps.max_positions is ge=0 -> 0 is a valid inclusive boundary value
    assert tunables.coerce("caps.max_positions", "0") == 0


def test_coerce_exclusive_bound_metadata_flags_set():
    by = {t.key: t for t in tunables.catalog()}
    assert by["caps.max_order_pct"].minimum_exclusive is True   # gt=0
    assert by["caps.max_order_pct"].maximum_exclusive is False  # le=1.0 (inclusive)
    assert by["caps.max_positions"].minimum_exclusive is False  # ge=0 (inclusive)


# --- set_value: writes, preserves comments, validates -----------------------------

def test_set_nested_preserves_comments_and_changes_value(tmp_path):
    p = _write(tmp_path)
    tunables.set_value(p, "caps.max_positions", "8")
    text = p.read_text()
    assert "max_positions: 8" in text
    assert "# small account: at most 4 names" in text      # inline comment kept
    assert "max_order_pct: 0.5" in text                    # sibling untouched
    assert "# conviction-driven" in text                   # unrelated comment kept
    assert RoboConfig.from_yaml(p).caps.max_positions == 8  # reloads correctly


def test_set_top_level_enum(tmp_path):
    p = _write(tmp_path)
    tunables.set_value(p, "mode", "rebalance")
    assert "mode: rebalance" in p.read_text()
    assert "# conviction-driven" in p.read_text()          # comment preserved
    assert RoboConfig.from_yaml(p).mode == "rebalance"


def test_set_top_level_bool(tmp_path):
    p = _write(tmp_path)
    tunables.set_value(p, "dry_run", "true")
    assert "dry_run: true" in p.read_text()
    assert RoboConfig.from_yaml(p).dry_run is True


def test_set_rejects_out_of_bounds_without_writing(tmp_path):
    p = _write(tmp_path)
    before = p.read_text()
    with pytest.raises(ValueError):
        tunables.set_value(p, "caps.max_per_name_weight", "2.0")  # le=1.0
    assert p.read_text() == before  # file untouched on rejection


def test_set_rejects_above_hard_max_without_writing(tmp_path):
    p = _write(tmp_path)
    before = p.read_text()
    with pytest.raises(ValueError):
        tunables.set_value(p, "caps.max_positions", "999")  # hard le=50
    assert p.read_text() == before


def test_set_inserts_absent_nested_key(tmp_path):
    p = _write(tmp_path)
    tunables.set_value(p, "caps.max_drawdown_pct", "25")  # not in the sample block
    assert RoboConfig.from_yaml(p).caps.max_drawdown_pct == 25
    assert "max_positions: 4" in p.read_text()            # existing lines intact


def test_set_inserts_absent_top_level_key(tmp_path):
    p = _write(tmp_path)
    tunables.set_value(p, "rebalance_threshold", "0.1")
    assert RoboConfig.from_yaml(p).rebalance_threshold == pytest.approx(0.1)


# --- from_yaml: a bad file must never escape as a raw traceback -------------------

def test_from_yaml_out_of_bounds_raises_clear_config_error(tmp_path):
    """A robo.yaml above a hard ceiling yields an actionable ConfigError that
    names the key, the violated bound, and the offending value — NOT a bare
    pydantic ValidationError that would crash a daemon with a traceback."""
    from pydantic import ValidationError

    # max_positions has a hard le=50; a previously-valid file with 99 must not
    # crash. (Regression: the new upper bounds used to surface as a raw
    # ValidationError through an unguarded _load_config / from_yaml.)
    p = tmp_path / "robo.yaml"
    p.write_text("caps:\n  max_positions: 99\n")

    with pytest.raises(ConfigError) as exc_info:
        RoboConfig.from_yaml(p)

    # A raw ValidationError must not be what escapes.
    assert not isinstance(exc_info.value, ValidationError)
    msg = str(exc_info.value)
    assert "caps.max_positions" in msg          # which key
    assert "99" in msg                          # the offending value
    assert "50" in msg or "less than or equal" in msg  # the violated bound
    assert str(p) in msg                        # which file


def test_from_yaml_other_out_of_bounds_caps_also_caught(tmp_path):
    """The other newly-ceilinged caps fields are guarded too."""
    for key, bad in (
        ("max_orders_per_day", 5000),     # le=1000
        ("max_turnover_pct", 99.0),       # le=10.0
        ("max_drawdown_pct", 250.0),      # le=100.0
    ):
        p = tmp_path / "robo.yaml"
        p.write_text(f"caps:\n  {key}: {bad}\n")
        with pytest.raises(ConfigError) as exc_info:
            RoboConfig.from_yaml(p)
        assert f"caps.{key}" in str(exc_info.value)


def test_from_yaml_malformed_yaml_raises_config_error(tmp_path):
    """An unparseable YAML file is surfaced as a ConfigError, not a raw
    yaml.YAMLError traceback."""
    p = tmp_path / "robo.yaml"
    p.write_text("caps: {max_positions: 4\n")  # missing closing brace
    with pytest.raises(ConfigError):
        RoboConfig.from_yaml(p)


def test_from_yaml_non_mapping_top_level_raises_config_error(tmp_path):
    """A YAML file whose top level is not a mapping is rejected clearly."""
    p = tmp_path / "robo.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        RoboConfig.from_yaml(p)


def test_from_yaml_valid_at_bound_still_loads(tmp_path):
    """The fix does not weaken the bounds: a value exactly at the ceiling is
    still accepted (guard catches only true violations)."""
    p = tmp_path / "robo.yaml"
    p.write_text("caps:\n  max_positions: 50\n")  # exactly le=50
    assert RoboConfig.from_yaml(p).caps.max_positions == 50


def test_load_config_surfaces_clean_error_not_traceback(tmp_path):
    """cli._load_config must exit non-zero with a clear message rather than let
    a ConfigError/traceback escape and silently halt the daemon."""
    import typer
    from typer.testing import CliRunner

    from investment_monitor.robo import cli

    (tmp_path / "robo.yaml").write_text("caps:\n  max_positions: 99\n")

    app = typer.Typer()

    @app.command()
    def boom() -> None:
        cli._load_config(tmp_path)

    result = CliRunner().invoke(app, [])
    assert result.exit_code == 1
    # The exception must have been handled (typer.Exit), not propagated raw.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Config error" in result.stderr
    assert "caps.max_positions" in result.stderr


def test_config_set_repairs_out_of_bounds_current_value(tmp_path):
    """`config set` must be able to REPAIR a robo.yaml whose CURRENT value for the
    target key is out of bounds.

    Regression: config_set loaded the current config (which exits on a ConfigError)
    only to display the OLD value, wrapped in `except AttributeError`. A typer.Exit
    from _load_config escaped and aborted the command BEFORE the write — so the very
    tool meant to repair config could not repair an out-of-bounds file. The write
    path (tunables.set_value) re-validates the full merged config, so a valid
    single-key repair must succeed even when the file currently holds a bad value.
    """
    from typer.testing import CliRunner

    from investment_monitor.robo import cli

    # max_positions has a hard le=50; 99 makes _load_config exit on a ConfigError.
    p = tmp_path / "robo.yaml"
    p.write_text("caps:\n  max_positions: 99  # out of bounds\n")

    result = CliRunner().invoke(
        cli.app, ["config", "set", "caps.max_positions", "8", "--config", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    # The repair was written and now reloads cleanly within bounds.
    assert RoboConfig.from_yaml(p).caps.max_positions == 8
    assert "max_positions: 8" in p.read_text()
    assert "# out of bounds" in p.read_text()  # inline comment preserved
    # Old value could not be read, so it is reported as unavailable (not crashed).
    assert "(unavailable)" in result.output


def test_config_set_still_blocks_invalid_write_when_sibling_bad(tmp_path):
    """The repair tolerance must NOT weaken the safety check: if the single-key
    write would still leave the merged config invalid (a different sibling field is
    out of bounds), tunables.set_value rejects it and nothing is written."""
    from typer.testing import CliRunner

    from investment_monitor.robo import cli

    # max_per_name_weight (le=1.0) is the bad sibling; we try to set a *different*
    # key to a value that is itself valid, but the merged config stays invalid.
    p = tmp_path / "robo.yaml"
    p.write_text("caps:\n  max_per_name_weight: 2.0\n  max_positions: 4\n")
    before = p.read_text()

    result = CliRunner().invoke(
        cli.app, ["config", "set", "caps.max_positions", "8", "--config", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert p.read_text() == before  # file untouched: invalid write blocked
    assert "Rejected" in result.output
