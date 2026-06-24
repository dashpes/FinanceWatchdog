"""Tests for the tunable-settings catalog, validation, and YAML writer."""

from __future__ import annotations

import pytest

from investment_monitor.robo import tunables
from investment_monitor.robo.config import RoboConfig

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
