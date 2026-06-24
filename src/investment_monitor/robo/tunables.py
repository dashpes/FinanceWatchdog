"""User-tunable settings: one catalog, many front-ends.

A single source of truth for the robo's adjustable knobs, derived from the Pydantic
``RoboConfig`` schema. Any field tagged with ``json_schema_extra={"x_ui": ...}`` is
"tunable"; this module flattens those (including nested ``caps.*`` / ``sizing.*``)
into a dotted-key catalog with type, default, hard bounds, enum choices, and UI hints.

A CLI (``investment-robo config``) renders from this catalog today; a GUI later
renders the same catalog. ``set_value`` validates input against the schema AND the
full model before writing, and writes in place so the hand-authored comments in
``robo.yaml`` survive — so the contract never forks across front-ends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from investment_monitor.robo.config import RoboConfig


@dataclass(frozen=True)
class Tunable:
    """One adjustable setting, front-end agnostic."""

    key: str            # dotted path, e.g. "caps.max_positions"
    title: str
    description: str
    type: str           # integer | number | boolean | enum | string
    default: Any
    minimum: float | None
    maximum: float | None
    minimum_exclusive: bool  # True if minimum came from exclusiveMinimum (strict >)
    maximum_exclusive: bool  # True if maximum came from exclusiveMaximum (strict <)
    choices: list[str] | None
    group: str
    control: str | None  # UI hint: slider | stepper | toggle | select
    step: float | None
    unit: str | None

    def as_dict(self) -> dict:
        return asdict(self)


# --- schema introspection ---------------------------------------------------------

def _deref(node: dict, defs: dict) -> dict:
    """Resolve a single ``$ref`` / ``allOf``-with-one-``$ref``, merging the referenced
    definition *under* the node's own keys (so the property's default/title/x_ui win)."""
    merged = dict(node)
    ref = None
    if "$ref" in merged:
        ref = merged.pop("$ref")
    elif "allOf" in merged and len(merged["allOf"]) == 1 and "$ref" in merged["allOf"][0]:
        ref = merged["allOf"][0]["$ref"]
        merged.pop("allOf")
    if ref is None:
        return merged
    target = dict(defs.get(ref.rsplit("/", 1)[-1], {}))
    target.update(merged)
    return target


def _entry(key: str, node: dict) -> Tunable:
    choices = node.get("enum")
    typ = node.get("type")
    ui_type = "enum" if choices else (typ if typ in ("integer", "number", "boolean", "string") else (typ or "string"))
    xui = node.get("x_ui") or {}
    # Prefer inclusive bounds; fall back to exclusive ones, tracking which kind so
    # coerce() can use strict comparison for exclusive bounds (gt/lt fields).
    has_incl_min = "minimum" in node
    has_incl_max = "maximum" in node
    minimum = node.get("minimum", node.get("exclusiveMinimum"))
    maximum = node.get("maximum", node.get("exclusiveMaximum"))
    return Tunable(
        key=key,
        title=node.get("title", key),
        description=node.get("description", ""),
        type=ui_type,
        default=node.get("default"),
        minimum=minimum,
        maximum=maximum,
        minimum_exclusive=(minimum is not None and not has_incl_min),
        maximum_exclusive=(maximum is not None and not has_incl_max),
        choices=list(choices) if choices else None,
        group=xui.get("group", "General"),
        control=xui.get("control"),
        step=xui.get("step"),
        unit=xui.get("unit"),
    )


def _walk(props: dict, defs: dict, prefix: str = "") -> list[Tunable]:
    out: list[Tunable] = []
    for name, raw in props.items():
        node = _deref(raw, defs)
        if "properties" in node:                 # nested model -> recurse
            out.extend(_walk(node["properties"], defs, prefix=f"{prefix}{name}."))
        elif "x_ui" in node:                      # tunable leaf
            out.append(_entry(f"{prefix}{name}", node))
    return out


def catalog() -> list[Tunable]:
    """All tunable settings, flattened to dotted keys (ordered by group, then key)."""
    schema = RoboConfig.model_json_schema()
    items = _walk(schema.get("properties", {}), schema.get("$defs", {}))
    return sorted(items, key=lambda t: (t.group, t.key))


def _by_key() -> dict[str, Tunable]:
    return {t.key: t for t in catalog()}


# --- read / validate / write ------------------------------------------------------

def get_value(config: RoboConfig, key: str) -> Any:
    """Current value at a dotted key (an enum is returned as its plain string value)."""
    obj: Any = config
    for seg in key.split("."):
        obj = getattr(obj, seg)
    return obj.value if isinstance(obj, Enum) else obj


def coerce(key: str, raw: str) -> Any:
    """Coerce a CLI string to the tunable's type and check type + bounds + choices.

    Raises ``ValueError`` with a clear message on bad input. This is a friendly
    pre-check; ``set_value`` additionally re-validates the whole model (catching
    exclusive bounds and cross-field rules exactly).
    """
    t = _by_key().get(key)
    if t is None:
        raise ValueError(f"unknown setting '{key}' — run `config list` to see options")
    raw = raw.strip()
    if t.type == "boolean":
        low = raw.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{key}: expected true/false, got '{raw}'")
    if t.type == "enum":
        if raw not in (t.choices or []):
            raise ValueError(f"{key}: must be one of {t.choices}, got '{raw}'")
        return raw
    if t.type == "integer":
        try:
            v: Any = int(raw)
        except ValueError:
            raise ValueError(f"{key}: expected an integer, got '{raw}'") from None
    elif t.type == "number":
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"{key}: expected a number, got '{raw}'") from None
    else:
        return raw
    if t.minimum is not None:
        if t.minimum_exclusive and v <= t.minimum:
            raise ValueError(f"{key}: {v} must be greater than {t.minimum}")
        if not t.minimum_exclusive and v < t.minimum:
            raise ValueError(f"{key}: {v} is below the minimum {t.minimum}")
    if t.maximum is not None:
        if t.maximum_exclusive and v >= t.maximum:
            raise ValueError(f"{key}: {v} must be less than {t.maximum}")
        if not t.maximum_exclusive and v > t.maximum:
            raise ValueError(f"{key}: {v} is above the maximum {t.maximum}")
    return v


def set_value(config_path: Path, key: str, raw: str) -> Any:
    """Validate ``raw`` for ``key``, then write it into the YAML preserving comments.

    Returns the coerced value. Raises ``ValueError`` (leaving the file untouched) if
    the value is the wrong type/out of bounds, or if the resulting config is invalid.
    """
    value = coerce(key, raw)
    data: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    _apply(data, key, value)
    try:
        RoboConfig(**data)  # full cross-field validation
    except Exception as exc:  # noqa: BLE001 - surface a clean message, don't write
        raise ValueError(f"setting {key}={value} would make the config invalid: {exc}") from exc
    _write_in_place(config_path, key, value)
    return value


def _apply(data: dict, key: str, value: Any) -> None:
    segs = key.split(".")
    d = data
    for seg in segs[:-1]:
        nxt = d.setdefault(seg, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot set {key}: '{seg}' is not a mapping")
        d = nxt
    d[segs[-1]] = value


# --- comment-preserving YAML writer (pyyaml can't round-trip comments) ------------

def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _key_of(stripped: str) -> str:
    return stripped.split(":", 1)[0].strip()


def _replace_value(line: str, key: str, val: str) -> str:
    """Rewrite the scalar value on a ``key: value  # comment`` line, keeping the
    indentation and any inline comment (our scalar values never contain '#')."""
    indent = line[: len(line) - len(line.lstrip())]
    newline = "\n" if line.endswith("\n") else ""
    body = line[len(indent):].rstrip("\n")
    hpos = body.find("#")
    comment = ("  " + body[hpos:]) if hpos != -1 else ""
    return f"{indent}{key}: {val}{comment}{newline}"


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _write_in_place(path: Path, key: str, value: Any) -> None:
    segs = key.split(".")
    val = _fmt(value)
    lines = path.read_text().splitlines(keepends=True) if path.exists() else []
    if len(segs) == 1:
        lines = _set_top_level(lines, segs[0], val)
    elif len(segs) == 2:
        lines = _set_nested(lines, segs[0], segs[1], val)
    else:
        raise ValueError(f"unsupported nesting depth for '{key}'")
    path.write_text("".join(lines))


def _set_top_level(lines: list[str], key: str, val: str) -> list[str]:
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if _indent_of(line) == 0 and _key_of(stripped) == key:
            lines[i] = _replace_value(line, key, val)
            return lines
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"{key}: {val}\n")
    return lines


def _set_nested(lines: list[str], parent: str, child: str, val: str) -> list[str]:
    pi = next(
        (i for i, ln in enumerate(lines)
         if _indent_of(ln) == 0 and _key_of(ln.lstrip()) == parent
         and ln.lstrip().rstrip("\n").endswith(":")),
        None,
    )
    if pi is None:  # no such block — append a fresh one
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{parent}:\n  {child}: {val}\n")
        return lines
    block_indent: int | None = None
    insert_at = pi + 1
    j = pi + 1
    while j < len(lines):
        line = lines[j]
        stripped = line.lstrip()
        if stripped.strip() == "" or stripped.startswith("#"):
            j += 1
            insert_at = j
            continue
        if _indent_of(line) == 0:  # next top-level key ends the block
            break
        if block_indent is None:
            block_indent = _indent_of(line)
        if _indent_of(line) == block_indent and _key_of(stripped) == child:
            lines[j] = _replace_value(line, child, val)
            return lines
        insert_at = j + 1
        j += 1
    indent = " " * (block_indent or 2)
    lines.insert(insert_at, f"{indent}{child}: {val}\n")
    return lines
