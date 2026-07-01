"""Pure text helpers for the ``investment-robo init`` onboarding wizard.

Kept free of I/O and prompting so the ``.env`` / ``robo.yaml`` edits are fully
unit-testable. The wizard (``cli.py:init``) handles prompting and the actual file
writes (at 0600 for secrets); these functions only transform text and never touch
disk. This is what lets a one-line installer scaffold a new machine's config
deterministically.
"""

from __future__ import annotations

import re

_KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines into a dict, ignoring blanks and ``#`` comments.

    A commented ``# KEY=...`` line is NOT parsed (it is documentation, not a value),
    so ``parse_env`` reflects only the keys actually set.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if _IDENT.match(key):
            out[key] = value.strip()
    return out


def upsert_env(text: str, updates: dict[str, str]) -> str:
    """Return ``.env`` text with each ``KEY`` set to its new value.

    An existing uncommented ``KEY=...`` line is updated in place — comments, ordering,
    and every other key are preserved. Keys not already present are appended at the
    end (after a blank separator). Commented ``# KEY=`` example lines are left intact
    as documentation; the real assignment is appended below them. Values are written
    verbatim (callers pass already-safe values — API tokens contain no newlines).
    """
    lines = text.splitlines()
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        m = _KEY_LINE.match(line)
        if m and m.group(2) in remaining:
            key = m.group(2)
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    if remaining:
        if out and out[-1].strip() != "":
            out.append("")
        for key, val in remaining.items():
            out.append(f"{key}={val}")
    result = "\n".join(out)
    if text.endswith("\n") or not text:
        result += "\n"
    return result


def set_yaml_scalar(text: str, key: str, value: str) -> str:
    """Return YAML text with the top-level scalar ``key:`` set to ``value``.

    Replaces the first column-0 ``key: ...`` line, preserving any trailing ``# comment``.
    If the key is absent it is appended. Only top-level keys are handled, which is all
    the wizard needs (``account_id`` / ``dry_run`` / ``mode`` in robo.yaml). ``value``
    is written as-is, so the caller formats it (``true``, or a quoted ``"5OL21018"``).
    """
    lines = text.splitlines()
    pat = re.compile(rf"^{re.escape(key)}\s*:(?P<rest>.*)$")
    for i, line in enumerate(lines):
        if pat.match(line):
            comment = ""
            hash_idx = line.find("#")
            if hash_idx != -1:
                comment = "  " + line[hash_idx:]
            lines[i] = f"{key}: {value}{comment}"
            break
    else:
        lines.append(f"{key}: {value}")
    result = "\n".join(lines)
    if text.endswith("\n") or not text:
        result += "\n"
    return result
