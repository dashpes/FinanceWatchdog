"""Shared helpers for the modern, object-based Ollama Python client.

As of ollama-python >= 0.4, the client returns typed response objects rather
than plain dicts: ``client.list()`` yields a ``ListResponse`` whose ``.models``
is a sequence of objects each exposing ``.model`` (the tag); ``client.generate()``
yields a ``GenerateResponse`` exposing ``.response`` (the text). These objects
still support dict-style access for backwards compatibility, but attribute access
is the forward-looking idiom.

Centralizing the extraction here means there is exactly one place that knows the
shape of an Ollama response, shared by LocalLLM and ResearchScorer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def model_names(list_response: Any) -> list[str]:
    """Extract the available model tags from a ``client.list()`` response.

    Args:
        list_response: The ``ListResponse`` returned by ``client.list()``.

    Returns:
        The list of model tags (e.g. ``["phi3:mini", "qwen2.5:7b"]``).
    """
    models: Sequence[Any] = getattr(list_response, "models", None) or []
    names: list[str] = []
    for m in models:
        name = getattr(m, "model", None)
        if name:
            names.append(name)
    return names


def model_matches(names: list[str], wanted: str) -> bool:
    """Return True if ``wanted`` is satisfied by any tag in ``names``.

    Matching rules (tag-aware, so different sizes of the same family do NOT
    match each other - ``qwen2.5:7b`` does not satisfy ``qwen2.5:32b``):

    - exact tag match (``"qwen2.5:7b"`` == ``"qwen2.5:7b"``);
    - if the caller did not pin a tag (``"qwen2.5"``), any tag of that base name;
    - if the caller asked for ``":latest"``, a bare install of that base name.

    Args:
        names: Installed model tags (e.g. from :func:`model_names`).
        wanted: The model the caller wants to use.

    Returns:
        True if a matching model is present.
    """
    if wanted in names:
        return True
    if ":" not in wanted:
        # No tag pinned: any size/tag of this base name counts.
        return any(name.split(":")[0] == wanted for name in names)
    base, tag = wanted.split(":", 1)
    if tag == "latest":
        # "name:latest" is satisfied by a bare "name" install.
        return base in names
    return False


def has_model(list_response: Any, wanted: str) -> bool:
    """Return True if ``wanted`` is present among a ``client.list()`` response.

    Args:
        list_response: The ``ListResponse`` returned by ``client.list()``.
        wanted: The model the caller wants to use.

    Returns:
        True if a matching model is installed.
    """
    return model_matches(model_names(list_response), wanted)


def response_text(generate_response: Any) -> str:
    """Extract the generated text from a ``client.generate()`` response.

    Args:
        generate_response: The ``GenerateResponse`` returned by ``generate()``.

    Returns:
        The stripped response text, or an empty string if absent.
    """
    text = getattr(generate_response, "response", None)
    return (text or "").strip()
