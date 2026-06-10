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


def has_model(list_response: Any, wanted: str) -> bool:
    """Return True if ``wanted`` is present among the available models.

    Matches either the exact tag (``"phi3:mini"``) or any tag sharing the same
    base name (``"phi3:mini"`` also satisfies a request for ``"phi3"``), so a
    pinned ``:latest`` or differing tag still counts as available.

    Args:
        list_response: The ``ListResponse`` returned by ``client.list()``.
        wanted: The model the caller wants to use.

    Returns:
        True if a matching model is installed.
    """
    base = wanted.split(":")[0]
    for name in model_names(list_response):
        if name == wanted or name.startswith(base):
            return True
    return False


def response_text(generate_response: Any) -> str:
    """Extract the generated text from a ``client.generate()`` response.

    Args:
        generate_response: The ``GenerateResponse`` returned by ``generate()``.

    Returns:
        The stripped response text, or an empty string if absent.
    """
    text = getattr(generate_response, "response", None)
    return (text or "").strip()
