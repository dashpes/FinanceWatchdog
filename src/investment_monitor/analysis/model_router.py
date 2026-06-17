"""Config-driven model routing (Phase 5).

Maps a *role* (triage / synthesis / scoring / embedding) to a concrete Ollama
model name from ``Settings.model_roles``, falling back to ``Settings.ollama_model``.
This keeps model choices in config: a small/fast model handles high-volume triage,
a stronger model writes theses, without rewiring callers — they just ask the router
for the model name to construct a ``LocalLLM`` with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from investment_monitor.config import Settings


class ModelRouter:
    """Resolve a role -> model name (pure config lookup)."""

    def __init__(self, settings: "Settings") -> None:
        self._roles = dict(getattr(settings, "model_roles", {}) or {})
        self._default = settings.ollama_model

    def get_model(self, role: str) -> str:
        """Model name for ``role``; falls back to the default Ollama model."""
        return self._roles.get(role) or self._default

    def roles(self) -> dict[str, str]:
        return dict(self._roles)
