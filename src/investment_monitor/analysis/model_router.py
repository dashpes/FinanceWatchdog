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

    def resolve(
        self,
        role: str,
        *,
        installed: list[str] | None = None,
        base_url: str | None = None,
    ) -> str:
        """Resolve a role to a model that is ACTUALLY installed.

        Returns the configured model if present; else a same-family installed model
        (same name before the ``:`` tag, e.g. ``qwen2.5:7b`` -> ``qwen2.5:14b``); else
        the configured name unchanged (the caller's fallback handles a missing model).

        This avoids ``LocalLLM.is_available``'s prefix-match false-positive, which
        otherwise reports an uninstalled model as available and then 404s on generate.
        """
        preferred = self.get_model(role)
        if installed is None:
            installed = self._installed(base_url)
        if not installed:  # couldn't query Ollama — trust the configured name
            return preferred
        if preferred in installed:
            return preferred
        base = preferred.split(":")[0]
        family = [m for m in installed if m.split(":")[0] == base]
        return family[0] if family else preferred

    @staticmethod
    def _installed(base_url: str | None) -> list[str] | None:
        try:
            import ollama

            client = ollama.Client(host=base_url) if base_url else ollama.Client()
            return [m.model for m in client.list().models]
        except Exception:  # noqa: BLE001 - can't list -> caller trusts config
            return None
