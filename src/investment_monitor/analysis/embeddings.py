"""Ollama embeddings client (Phase 5) for semantic memory.

Mirrors ``LocalLLM``'s lazy-client + is_available pattern. Used to embed theses and
news so the agent can recall similar past setups and skip near-duplicates. Degrades
gracefully: callers check ``is_available()`` and skip memory features when the
embedding model isn't installed.
"""

from __future__ import annotations

from loguru import logger


class EmbeddingsClient:
    """Thin wrapper over the Ollama embeddings endpoint."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None) -> None:
        self.model = model
        self._base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self._base_url) if self._base_url else ollama.Client()
        return self._client

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not available"
            return False

    def embed_one(self, text: str) -> list[float] | None:
        """Embed a single string, or None on failure."""
        try:
            resp = self.client.embeddings(model=self.model, prompt=text)
            vector = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            return list(vector) if vector else None
        except Exception as exc:  # noqa: BLE001 - embedding is best-effort
            logger.warning("embedding failed: {e}", e=exc)
            return None

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed_one(t) for t in texts]
