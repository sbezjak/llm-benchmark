"""Ollama `/api/embed` adapter - the free local embedding lane for the cache.

The semantic cache needs a vector per query; running that through the local
Ollama baseline keeps the whole experiment $0 (no `sentence_transformers`, no
paid embedding API). This is a thin sibling of `ollama.py`: same base_url, same
raw-httpx + respx-mockable discipline, and the request/response are logged at
INFO like every other model call so the report shows the vectors' provenance
(dimension + which model produced them, not the raw floats).

Not a `Provider` - it returns embeddings, not a `Measurement`. It sits behind
the same `providers/` seam only because it is the one place that issues the HTTP
call for embeddings, which is what the seam is about.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedError(RuntimeError):
    """Raised when the Ollama embed backend errors or returns an unexpected payload."""


class OllamaEmbedder:
    """Ollama `/api/embed` adapter.

    `/api/embed` (not the older `/api/embeddings`) takes `{"model", "input"}`
    where input is a string OR a list of strings, and returns
    `{"embeddings": [[...], ...]}` - one vector per input, in order. The default
    model is `nomic-embed-text`, the standard free Ollama embedding model; pass
    another via the constructor. The model must be pulled locally first
    (`ollama pull <model>`) - a missing model surfaces as a loud HTTP error.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch in one call - order preserved, one vector per input."""
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    raise OllamaEmbedError(f"{url} returned HTTP {resp.status_code}: {resp.text}")
                obj = resp.json()
            except httpx.HTTPError as e:
                raise OllamaEmbedError(f"request to {url} failed: {e}") from e
        embeddings = obj.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaEmbedError(f"expected {len(texts)} embeddings, got {embeddings!r}")
        dim = len(embeddings[0]) if embeddings and embeddings[0] else 0
        logger.info(
            "ollama_embed model=%s url=%s n=%d dim=%d\nINPUTS:\n%s",
            self.model,
            url,
            len(texts),
            dim,
            "\n".join(f"  {t}" for t in texts),
        )
        return embeddings

    async def embed(self, text: str) -> list[float]:
        """Embed one string - the single-query path the cache lookup uses."""
        return (await self.embed_many([text]))[0]
