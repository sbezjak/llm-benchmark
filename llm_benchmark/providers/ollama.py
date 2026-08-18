from __future__ import annotations

import json
import logging
import time

import httpx

from llm_benchmark.providers.base import Measurement, Provider

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the Ollama backend returns an error or unexpected payload."""


class OllamaProvider(Provider):
    """Ollama `/api/generate` adapter.

    Streamed (`stream=True`): the endpoint replies with newline-delimited JSON,
    one object per token-ish chunk carrying a `response` fragment, and a final
    object with `done: true` plus the token counts (`prompt_eval_count`,
    `eval_count`). Streaming buys an honest time-to-first-token — the moment the
    first fragment arrives — which a single `stream=False` blob can't give.
    Sampling params pass through Ollama's `options` object; the low-temperature
    default favors eval reproducibility, but callers can override.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        # The model's chat template, fetched from /api/show on first use and
        # cached. Logging it makes tokens_in legible: the count includes this
        # wrapping (role headers, system framing, separators), not just the
        # visible prompt words. `None` = not fetched yet; `""` = unavailable.
        self._template: str | None = None

    async def _get_template(self, client: httpx.AsyncClient) -> str:
        """The model's chat template (the wrapping Ollama applies server-side
        before the model sees the prompt). Best-effort: a failure here must
        never fail a measurement, so on error we cache and log "unavailable".
        Structure only — `{{ .Content }}` placeholders, not a rendered string;
        rendering Ollama's Go template in Python would reimplement the server."""
        if self._template is not None:
            return self._template
        try:
            resp = await client.post(f"{self.base_url}/api/show", json={"model": self.model})
            resp.raise_for_status()
            self._template = resp.json().get("template", "") or ""
        except (httpx.HTTPError, json.JSONDecodeError):
            self._template = ""
        return self._template

    async def measure(self, prompt: str) -> Measurement:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": self.temperature},
        }

        chunks: list[str] = []
        tokens_in = 0
        tokens_out = 0
        ttft_ms: float | None = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            template = await self._get_template(client)
            logger.info(
                "ollama.measure model=%s url=%s\n"
                "SYSTEM TEMPLATE (wraps the prompt server-side; the extra "
                "tokens_in over the visible words are these):\n%s\nPROMPT:\n%s",
                self.model,
                url,
                template or "(unavailable)",
                prompt,
            )
            start = time.monotonic()
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        obj = json.loads(line)
                        fragment = obj.get("response", "")
                        if fragment and ttft_ms is None:
                            ttft_ms = (time.monotonic() - start) * 1000.0
                        chunks.append(fragment)
                        if obj.get("done"):
                            tokens_in = obj.get("prompt_eval_count", 0)
                            tokens_out = obj.get("eval_count", 0)
            except httpx.HTTPError as e:
                raise OllamaError(f"Ollama request failed: {e}") from e
            except json.JSONDecodeError as e:
                raise OllamaError(f"Ollama returned a non-JSON stream line: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000.0
        text = "".join(chunks)
        logger.info(
            "ollama.measure done model=%s tokens_in=%d tokens_out=%d "
            "latency_ms=%.1f ttft_ms=%s\nRESPONSE:\n%s",
            self.model,
            tokens_in,
            tokens_out,
            latency_ms,
            f"{ttft_ms:.1f}" if ttft_ms is not None else "None",
            text,
        )
        return Measurement(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model=self.model,
        )
