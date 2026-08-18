from __future__ import annotations

import json
import logging
import time

import httpx

from llm_benchmark.providers.base import Measurement, Provider

logger = logging.getLogger(__name__)


class OpenAICompatError(RuntimeError):
    """Raised when an OpenAI-compatible backend returns an error or unexpected payload."""


class OpenAICompatProvider(Provider):
    """Chat Completions adapter for OpenAI-compatible backends.

    Streamed SSE: the endpoint replies with `data: {...}` lines, each chunk
    carrying a `choices[0].delta.content` text fragment. With
    `stream_options.include_usage` the final data chunk carries the provider's
    own `usage` object (`prompt_tokens` / `completion_tokens`) — the token
    counts come from the source's counter, never a re-tokenization here, same
    fair-cost discipline as the Ollama adapter. The stream ends with
    `data: [DONE]`. Streaming buys an honest time-to-first-token.

    Unlike Ollama there is no `/api/show` to reveal the server-side prompt
    wrapping, so the transparency analog here is logging the returned `usage`
    object verbatim — the report shows exactly where each number came from.

    DeepSeek's API is OpenAI-compatible: the same class serves it with only a
    `base_url` + key swap, no new parsing.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        temperature: float | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required (never hardcode it; read it from the env)")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # None omits the field entirely (the default): current-gen models
        # (gpt-5.6-luna confirmed by smoke test) REJECT non-default sampling
        # params with a 400, so the low-temperature reproducibility discipline
        # can't carry over here. Pass a value only for backends that support it.
        self.temperature = temperature

    async def measure(self, prompt: str) -> Measurement:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self._api_key}"}

        chunks: list[str] = []
        usage: dict = {}
        ttft_ms: float | None = None

        logger.info("openai_compat.measure model=%s url=%s\nPROMPT:\n%s", self.model, url, prompt)
        request_id = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            start = time.monotonic()
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        # Read the body before raising: the provider's error
                        # JSON says WHICH field was rejected - without it a 400
                        # is undebuggable.
                        detail = (await resp.aread()).decode(errors="replace")
                        raise OpenAICompatError(f"{url} returned HTTP {resp.status_code}: {detail}")
                    # The provider's own request id - a verifiable receipt for
                    # this paid call, cross-referable in the provider console.
                    request_id = resp.headers.get("x-request-id") or resp.headers.get(
                        "request-id", ""
                    )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        choices = obj.get("choices") or []
                        if choices:
                            fragment = (choices[0].get("delta") or {}).get("content") or ""
                            if fragment and ttft_ms is None:
                                ttft_ms = (time.monotonic() - start) * 1000.0
                            chunks.append(fragment)
                        if obj.get("usage"):
                            usage = obj["usage"]
            except httpx.HTTPError as e:
                raise OpenAICompatError(f"request to {url} failed: {e}") from e
            except json.JSONDecodeError as e:
                raise OpenAICompatError(f"{url} returned a non-JSON stream chunk: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000.0
        text = "".join(chunks)
        logger.info(
            "openai_compat.measure done model=%s request_id=%s latency_ms=%.1f ttft_ms=%s\n"
            "USAGE (verbatim from the provider - the source of every token count):\n%s\n"
            "RESPONSE:\n%s",
            self.model,
            request_id or "(none)",
            latency_ms,
            f"{ttft_ms:.1f}" if ttft_ms is not None else "None",
            json.dumps(usage) if usage else "(none returned)",
            text,
        )
        return Measurement(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model=self.model,
            request_id=request_id or None,
        )
