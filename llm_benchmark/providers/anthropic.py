from __future__ import annotations

import json
import logging
import time

import httpx

from llm_benchmark.providers.base import Measurement, Provider

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicError(RuntimeError):
    """Raised when the Anthropic backend returns an error or unexpected payload."""


class AnthropicProvider(Provider):
    """Anthropic Messages API adapter (Haiku and Sonnet ride one class).

    Streamed SSE: `event:`/`data:` line pairs. The token counts come from the
    provider's own `usage` objects - `input_tokens` on the `message_start`
    event, `output_tokens` on the final `message_delta` - never a
    re-tokenization here. Text arrives as `content_block_delta` events with
    `text_delta` payloads; the first one is the honest time-to-first-token.
    Both usage objects are logged verbatim so the report shows where each
    number came from.

    `max_tokens` is required by this API (unlike Chat Completions), so it is a
    constructor param with a benchmark-sized default. Sampling params are
    omitted by default: current-gen models (claude-sonnet-5) reject non-default
    values with a 400, same constraint the OpenAI adapter hit.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required (never hardcode it; read it from the env)")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def measure(self, prompt: str) -> Measurement:
        url = f"{self.base_url}/v1/messages"
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}

        chunks: list[str] = []
        start_usage: dict = {}
        delta_usage: dict = {}
        ttft_ms: float | None = None

        logger.info("anthropic.measure model=%s url=%s\nPROMPT:\n%s", self.model, url, prompt)
        request_id = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            start = time.monotonic()
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode(errors="replace")
                        raise AnthropicError(f"{url} returned HTTP {resp.status_code}: {detail}")
                    # The provider's own request id - a verifiable receipt for
                    # this paid call, cross-referable in the Anthropic console.
                    request_id = resp.headers.get("request-id", "")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        obj = json.loads(line[len("data:") :].strip())
                        kind = obj.get("type")
                        if kind == "message_start":
                            start_usage = obj.get("message", {}).get("usage", {}) or {}
                        elif kind == "content_block_delta":
                            delta = obj.get("delta", {}) or {}
                            if delta.get("type") == "text_delta":
                                fragment = delta.get("text", "")
                                if fragment and ttft_ms is None:
                                    ttft_ms = (time.monotonic() - start) * 1000.0
                                chunks.append(fragment)
                        elif kind == "message_delta":
                            delta_usage = obj.get("usage", {}) or {}
            except httpx.HTTPError as e:
                raise AnthropicError(f"request to {url} failed: {e}") from e
            except json.JSONDecodeError as e:
                raise AnthropicError(f"{url} returned a non-JSON data line: {e}") from e

        latency_ms = (time.monotonic() - start) * 1000.0
        text = "".join(chunks)
        logger.info(
            "anthropic.measure done model=%s request_id=%s latency_ms=%.1f ttft_ms=%s\n"
            "USAGE (verbatim from the provider - the source of every token count):\n"
            "message_start: %s\nmessage_delta: %s\n"
            "RESPONSE:\n%s",
            self.model,
            request_id or "(none)",
            latency_ms,
            f"{ttft_ms:.1f}" if ttft_ms is not None else "None",
            json.dumps(start_usage) if start_usage else "(none)",
            json.dumps(delta_usage) if delta_usage else "(none)",
            text,
        )
        return Measurement(
            text=text,
            tokens_in=start_usage.get("input_tokens", 0),
            tokens_out=delta_usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            model=self.model,
            request_id=request_id or None,
        )
