from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    """One (model, prompt) generation, with the numbers benchmarking needs.

    `generate` returns just the completion string; `measure` returns this,
    widening the seam so the cost and latency axes have something to read.
    All timing is wall-clock in milliseconds.

    - `text` — the completion (what `generate` hands back).
    - `tokens_in` / `tokens_out` — prompt and completion token counts, taken
      from the provider's own report (each provider counts differently, so a
      fair cost number uses the source's count, not a re-tokenization here).
    - `latency_ms` — total wall-clock around the whole call.
    - `ttft_ms` — time to the first streamed chunk. `None` when the backend
      wasn't streamed, so there was no first-token event to time.
    - `model` — the model that produced this, so a cached measurement is
      self-describing once more than one model is in play.
    - `request_id` — the provider's own request id from the response headers
      (`x-request-id` / `request-id`), a verifiable receipt that cross-references
      the provider console. Captured in-record so it survives in the cache, not
      only in the run log (a log-only receipt is one lost `tee|tail` from gone).
      `None` for backends that don't return one (local Ollama) or older captures.
    """

    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    ttft_ms: float | None
    model: str
    request_id: str | None = None


class Provider(ABC):
    """Abstract LLM backend.

    Concrete providers are constructed with their own config (model name,
    base URL, timeouts, sampling params) and expose `measure`, which maps a
    prompt to a `Measurement` (text + tokens + latency). `generate` is the
    narrow string-only view over the same call, kept so scorers and the
    LLM-judge (which use a provider's `generate` as their callable) treat any
    backend identically without knowing about the measurement fields.
    """

    @abstractmethod
    async def measure(self, prompt: str) -> Measurement: ...

    async def generate(self, prompt: str) -> str:
        """The completion string alone. Thin view over `measure` so the
        scorer/judge seam stays a plain `str -> str` callable."""
        return (await self.measure(prompt)).text
