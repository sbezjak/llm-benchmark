"""Per-model pricing and cost-per-call.

Rates are USD per 1M tokens (input, output), taken from each provider's
public rate card, verified Aug 2026. Cost math runs off a cached
`Measurement` — never a re-call. The local Ollama baseline is $0 by
definition.

Why the rates are a hardcoded, dated table and NOT fetched live. This is a
deliberate choice, not a shortcut. There is no reliable machine-readable
pricing endpoint - provider rates live on marketing pages, so "fetch the
price" means scraping a page that breaks silently on redesign. Worse, a
live fetch would put a moving number and a network dependency into the one
calculation the whole project needs pinned: cost is computed off the frozen
artifact and must reproduce the same dollars on every replay. The
production pattern for cost tracking is exactly this - a versioned, dated
rate table reviewed on a cadence - not a per-run fetch. The discipline that
makes it production-grade is the review cadence, so:

  REVIEW CADENCE - re-verify this table against each provider's rate card
  before any full paid sweep, and whenever a dated caveat below expires.

Dated caveats (the review triggers):
- `claude-sonnet-5` is the introductory rate, valid through 2026-08-31
  (standard rate is $3/$15) - re-check if a sweep runs past that date.
- DeepSeek announced a price rise (no date yet) - re-check its rate card on
  sweep day.
"""

from __future__ import annotations

from llm_benchmark.providers.base import Measurement

PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "deepseek-v4-pro": (0.435, 0.87),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "llama3.2": (0.0, 0.0),
}


# The Batch API discount: non-interactive batch runs are billed at half the
# per-token rate on both Anthropic and the OpenAI-compatible backends. It is a
# multiplier on the rate card, not a different card - so the one price table
# above stays the source of truth and `batch=True` just halves the result.
BATCH_DISCOUNT = 0.5


def cost_components_usd(m: Measurement, batch: bool = False) -> tuple[float, float]:
    """Input and output dollar cost of one call, split by token type. The split
    is what a trace/dashboard shows as per-component cost; `cost_usd` is their
    sum, so the two never drift.

    `batch=True` applies the Batch API's 50% discount - the same token counts,
    billed at half price because the request went through the async batch lane
    instead of the interactive one. The caller (the batch runner) knows which
    lane a capture came from; the price table itself does not change."""
    price_in, price_out = PRICE_PER_MTOK[m.model]
    factor = BATCH_DISCOUNT if batch else 1.0
    return (
        m.tokens_in * price_in * factor / 1_000_000,
        m.tokens_out * price_out * factor / 1_000_000,
    )


def cost_usd(m: Measurement, batch: bool = False) -> float:
    """Dollar cost of one measured call, from the provider's own token counts.
    `batch=True` bills at the Batch API's half rate (see `cost_components_usd`)."""
    in_cost, out_cost = cost_components_usd(m, batch=batch)
    return in_cost + out_cost
