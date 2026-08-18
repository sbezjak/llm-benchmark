"""Anthropic adapter: mocked contracts, then a paid smoke test per model.

Same register split as the OpenAI-compatible tests: respx-mocked contract
tests lock the SSE parsing ($0, keyless); the billed smoke runs ONE
fabricated item per model (Haiku, Sonnet) against the real API to prove
auth + schema before any sweep.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
import respx

from llm_benchmark.pricing import cost_usd
from llm_benchmark.providers.anthropic import AnthropicError, AnthropicProvider

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"


def _sse_body(fragments: list[str], input_tokens: int, output_tokens: int) -> str:
    """SSE exactly as the Messages API streams it: `message_start` carries
    `usage.input_tokens`, text arrives as `content_block_delta`/`text_delta`
    events, and the final `message_delta` carries `usage.output_tokens`."""
    events: list[tuple[str, dict]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_test", "usage": {"input_tokens": input_tokens}},
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0}),
    ]
    events += [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f}},
        )
        for f in fragments
    ]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


@pytest.mark.mocked
@respx.mock
async def test_anthropic_measure_captures_all_fields():
    body = _sse_body(["Ljub", "ljana."], input_tokens=21, output_tokens=8)
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

    provider = AnthropicProvider(model=HAIKU, api_key="test-key")
    m = await provider.measure("What is the capital of Slovenia?")

    assert m.text == "Ljubljana."
    assert m.tokens_in == 21  # from message_start usage.input_tokens
    assert m.tokens_out == 8  # from message_delta usage.output_tokens
    assert m.model == HAIKU
    assert m.latency_ms >= 0.0
    assert m.ttft_ms is not None
    assert m.ttft_ms <= m.latency_ms

    # `generate` is the string-only view over the same call.
    assert await provider.generate("q") == "Ljubljana."


@pytest.mark.mocked
@respx.mock
async def test_anthropic_ttft_none_when_no_text():
    """A stream with no text deltas never has a first-token moment."""
    body = _sse_body([], input_tokens=5, output_tokens=0)
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, content=body))

    m = await AnthropicProvider(model=HAIKU, api_key="test-key").measure("q")

    assert m.text == ""
    assert m.tokens_out == 0
    assert m.ttft_ms is None


@pytest.mark.mocked
@respx.mock
async def test_anthropic_http_error_surfaces_detail():
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "message": "bad field"}}
        )
    )

    with pytest.raises(AnthropicError, match="bad field"):
        await AnthropicProvider(model=HAIKU, api_key="test-key").measure("q")


@pytest.mark.billed
@pytest.mark.parametrize("model", [HAIKU, SONNET])
async def test_anthropic_billed_smoke(model: str):
    """SMOKE: one fabricated item per model against the real Anthropic API
    (fractions of a cent). Proves auth, SSE schema, and usage capture before
    anything bigger. Raw measurement cached to disk - cost math reads the
    artifact, never re-calls."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set (expected in gitignored .env)")

    prompt = "What is 2+2? Reply with only the number."
    provider = AnthropicProvider(model=model, api_key=api_key)
    m = await provider.measure(prompt)

    assert "4" in m.text
    assert m.tokens_in > 0
    assert m.tokens_out > 0
    assert m.latency_ms > 0.0

    cost = cost_usd(m)
    assert 0.0 < cost < 0.005  # sanity ceiling; Sonnet may spend thinking tokens

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    record = {
        "captured_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": "anthropic",
        "prompt": prompt,
        "measurement": asdict(m),
        "cost_usd": cost,
    }
    (cache_dir / f"{m.model}__smoke-arithmetic.json").write_text(json.dumps(record, indent=2))
