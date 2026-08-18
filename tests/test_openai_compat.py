"""OpenAI-compatible adapter: mocked contracts, then a paid smoke test.

Register split, same as the Ollama tests:

- mocked (respx, $0, keyless): lock the parsing contract — text reassembly
  from SSE deltas, token counts taken from the provider's `usage` object,
  a real time-to-first-token, and the DeepSeek base_url swap riding the
  same class with no new parsing.
- billed (real spend, opt-in): SMOKE only — one fabricated item against the
  real API to prove auth + schema before any sweep. Never a full suite here.
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
from llm_benchmark.providers.openai_compat import OpenAICompatError, OpenAICompatProvider

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.6-luna"
DEEPSEEK_MODEL = "deepseek-v4-pro"


def _sse_body(fragments: list[str], prompt_tokens: int, completion_tokens: int) -> str:
    """SSE exactly as Chat Completions streams it: one `data:` chunk per text
    delta, a final chunk carrying `usage` (choices empty), then `[DONE]`."""
    chunks = [
        {"choices": [{"delta": {"content": f}, "index": 0}], "usage": None} for f in fragments
    ]
    chunks.append(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@pytest.mark.mocked
@respx.mock
async def test_openai_measure_captures_all_fields():
    body = _sse_body(["4", ""], prompt_tokens=14, completion_tokens=1)
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

    provider = OpenAICompatProvider(model="gpt-5.6-luna", api_key="test-key")
    m = await provider.measure("What is 2+2? Reply with only the number.")

    assert m.text == "4"
    assert m.tokens_in == 14  # from usage.prompt_tokens, not re-tokenized
    assert m.tokens_out == 1  # from usage.completion_tokens
    assert m.model == "gpt-5.6-luna"
    assert m.latency_ms >= 0.0
    assert m.ttft_ms is not None
    assert m.ttft_ms <= m.latency_ms

    # `generate` is the string-only view over the same call.
    assert await provider.generate("q") == "4"


@pytest.mark.mocked
@respx.mock
async def test_openai_ttft_none_when_no_text():
    """A stream with no content deltas never has a first-token moment."""
    body = _sse_body([], prompt_tokens=5, completion_tokens=0)
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=body))

    m = await OpenAICompatProvider(model="gpt-5.6-luna", api_key="test-key").measure("q")

    assert m.text == ""
    assert m.tokens_out == 0
    assert m.ttft_ms is None


@pytest.mark.mocked
@respx.mock
async def test_openai_http_error_raises_adapter_error():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))

    with pytest.raises(OpenAICompatError):
        await OpenAICompatProvider(model="gpt-5.6-luna", api_key="wrong").measure("q")


@pytest.mark.mocked
@respx.mock
async def test_deepseek_mocked_rides_openai_adapter():
    """DeepSeek is a config variant of the same class: base_url + key swap,
    identical parsing. This test is the $0 proof of that seam."""
    body = _sse_body(["Paris", "."], prompt_tokens=9, completion_tokens=2)
    respx.post(DEEPSEEK_URL).mock(return_value=httpx.Response(200, content=body))

    provider = OpenAICompatProvider(
        model="deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
    )
    m = await provider.measure("Capital of France?")

    assert m.text == "Paris."
    assert m.tokens_in == 9
    assert m.tokens_out == 2
    assert m.model == "deepseek-v4-pro"


@pytest.mark.billed
@pytest.mark.parametrize("model", [OPENAI_MODEL])
async def test_openai_billed_smoke(model: str):
    """SMOKE: one fabricated item against the real OpenAI API (~hundredths of
    a cent). Proves auth, response schema, and usage capture before anything
    bigger. The raw measurement is cached to disk - cost math reads the
    artifact, never re-calls. Model in the test id (bracketed) so the report
    names the model per row, matching the Anthropic smoke."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set (expected in gitignored .env)")

    prompt = "What is 2+2? Reply with only the number."
    provider = OpenAICompatProvider(model=model, api_key=api_key)
    m = await provider.measure(prompt)

    assert "4" in m.text
    assert m.tokens_in > 0
    assert m.tokens_out > 0
    assert m.latency_ms > 0.0

    cost = cost_usd(m)
    assert 0.0 < cost < 0.001  # sanity ceiling: a smoke item costs well under a tenth of a cent

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    record = {
        "captured_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": "openai",
        "prompt": prompt,
        "measurement": asdict(m),
        "cost_usd": cost,
    }
    (cache_dir / f"{m.model}__smoke-arithmetic.json").write_text(json.dumps(record, indent=2))


@pytest.mark.billed
@pytest.mark.parametrize("model", [DEEPSEEK_MODEL])
async def test_deepseek_billed_smoke(model: str):
    """SMOKE: one fabricated item against the real DeepSeek API - the billed
    proof that the OpenAI-compatible adapter serves DeepSeek with only a
    base_url + key swap. Re-check DeepSeek's rate card the day of a real sweep
    (announced price rise, no date). Raw measurement cached to disk. Model in
    the test id (bracketed) so the report names the model per row."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set (expected in gitignored .env)")

    prompt = "What is 2+2? Reply with only the number."
    provider = OpenAICompatProvider(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )
    m = await provider.measure(prompt)

    assert "4" in m.text
    assert m.tokens_in > 0
    assert m.tokens_out > 0
    assert m.latency_ms > 0.0

    cost = cost_usd(m)
    assert 0.0 < cost < 0.001  # DeepSeek is cheap; a smoke item is well under a cent

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    record = {
        "captured_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": "deepseek",
        "prompt": prompt,
        "measurement": asdict(m),
        "cost_usd": cost,
    }
    (cache_dir / f"{m.model}__smoke-arithmetic.json").write_text(json.dumps(record, indent=2))
