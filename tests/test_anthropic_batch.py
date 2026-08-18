"""Batch adapter: mocked lifecycle contracts, the runner's cache, a paid smoke.

Same register split as `test_anthropic.py`: respx-mocked contract tests lock the
submit -> poll -> retrieve lifecycle and the custom_id keying ($0, keyless); the
billed smoke submits ONE tiny fabricated job against the real Batches API to
prove auth + schema + the half-price booking before anything bigger.
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

from llm_benchmark.dataset import GoldenItem
from llm_benchmark.pricing import cost_usd
from llm_benchmark.providers.anthropic_batch import AnthropicBatchError, AnthropicBatchProvider
from llm_benchmark.providers.base import Measurement
from llm_benchmark.runners.batch_sweep import run_batch_sweep
from llm_benchmark.scorers.base import Scorer, ScoreResult

BASE = "https://api.anthropic.com"
BATCHES_URL = f"{BASE}/v1/messages/batches"
HAIKU = "claude-haiku-4-5-20251001"
BATCH_ID = "msgbatch_test"


def _batch_obj(status: str, *, results: bool = False) -> dict:
    return {
        "id": BATCH_ID,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {"processing": 0, "succeeded": 2, "errored": 0},
        "results_url": f"{BATCHES_URL}/{BATCH_ID}/results" if results else None,
    }


def _succeeded(text: str, tin: int, tout: int) -> dict:
    return {
        "type": "succeeded",
        "message": {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": tin, "output_tokens": tout},
        },
    }


def _results_jsonl(entries: dict[str, dict]) -> str:
    return "\n".join(
        json.dumps({"custom_id": cid, "result": result}) for cid, result in entries.items()
    )


@pytest.mark.mocked
@respx.mock
async def test_batch_run_full_lifecycle():
    """submit -> ended on first poll -> retrieve; answers keyed by custom_id,
    ttft is None (nothing streamed), latency is the shared job turnaround."""
    respx.post(BATCHES_URL).mock(return_value=httpx.Response(200, json=_batch_obj("in_progress")))
    respx.get(f"{BATCHES_URL}/{BATCH_ID}").mock(
        return_value=httpx.Response(200, json=_batch_obj("ended", results=True))
    )
    respx.get(f"{BATCHES_URL}/{BATCH_ID}/results").mock(
        return_value=httpx.Response(
            200,
            text=_results_jsonl(
                {
                    "q1__rep1": _succeeded("Ljubljana.", 21, 8),
                    "q2__rep1": _succeeded("4", 15, 2),
                }
            ),
        )
    )

    provider = AnthropicBatchProvider(model=HAIKU, api_key="test-key", poll_interval=0.0)
    measurements, errors = await provider.run(
        {"q1__rep1": "Capital of Slovenia?", "q2__rep1": "2+2?"}
    )

    assert errors == {}
    assert set(measurements) == {"q1__rep1", "q2__rep1"}
    m = measurements["q1__rep1"]
    assert m.text == "Ljubljana."
    assert m.tokens_in == 21
    assert m.tokens_out == 8
    assert m.model == HAIKU
    assert m.ttft_ms is None  # batch is never streamed - no first-token event
    assert m.latency_ms > 0.0
    # Every item in a batch shares the one job turnaround.
    assert measurements["q1__rep1"].latency_ms == measurements["q2__rep1"].latency_ms

    # The whole point of the lane: the same tokens cost exactly half.
    assert cost_usd(m, batch=True) == pytest.approx(cost_usd(m) * 0.5)


@pytest.mark.mocked
@respx.mock
async def test_batch_poll_waits_until_ended():
    """The first poll is still in_progress; the loop waits and re-polls."""
    respx.post(BATCHES_URL).mock(return_value=httpx.Response(200, json=_batch_obj("in_progress")))
    respx.get(f"{BATCHES_URL}/{BATCH_ID}").mock(
        side_effect=[
            httpx.Response(200, json=_batch_obj("in_progress")),
            httpx.Response(200, json=_batch_obj("ended", results=True)),
        ]
    )
    respx.get(f"{BATCHES_URL}/{BATCH_ID}/results").mock(
        return_value=httpx.Response(200, text=_results_jsonl({"q1__rep1": _succeeded("ok", 3, 1)}))
    )

    provider = AnthropicBatchProvider(model=HAIKU, api_key="test-key", poll_interval=0.0)
    measurements, errors = await provider.run({"q1__rep1": "q"})

    assert errors == {}
    assert measurements["q1__rep1"].text == "ok"


@pytest.mark.mocked
@respx.mock
async def test_batch_surfaces_non_succeeded_result():
    """An errored line lands in `errors`, not silently dropped; the succeeded
    sibling still comes back."""
    respx.post(BATCHES_URL).mock(return_value=httpx.Response(200, json=_batch_obj("in_progress")))
    respx.get(f"{BATCHES_URL}/{BATCH_ID}").mock(
        return_value=httpx.Response(200, json=_batch_obj("ended", results=True))
    )
    respx.get(f"{BATCHES_URL}/{BATCH_ID}/results").mock(
        return_value=httpx.Response(
            200,
            text=_results_jsonl(
                {
                    "good__rep1": _succeeded("fine", 3, 1),
                    "bad__rep1": {
                        "type": "errored",
                        "errored": {"error": {"type": "invalid_request", "message": "nope"}},
                    },
                }
            ),
        )
    )

    provider = AnthropicBatchProvider(model=HAIKU, api_key="test-key", poll_interval=0.0)
    measurements, errors = await provider.run({"good__rep1": "q", "bad__rep1": "q"})

    assert set(measurements) == {"good__rep1"}
    assert "bad__rep1" in errors
    assert "errored" in errors["bad__rep1"]


@pytest.mark.mocked
@respx.mock
async def test_batch_http_error_surfaces_detail():
    respx.post(BATCHES_URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "message": "bad batch"}}
        )
    )
    provider = AnthropicBatchProvider(model=HAIKU, api_key="test-key")
    with pytest.raises(AnthropicBatchError, match="bad batch"):
        await provider.submit({"q1__rep1": "q"})


# --- the runner: cache is the spend control -------------------------------


class _StubBatchProvider:
    """Duck-typed stand-in for AnthropicBatchProvider.run - records the prompts
    it was asked to submit so a test can assert the cache skipped a re-submit."""

    def __init__(self, model: str, answers: dict[str, str]) -> None:
        self.model = model
        self._answers = answers
        self.submitted: list[dict[str, str]] = []
        # Mirror the real provider's receipt attributes (set on each submit).
        self.last_batch_id: str | None = None
        self.last_request_counts: dict | None = None

    async def run(self, prompts: dict[str, str]):
        self.submitted.append(dict(prompts))
        self.last_batch_id = f"msgbatch_stub_{len(self.submitted)}"
        self.last_request_counts = {"succeeded": len(prompts)}
        measurements = {
            cid: Measurement(
                text=self._answers[cid],
                tokens_in=10,
                tokens_out=5,
                latency_ms=90_000.0,  # a 90s job turnaround, shared by all
                ttft_ms=None,
                model=self.model,
            )
            for cid in prompts
        }
        return measurements, {}


class _StubScorer(Scorer):
    name = "stub"

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        return ScoreResult(passed=True, score=1.0, reason="stub-pass")


@pytest.mark.mocked
async def test_run_batch_sweep_writes_records_and_caches(tmp_path: Path):
    items = [
        GoldenItem("q1", "Capital of Slovenia?", "Ljubljana", "easy", "factual"),
        GoldenItem("q2", "2+2?", "4", "easy", "arith"),
    ]
    provider = _StubBatchProvider(HAIKU, {"q1__rep1": "Ljubljana.", "q2__rep1": "4"})

    records = await run_batch_sweep(
        items, HAIKU, provider, _StubScorer(), reps=1, cache_dir=tmp_path
    )

    assert len(records) == 2
    r = records[0]
    assert r["mode"] == "batch"
    assert r["batch_turnaround_s"] == pytest.approx(90.0)
    assert r["measurement"]["ttft_ms"] is None
    # Half-price booking: 10 in-tok * $1/Mtok * 0.5 + 5 out-tok * $5/Mtok * 0.5.
    assert r["cost_usd"] == pytest.approx((10 * 1.0 + 5 * 5.0) * 0.5 / 1_000_000)
    assert (tmp_path / f"{HAIKU}__q1__rep1.json").exists()
    assert len(provider.submitted) == 1 and len(provider.submitted[0]) == 2

    # Second run: everything is cached, so NOTHING is submitted (the spend control).
    records2 = await run_batch_sweep(
        items, HAIKU, provider, _StubScorer(), reps=1, cache_dir=tmp_path
    )
    assert len(records2) == 2
    assert len(provider.submitted) == 1  # no second submit


# --- billed smoke: real Batches API, tiny fabricated job ------------------


@pytest.mark.billed
async def test_batch_billed_smoke():
    """SMOKE: a two-item fabricated batch against the real Batches API. Proves
    auth, the submit->poll->retrieve schema, and the half-price booking before
    any real sweep. A batch can take minutes; the poll_timeout is the backstop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set (expected in gitignored .env)")

    provider = AnthropicBatchProvider(
        model=HAIKU, api_key=api_key, poll_interval=10.0, poll_timeout=1800.0
    )
    prompts = {
        "smoke_a__rep1": "What is 2+2? Reply with only the number.",
        "smoke_b__rep1": "What is the capital of Slovenia? Reply with only the city.",
    }
    measurements, errors = await provider.run(prompts)

    assert errors == {}, errors
    assert "4" in measurements["smoke_a__rep1"].text
    assert "Ljubljana" in measurements["smoke_b__rep1"].text
    for m in measurements.values():
        assert m.tokens_in > 0 and m.tokens_out > 0
        assert m.ttft_ms is None  # batch is never streamed
        full = cost_usd(m)
        assert cost_usd(m, batch=True) == pytest.approx(full * 0.5)

    cache_dir = Path("cache-batch")
    cache_dir.mkdir(exist_ok=True)
    for cid, m in measurements.items():
        record = {
            "captured_at": dt.datetime.now(dt.UTC).isoformat(),
            "provider": "anthropic_batch",
            "custom_id": cid,
            "measurement": asdict(m),
            "cost_usd": cost_usd(m, batch=True),
            "mode": "batch",
        }
        (cache_dir / f"{m.model}__smoke-batch-{cid}.json").write_text(json.dumps(record, indent=2))
