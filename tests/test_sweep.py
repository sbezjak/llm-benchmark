"""The sweep runner: a $0 mocked capture+cache test, then the billed sweep.

Register split, same discipline as the provider tests:

- mocked ($0, no HTTP): lock the runner's capture+cache loop. Fake provider +
  fake scorer are injected, so the core is exercised with zero network and zero
  spend. The load-bearing assertion is idempotency: a second run reads the
  cached capture instead of re-calling the provider (the spend control).
- billed (real spend, opt-in): a thin SMOKE on 2 items x all models. The full
  sweep is not a test - it is the `python -m llm_benchmark.sweep` job, which
  reuses the smoke's captures and renders the keeper report off `cache/`. The
  smoke is the last cheap gate before that job spends. Never fired blind -
  monitor the log.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from llm_benchmark.dataset import GoldenItem, load_golden_set
from llm_benchmark.providers.base import Measurement, Provider
from llm_benchmark.runners.sweep import (
    build_default_providers,
    capture_path,
    default_scorer,
    format_table,
    run_sweep,
    summarize,
)
from llm_benchmark.scorers.base import Scorer, ScoreResult

ITEM = GoldenItem(
    id="factual_001",
    question="What is the capital of Slovenia?",
    expected="Ljubljana",
    difficulty="easy",
    category="factual",
)


class FakeProvider(Provider):
    """Returns a fixed Measurement and counts how many times it was called, so
    a test can prove the cache prevented a second call."""

    def __init__(self, model: str = "gpt-5.6-luna") -> None:
        self.model = model
        self.calls = 0

    async def measure(self, prompt: str) -> Measurement:
        self.calls += 1
        return Measurement(
            text="Ljubljana.",
            tokens_in=10,
            tokens_out=5,
            latency_ms=12.0,
            ttft_ms=8.0,
            model=self.model,
        )


class FakeScorer(Scorer):
    """Fixed verdict; counts calls to prove the judge is also cache-gated."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        self.calls += 1
        # Report judge usage (model "llama3.2" is priced at $0) so the evaluator
        # span's model/usage/cost path is exercised.
        return ScoreResult(
            passed=True,
            score=0.9,
            reason="fake verdict",
            judge_model="llama3.2",
            judge_tokens_in=7,
            judge_tokens_out=3,
        )


@pytest.mark.mocked
async def test_run_sweep_captures_and_writes(tmp_path: Path):
    provider = FakeProvider()
    scorer = FakeScorer()

    records = await run_sweep(
        [ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path
    )

    assert len(records) == 1
    rec = records[0]
    # The record is self-describing: model, item, the raw measurement, cost off
    # that measurement, and the judge verdict inline.
    assert rec["model"] == "gpt-5.6-luna"
    assert rec["item_id"] == "factual_001"
    assert rec["measurement"]["tokens_in"] == 10
    assert rec["measurement"]["tokens_out"] == 5
    # cost = (10 * 0.20 + 5 * 1.20) / 1e6 = 8e-6, computed off the Measurement.
    assert rec["cost_usd"] == pytest.approx(8e-6)
    assert rec["judge"]["score"] == 0.9
    assert rec["judge"]["passed"] is True

    # It landed on disk and parses.
    path = capture_path(tmp_path, "gpt-5.6-luna", "factual_001", 1)
    assert path.exists()
    assert json.loads(path.read_text())["item_id"] == "factual_001"

    assert provider.calls == 1
    assert scorer.calls == 1


@pytest.mark.mocked
async def test_run_sweep_is_idempotent(tmp_path: Path):
    """The spend control: a second sweep over an already-captured cell reads the
    cache and does NOT re-call the provider or the judge."""
    provider = FakeProvider()
    scorer = FakeScorer()

    await run_sweep([ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path)
    assert provider.calls == 1

    # Second run, same cache dir: cache hit, no new call.
    records = await run_sweep(
        [ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path
    )
    assert provider.calls == 1  # unchanged - the paid call did NOT fire again
    assert scorer.calls == 1  # judge is cache-gated too
    assert records[0]["measurement"]["tokens_in"] == 10  # served from disk


class RecordingTracer:
    """A fake Tracer that records the hook calls the runner makes, so a test can
    prove the seam fires on a miss (and is bypassed on a cache hit) without any
    real Langfuse client. Satisfies the `Tracer` protocol structurally."""

    def __init__(self) -> None:
        self.captures: list[tuple[str, str, int]] = []
        self.generations: list[str] = []
        self.scores: list[float] = []
        self.updates: list[dict] = []

    @contextmanager
    def capture(self, model: str, item_id: str, rep: int):
        self.captures.append((model, item_id, rep))
        yield self

    @contextmanager
    def generation(self, name: str, input: str, model: str | None = None):
        self.generations.append(name)
        yield self

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def score(self, value: float, comment: str) -> None:
        self.scores.append(value)


@pytest.mark.mocked
async def test_tracer_hook_fires_on_miss_and_record_is_unchanged(tmp_path: Path):
    """The live-instrumentation seam: on a cache MISS the runner opens the capture root and two
    generations (answer + judge), sets the measured output/usage/cost on each,
    and attaches the judge score - and the record written to disk is identical to
    the untraced path (tracing is observation, it never changes the artifact)."""
    tracer = RecordingTracer()
    provider = FakeProvider()
    scorer = FakeScorer()

    records = await run_sweep(
        [ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path, tracer=tracer
    )

    # The hook fired once, opening both generations (answer, then judge).
    assert tracer.captures == [("gpt-5.6-luna", "factual_001", 1)]
    assert tracer.generations == ["answer-generation", "judge-evaluation"]
    assert tracer.scores == [0.9]
    # The answer generation carries the real measured numbers, each in its
    # canonical field: token usage, the input/output/total cost split (not just
    # the aggregate), TTFT as completion_start_time, exact latency in metadata.
    gen_update = tracer.updates[0]
    assert gen_update["output"] == "Ljubljana."
    assert gen_update["usage_details"] == {"input": 10, "output": 5}
    assert gen_update["cost_details"] == {
        "input": pytest.approx(2e-6),  # 10 in-tok * $0.20/Mtok
        "output": pytest.approx(6e-6),  # 5 out-tok * $1.20/Mtok
        "total": pytest.approx(8e-6),
    }
    assert gen_update["metadata"] == {"latency_ms": 12.0, "ttft_ms": 8.0}
    assert "completion_start_time" in gen_update  # TTFT was present, so it is set
    # The judge generation carries the verdict as its output, plus the judge's own
    # model/usage/cost so eval spend is visible like the answer's. (Modeled as a
    # generation, not an evaluator span, because Langfuse only stores usage/cost
    # on generations - an evaluator span would drop them.)
    jgen_update = tracer.updates[1]
    assert jgen_update["output"] == {
        "score": 0.9,
        "passed": True,
        "reason": "fake verdict",
    }
    assert jgen_update["model"] == "llama3.2"
    assert jgen_update["usage_details"] == {"input": 7, "output": 3}
    assert jgen_update["cost_details"] == {"input": 0.0, "output": 0.0, "total": 0.0}
    # The root span carries the headline in/out and the eval reference + item
    # facets (difficulty/category), so the trace is self-describing and filterable.
    root_update = tracer.updates[2]
    assert root_update["input"] == ITEM.question
    assert root_update["output"] == "Ljubljana."
    assert root_update["metadata"] == {
        "expected": ITEM.expected,
        "difficulty": ITEM.difficulty,
        "category": ITEM.category,
    }

    # The record itself is exactly what the untraced runner produces.
    rec = records[0]
    assert rec["measurement"]["tokens_in"] == 10
    assert rec["cost_usd"] == pytest.approx(8e-6)
    assert rec["judge"]["score"] == 0.9


@pytest.mark.mocked
async def test_tracer_hook_skipped_on_cache_hit(tmp_path: Path):
    """A cache hit does no call and no scoring, so there is nothing to time - the
    tracer must not be opened at all on the second run."""
    provider = FakeProvider()
    scorer = FakeScorer()
    await run_sweep([ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path)

    tracer = RecordingTracer()
    await run_sweep(
        [ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path, tracer=tracer
    )
    assert tracer.captures == []  # cache hit -> no span opened
    assert provider.calls == 1  # and no re-call, as before


@pytest.mark.mocked
async def test_run_sweep_force_recalls(tmp_path: Path):
    """`force=True` is the escape hatch: it re-calls even on a cache hit."""
    provider = FakeProvider()
    scorer = FakeScorer()

    await run_sweep([ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path)
    await run_sweep(
        [ITEM], {"gpt-5.6-luna": provider}, scorer, reps=1, cache_dir=tmp_path, force=True
    )
    assert provider.calls == 2


@pytest.mark.mocked
def test_summarize_rolls_up_per_model():
    records = [
        {
            "model": "m1",
            "cost_usd": 1e-5,
            "measurement": {"latency_ms": 100.0, "ttft_ms": 40.0, "tokens_out": 5},
            "judge": {"score": 1.0, "passed": True},
        },
        {
            "model": "m1",
            "cost_usd": 3e-5,
            "measurement": {"latency_ms": 200.0, "ttft_ms": 60.0, "tokens_out": 15},
            "judge": {"score": 0.5, "passed": False},
        },
        {
            "model": "m2",
            "cost_usd": 0.0,
            "measurement": {"latency_ms": 50.0, "ttft_ms": None, "tokens_out": 3},
            "judge": {"score": 0.8, "passed": True},
        },
    ]
    summaries = summarize(records)

    by_model = {s.model: s for s in summaries}
    m1 = by_model["m1"]
    assert m1.n == 2
    assert m1.mean_cost_usd == pytest.approx(2e-5)
    assert m1.total_cost_usd == pytest.approx(4e-5)
    assert m1.mean_latency_ms == pytest.approx(150.0)
    assert m1.mean_ttft_ms == pytest.approx(50.0)
    assert m1.mean_score == pytest.approx(0.75)
    assert m1.pass_rate == pytest.approx(0.5)

    # m2's only ttft is None -> mean_ttft_ms is None, not a crash.
    assert by_model["m2"].mean_ttft_ms is None
    # Ordered by mean score desc: m2 (0.8) before m1 (0.75).
    assert [s.model for s in summaries] == ["m2", "m1"]

    # The table renders without error and names every model.
    table = format_table(summaries)
    assert "m1" in table and "m2" in table


# --------------------------------------------------------------------------
# Billed: the SMOKE only (2 items). The FULL sweep is a job, not a test -
# `python -m llm_benchmark.sweep` drives the grid and renders the report off
# the cache. pytest keeps the mocked harness tests above and this thin smoke,
# which proves the runner drives every real provider before the job spends.
# --------------------------------------------------------------------------

SMOKE_ITEM_IDS = ["factual_001", "reasoning_001"]  # one easy, one hard (reasoning tokens)


def _require_paid_providers() -> dict:
    providers = build_default_providers()
    # Ollama alone is always present; a real sweep needs the paid keys.
    if len(providers) < 2:
        pytest.skip("no paid provider keys in .env - the sweep needs OpenAI/Anthropic/DeepSeek")
    return providers


@pytest.mark.billed
async def test_sweep_smoke_two_items():
    """SMOKE: 2 items x all models, reps=1. Proves the runner drives every real
    provider end to end and the captures land in `cache/` and parse - before the
    full sweep job spends. Idempotent, so these captures are reused (not re-paid)
    when `python -m llm_benchmark.sweep` runs the full grid."""
    providers = _require_paid_providers()
    items = [i for i in load_golden_set() if i.id in SMOKE_ITEM_IDS]
    assert len(items) == len(SMOKE_ITEM_IDS)

    records = await run_sweep(items, providers, default_scorer(), reps=1)

    assert len(records) == len(items) * len(providers)
    for rec in records:
        assert rec["measurement"]["tokens_out"] >= 0
        assert rec["cost_usd"] >= 0.0
        assert "score" in rec["judge"]
        # It parses off disk.
        path = capture_path(Path("cache"), rec["model"], rec["item_id"], rec["rep"])
        assert json.loads(path.read_text())["item_id"] == rec["item_id"]

    summaries = summarize(records)
    total = sum(s.total_cost_usd for s in summaries)
    import logging

    logging.getLogger(__name__).info(
        "SWEEP SMOKE table (%d captures, total spend $%.5f):\n%s",
        len(records),
        total,
        format_table(summaries),
    )
    assert total < 0.10  # smoke ceiling: 2 items x 5 models is well under a dime
