"""The LLM-judge's usage contract ($0, mocked - no HTTP).

The judge is itself an LLM call, so its verdict now carries the grader's own
model + token counts (the `judge_*` fields on `ScoreResult`). Two paths:

- `judge_measure_fn` (the default path) returns a `Measurement`, so usage flows
  through to the verdict - this is what lets a trace show evaluator cost.
- `judge_fn` (the older text-only injection, e.g. the re-judge flow) returns a
  bare string, so there is no usage and the `judge_*` fields stay None.

Both are exercised with injected fakes, so this stays offline and free.
"""

from __future__ import annotations

import pytest

from llm_benchmark.providers.base import Measurement
from llm_benchmark.scorers.judge import LLMJudgeScorer

pytestmark = pytest.mark.mocked

_RAW = '{"reasoning": "ok", "correctness": 9, "relevance": 10}'


async def test_measure_path_populates_judge_usage():
    async def fake_measure(prompt: str) -> Measurement:
        return Measurement(
            text=_RAW, tokens_in=120, tokens_out=25, latency_ms=50.0, ttft_ms=None, model="llama3.2"
        )

    scorer = LLMJudgeScorer(judge_measure_fn=fake_measure)
    verdict = await scorer.score("q", "a", "e")

    assert verdict.score == pytest.approx((9 + 10) / 20.0)
    # The grader's own resource use rides along on the verdict.
    assert verdict.judge_model == "llama3.2"
    assert verdict.judge_tokens_in == 120
    assert verdict.judge_tokens_out == 25


async def test_text_only_judge_fn_has_no_usage():
    async def fake_text(prompt: str) -> str:
        return _RAW

    scorer = LLMJudgeScorer(judge_fn=fake_text)
    verdict = await scorer.score("q", "a", "e")

    assert verdict.score == pytest.approx((9 + 10) / 20.0)
    # Text-only path cannot know tokens, so usage stays None (backward compatible).
    assert verdict.judge_model is None
    assert verdict.judge_tokens_in is None
    assert verdict.judge_tokens_out is None
