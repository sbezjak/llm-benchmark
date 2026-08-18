"""Judge ablation runner: mocked ($0), idempotency the load-bearing assertion.

A fake judge provider returns a fixed rubric JSON and counts its calls, so the
test proves the re-judge grades cached answer text without touching HTTP and -
the spend control - that a second pass reads the judged cache instead of calling
the judge again. No answering model is constructed at all here: the ablation only
ever sees answer TEXT, which is the whole point (the paid answers are not
re-run)."""

from __future__ import annotations

import pytest

import llm_benchmark.rejudge as rejudge_cli
from llm_benchmark.providers.base import Measurement, Provider
from llm_benchmark.runners.rejudge import judge_comparison, rejudge_all, rejudge_path


class FakeJudge(Provider):
    """Returns a fixed judge verdict and counts calls, priced as gpt so
    cost_usd resolves. The model name must be in the pricing table."""

    def __init__(self) -> None:
        self.calls = 0

    async def measure(self, prompt: str) -> Measurement:
        self.calls += 1
        return Measurement(
            text='{"reasoning": "looks right", "correctness": 9, "relevance": 10}',
            tokens_in=120,
            tokens_out=25,
            latency_ms=200.0,
            ttft_ms=None,
            model="gpt-5.6-luna",
        )


def _cached_answer(answer_model: str, item_id: str, rep: int, text: str) -> dict:
    """A sweep capture, only the fields the re-judge reads populated."""
    return {
        "model": answer_model,
        "item_id": item_id,
        "rep": rep,
        "prompt": "What is the capital of Slovenia?",
        "expected": "Ljubljana",
        "measurement": {"text": text},
    }


@pytest.mark.mocked
async def test_rejudge_grades_cached_text_and_captures_cost(tmp_path):
    records = [
        _cached_answer("gpt-5.6-luna", "factual_001", 1, "Ljubljana."),
        _cached_answer("llama3.2", "factual_001", 1, "Ljubljana is the capital."),
    ]
    judge = FakeJudge()

    judged, spend = await rejudge_all(
        records, judge, judge_model="gpt-5.6-luna", judged_dir=tmp_path
    )

    assert judge.calls == 2  # one judge call per cached answer
    assert len(judged) == 2
    # (9 + 10) / 20 = 0.95, above the 0.7 threshold.
    assert judged[0]["judge"]["score"] == pytest.approx(0.95)
    assert judged[0]["judge"]["passed"] is True
    assert judged[0]["answer_text"] == "Ljubljana."
    assert judged[0]["judge_model"] == "gpt-5.6-luna"
    # Spend is real (positive) and the per-record cost is recorded.
    assert spend > 0.0
    assert judged[0]["judge_cost_usd"] > 0.0
    # It landed on disk under the judge+answer keyed name.
    assert rejudge_path(tmp_path, "gpt-5.6-luna", "gpt-5.6-luna", "factual_001", 1).exists()


@pytest.mark.mocked
async def test_rejudge_is_idempotent(tmp_path):
    records = [_cached_answer("gpt-5.6-luna", "factual_001", 1, "Ljubljana.")]

    judge = FakeJudge()
    await rejudge_all(records, judge, judge_model="gpt-5.6-luna", judged_dir=tmp_path)
    assert judge.calls == 1

    # Second pass: a fresh judge must NOT be called - the cache answers.
    judge2 = FakeJudge()
    judged, spend = await rejudge_all(
        records, judge2, judge_model="gpt-5.6-luna", judged_dir=tmp_path
    )
    assert judge2.calls == 0  # spend control: cache hit, no re-call
    assert spend == 0.0
    assert judged[0]["judge"]["score"] == pytest.approx(0.95)


def _sweep_record(model: str, item_id: str, rep: int, text: str, score: float) -> dict:
    """A full sweep capture (has a local `judge` verdict), the shape
    load_cached_records returns."""
    return {
        "model": model,
        "item_id": item_id,
        "rep": rep,
        "prompt": "q?",
        "expected": "a",
        "measurement": {"text": text},
        "cost_usd": 0.0,
        "judge": {"score": score, "passed": score >= 0.7, "reason": "local"},
    }


@pytest.mark.mocked
def test_judge_comparison_flags_passfail_disagreements():
    # Local passed a weak answer (0.75); the re-judge fails it (0.50) - a disagreement.
    records = [
        _sweep_record("llama3.2", "q1", 1, "twice", 0.75),
        _sweep_record("gpt-5.6-luna", "q1", 1, "three", 1.0),
    ]
    judged = [
        {
            "answer_model": "llama3.2",
            "item_id": "q1",
            "rep": 1,
            "judge": {"score": 0.50, "passed": False, "reason": "wrong"},
        },
        {
            "answer_model": "gpt-5.6-luna",
            "item_id": "q1",
            "rep": 1,
            "judge": {"score": 1.0, "passed": True, "reason": "right"},
        },
    ]
    rows, disagreements = judge_comparison(records, judged)

    assert len(disagreements) == 1
    d = disagreements[0]
    assert d["model"] == "llama3.2" and d["local_passed"] and not d["other_passed"]

    by_model = {r.model: r for r in rows}
    assert by_model["llama3.2"].local_pass == 1
    assert by_model["llama3.2"].other_pass == 0  # the re-judge caught it


@pytest.mark.mocked
def test_rejudge_cli_end_to_end(tmp_path, monkeypatch):
    """The CLI entry point: fake judge + fake cache, no HTTP, writes under a tmp
    cwd. Locks the job's glue - provider pick, limit, comparison - end to end."""
    monkeypatch.chdir(tmp_path)  # rejudge_all writes cache/judged under here
    judge = FakeJudge()
    monkeypatch.setattr(rejudge_cli, "build_default_providers", lambda **k: {"gpt-5.6-luna": judge})
    monkeypatch.setattr(
        rejudge_cli,
        "load_cached_records",
        lambda **k: [
            _sweep_record("gpt-5.6-luna", "q1", 1, "Ljubljana.", 1.0),
            _sweep_record("llama3.2", "q1", 1, "Zagreb", 0.4),
        ],
    )

    rc = rejudge_cli.main(["--judge", "gpt-5.6-luna", "--limit", "2"])

    assert rc == 0
    assert judge.calls == 2  # both cached answers re-graded
    assert (tmp_path / "cache" / "judged").exists()


@pytest.mark.mocked
def test_rejudge_cli_rejects_unknown_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        rejudge_cli, "build_default_providers", lambda **k: {"llama3.2": FakeJudge()}
    )
    # A judge with no provider (missing key) is a clean non-zero exit, not a crash.
    assert rejudge_cli.main(["--judge", "gpt-5.6-luna"]) == 2
