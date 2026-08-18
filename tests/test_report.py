"""The reporting step: render HTML off records, load records off the cache.

Both are pure/mocked ($0, no HTTP). The report is an artifact of the cached
data, so the test proves it renders the three tables and - the load-bearing part
- the verbatim prompt and response for every capture, since the raw output is
the ground truth the tables are computed from. The loader test proves the
provider-smoke captures are skipped so a smoke never pollutes the graded grid."""

from __future__ import annotations

import json

import pytest

from llm_benchmark.report import render_report
from llm_benchmark.runners.sweep import load_cached_records


def _record(model: str, item_id: str, rep: int, score: float, text: str) -> dict:
    return {
        "captured_at": "2026-08-11T00:00:00Z",
        "model": model,
        "item_id": item_id,
        "rep": rep,
        "prompt": "What is the capital of Slovenia?",
        "expected": "Ljubljana",
        "measurement": {
            "text": text,
            "tokens_in": 10,
            "tokens_out": 5,
            "latency_ms": 120.0,
            "ttft_ms": 40.0,
            "model": model,
        },
        "cost_usd": 0.0001,
        "judge": {
            "score": score,
            "passed": score >= 0.7,
            "reason": "graded ok",
            "judge_model": "llama3.2",
            "judge_tokens_in": 200,
            "judge_tokens_out": 30,
            "judge_cost_usd": 0.0,
        },
    }


@pytest.mark.mocked
def test_render_report_writes_tables_and_verbatim_trace(tmp_path):
    records = [
        _record("gpt-5.6-luna", "factual_001", 1, 1.0, "Ljubljana."),
        _record("gpt-5.6-luna", "factual_001", 2, 0.95, "The capital is Ljubljana."),
        _record("llama3.2", "factual_001", 1, 0.6, "I think it is Zagreb."),
        _record("llama3.2", "factual_001", 2, 0.9, "Ljubljana is the capital."),
    ]
    out = render_report(records, tmp_path / "r.html", title="Test report")
    html = out.read_text()

    # The three analysis tables are present.
    assert "Cost / latency / quality" in html
    assert "95% CI" in html
    assert "Paired significance" in html

    # Every capture's raw response text is in the trace, not just a summary.
    assert "I think it is Zagreb." in html
    assert "The capital is Ljubljana." in html
    assert html.count("<details>") == len(records)  # one trace block per capture

    # An answer preview sits in the always-visible summary of every trace, so a
    # wrong answer is scannable/findable without expanding the collapsed block.
    assert html.count('class="ans"') == len(records)
    assert "&ldquo;I think it is Zagreb.&rdquo;" in html  # the preview, in the summary

    # WHO graded is stated up front and on every trace (the judge model + its cost).
    assert "graded by llama3.2" in html
    assert "judge llama3.2" in html
    assert "free" in html  # the free local judge's $0 cost cell


@pytest.mark.mocked
def test_render_report_falls_back_to_run_judge_for_old_captures(tmp_path):
    # A capture with no stored judge model uses the run's judge model as the label.
    rec = _record("gpt-5.6-luna", "q1", 1, 1.0, "Ljubljana.")
    del rec["judge"]["judge_model"]
    out = render_report([rec], tmp_path / "r.html", title="t", judge_model="gpt-5.6-luna")
    html = out.read_text()
    assert "graded by gpt-5.6-luna" in html and "judge gpt-5.6-luna" in html


@pytest.mark.mocked
def test_render_report_handles_unpassed_model(tmp_path):
    # A model that never passes must render (cost-per-pass is n/a, not a crash).
    records = [_record("weak", "q1", 1, 0.3, "wrong"), _record("weak", "q1", 2, 0.4, "also wrong")]
    out = render_report(records, tmp_path / "r.html", title="t")
    assert "n/a" in out.read_text()


@pytest.mark.mocked
def test_load_cached_records_skips_smoke(tmp_path):
    good = _record("gpt-5.6-luna", "factual_001", 1, 1.0, "Ljubljana.")
    (tmp_path / "gpt-5.6-luna__factual_001__rep1.json").write_text(json.dumps(good))
    # A provider-smoke capture carries "smoke" in its name and must be ignored.
    (tmp_path / "gpt-5.6-luna__smoke-arithmetic.json").write_text(json.dumps(good))

    records = load_cached_records(cache_dir=tmp_path)
    assert len(records) == 1
    assert records[0]["item_id"] == "factual_001"
