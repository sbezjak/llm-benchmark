"""The scoreboard renderer: pure/mocked ($0, no HTTP, no model).

The page is an artifact of `findings.json`, and it is deliberately NOT a
leaderboard: it leads with its own limits and surfaces only the two reads that
stay honest at a tiny sample size (quality intervals overlap -> don't rank; the
grader flips under an order swap -> validate the instrument). It is plain tables
with the real model ids in every row - no charts. The test feeds a small findings
dict and proves the render carries the scope/limits framing, both lessons as
tables, and no external asset a strict CSP would block - and that it does NOT
overclaim with a recommendation verdict. Numbers come off the dict, never a
re-call."""

from __future__ import annotations

import pytest

from llm_benchmark.report_dashboard import PAID_JUDGE, render_dashboard


def _model_row(model, score, ci, cost, *, ttft=500.0, p95=2000.0):
    return {
        "model": model,
        "n": 4,
        "mean_score": score,
        "score_ci_lo": ci[0],
        "score_ci_hi": ci[1],
        "passes": 4,
        "pass_rate": 1.0,
        "cost_per_pass_usd": cost or None,
        "mean_cost_usd": cost,
        "total_cost_usd": cost * 4,
        "mean_latency_ms": 900.0,
        "lat_p50_ms": 800.0,
        "lat_p95_ms": p95,
        "mean_ttft_ms": ttft,
        "mean_tokens_in": 30.0,
        "mean_tokens_out": 40.0,
    }


def _view(models_rows):
    return {
        "models": models_rows,
        "paired": {"baseline": models_rows[0]["model"], "diffs": []},
        "separation": {
            "n_items": 10,
            "paid_pairs": [],
            "identical_pairs": 3,
            "max_paid_reps_needed": 16.0,
            "min_paid_reps_needed": 16.0,
            "contrast_weakest": {
                "pair": ["gpt-5.6-luna", "llama3.2"],
                "reps_needed": 5.0,
                "mean_diff": 0.08,
            },
        },
    }


def _standing(model, win_rate, *, wins=6, ties=1, flips=2, elo=1000.0):
    return {
        "model": model,
        "games": 40,
        "wins": wins,
        "losses": 30,
        "genuine_ties": ties,
        "flips": flips,
        "win_rate": win_rate,
        "elo": elo,
    }


def _findings():
    gpt = "gpt-5.6-luna"
    llama = "llama3.2"
    models = [gpt, llama]
    rows = [_model_row(gpt, 1.0, (1.0, 1.0), 0.00009), _model_row(llama, 0.915, (0.82, 0.99), 0.0)]
    view = _view(rows)
    absolute_order = [gpt, llama]
    return {
        "generated": "2026-08-15",
        "suites": ["easy", "adversarial"],
        "judges": ["free-local", PAID_JUDGE],
        "headline_judge": PAID_JUDGE,
        "paid_judge_model": gpt,
        "models": models,
        "n_captures": 8,
        "views": {
            "easy": {"free-local": view, PAID_JUDGE: view},
            "adversarial": {"free-local": view, PAID_JUDGE: view},
        },
        "pairwise": {
            "judge_models": {"free-local": llama, PAID_JUDGE: gpt},
            "anchor": PAID_JUDGE,
            "views": {
                suite: {
                    "free-local": {
                        "standings": [_standing(llama, 0.53), _standing(gpt, 0.40)],
                        "flip_rate": 0.46,
                        "n_flips": 1,
                        "first_answer_wins": 1,  # both suites -> 2/2 = 100% cheap
                        "first_answer_bias": 1.0,
                        "absolute_order": absolute_order,
                        "pairwise_order": [llama, gpt],
                        "kendall_tau": 0.0,
                    },
                    PAID_JUDGE: {
                        "standings": [_standing(gpt, 0.55), _standing(llama, 0.42)],
                        "flip_rate": 0.21,
                        "n_flips": 1,
                        "first_answer_wins": 0,  # paid judge: not position-driven
                        "first_answer_bias": 0.0,
                        "absolute_order": absolute_order,
                        "pairwise_order": [gpt, llama],
                        "kendall_tau": 0.0,
                    },
                }
                for suite in ("easy", "adversarial")
            },
        },
        "candidates": {
            "token_efficiency": [
                {"model": gpt, "mean_tokens_in": 22.0, "mean_tokens_out": 40.0},
                {"model": llama, "mean_tokens_in": 57.0, "mean_tokens_out": 60.0},
            ],
            "ttft": [
                {"model": llama, "n_ttft": 4, "mean_ttft_ms": 487.0, "p50_ttft_ms": 480.0},
                {"model": gpt, "n_ttft": 4, "mean_ttft_ms": 1199.0, "p50_ttft_ms": 1180.0},
            ],
            "prefix_caching": {"distinct_prompts": 20, "common_prefix_chars": 0, "applies": False},
        },
        "spend": {"total_answer_cost_usd": 0.1319},
    }


@pytest.mark.mocked
def test_dashboard_leads_with_scope_and_two_honest_reads(tmp_path):
    out = render_dashboard(
        _findings(), tmp_path / "d.html", title="Test sketch", n_items=20, n_reps=2
    )
    doc = out.read_text()

    # scope / limits framing is present and up front, in plain words
    assert "a demo, not a verdict" in doc
    # the KPI stat tiles: number + label, n_captures pulled from the dict
    assert ">20</div>" in doc and "items" in doc and "reps" in doc and "models" in doc
    assert ">8</div>" in doc and "graded answers" in doc

    # technical section headings, no chatty full-sentence heads
    for heading in (
        "Judge reliability - position bias",
        "Self-preference",
        "Cost, latency, quality",
        "Limits",
    ):
        assert f"<h2>{heading}</h2>" in doc

    # judge-reliability table names both graders and carries the flip data
    assert "grader" in doc and "flip rate" in doc and "kept first-shown" in doc
    assert "46%" in doc  # fixture flip rate
    # computed first-answer-bias read (fixture: both flips kept the first-shown answer -> 100%)
    assert "100%" in doc.replace("<b>", "").replace("</b>", "")
    assert "was shown first" in doc
    assert "easy__factual_004__claude-haiku-4-5-20251001__claude-sonnet-5.json" in doc
    assert "evidence/judge-position-bias.md" in doc

    # self-preference: the cheap grader ranks its own pool; read references it
    assert "head-to-head win" in doc and "solo score" in doc
    assert "pool it competes in" in doc

    # cost/latency/quality: the real columns and the free local cost cell
    assert "$ / query" in doc and "lat p95" in doc
    assert "free" in doc  # the local model's cost cell
    assert "95% CI" in doc

    # honest limits + a realistic production section (not overclaiming)
    assert "No best model" in doc
    assert "panel of judges" in doc

    # every finding cites where to verify it
    assert "Check it" in doc
    assert "reports/findings.json" in doc

    # it does NOT overclaim: no confident recommendation verdict table
    assert "which model, when" not in doc.lower()
    assert "Default paid pick" not in doc

    # self-contained: no external asset a strict CSP would block
    for needle in ("http://", "https://", "<script", "src=", "@import"):
        assert needle not in doc


@pytest.mark.mocked
def test_dashboard_shows_exact_model_ids(tmp_path):
    # the actual version ids people pin, not friendly aliases
    out = render_dashboard(_findings(), tmp_path / "d.html", title="Test")
    doc = out.read_text()
    assert "gpt-5.6-luna" in doc and "llama3.2" in doc


@pytest.mark.mocked
def test_dashboard_spend_includes_judging(tmp_path):
    # when a spend breakdown is supplied, the total is answers PLUS judging - not
    # the answer-only figure - and the split is stated so it is not understated.
    spend = {
        "answer_generation_usd": 0.13,
        "solo_judge_gpt_usd": 0.026,
        "pairwise_judge_gpt_usd": 0.057,
        "total_usd": 0.213,
    }
    out = render_dashboard(_findings(), tmp_path / "d.html", title="Test", spend=spend)
    doc = out.read_text()
    assert "$0.21" in doc  # the honest total, not the $0.13 answer-only figure
    assert "plus the paid judging" in doc


@pytest.mark.mocked
def test_dashboard_spend_falls_back_to_answer_only(tmp_path):
    # with no breakdown, it uses the findings answer cost and says so plainly.
    out = render_dashboard(_findings(), tmp_path / "d.html", title="Test", spend=None)
    doc = out.read_text()
    assert "answer generation only" in doc
