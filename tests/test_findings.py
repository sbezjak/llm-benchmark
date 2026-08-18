"""Aggregator: pure math off fabricated caches, so $0 and mocked.

No provider, no HTTP, no file read (`build_findings` takes the item->suite map
injected). The load-bearing checks: the paid arm is produced by swapping each
record's judge and re-running the SAME stats, so a judge that disagrees on the
hard suite must show up as a different paid-arm ranking there; and the candidate
findings (token efficiency, TTFT, prefix opportunity) read straight off the
measurements. A tiny fixture stands in for the 200-capture cache."""

from __future__ import annotations

import pytest

from llm_benchmark.runners.findings import (
    FREE_JUDGE,
    PAID_JUDGE,
    build_findings,
    build_pairwise,
    pairwise_view,
    prefix_opportunity,
    separation_analysis,
    token_efficiency,
    ttft_summary,
    with_judge,
)


def _rec(model, item, rep, *, score, prompt="q", lat=100.0, ttft=5.0, tin=10, tout=20, cost=0.001):
    """One capture dict, the fields findings reads populated. `score` is the
    FREE-local judge's verdict embedded in the record."""
    return {
        "model": model,
        "item_id": item,
        "rep": rep,
        "prompt": prompt,
        "expected": "e",
        "measurement": {
            "text": "a",
            "tokens_in": tin,
            "tokens_out": tout,
            "latency_ms": lat,
            "ttft_ms": ttft,
        },
        "cost_usd": cost,
        "judge": {"score": score, "passed": score >= 0.7, "reason": "free"},
    }


def _verdict(score):
    return {"score": score, "passed": score >= 0.7, "reason": "paid"}


# Two models, two suites (one easy item, one hard item), two reps each.
SMAP = {"easy_1": "easy", "hard_1": "adversarial"}


def _fixture_cache():
    records = []
    for rep in (1, 2):
        # strong: aces both suites under either judge
        records.append(_rec("strong", "easy_1", rep, score=1.0, prompt="Ep", tin=10, ttft=100.0))
        records.append(_rec("strong", "hard_1", rep, score=1.0, prompt="Hp", tin=10, ttft=100.0))
        # weak: the FREE judge over-rates it on the HARD item (0.9), the paid
        # judge docks it (0.3). On easy both judges agree it's fine.
        records.append(_rec("weak", "easy_1", rep, score=0.9, prompt="Ep", tin=40, ttft=500.0))
        records.append(_rec("weak", "hard_1", rep, score=0.9, prompt="Hp", tin=40, ttft=500.0))
    # paid verdicts: agree on easy + on strong; disagree on weak's HARD answers.
    judged = {}
    for rep in (1, 2):
        judged[("strong", "easy_1", rep)] = _verdict(1.0)
        judged[("strong", "hard_1", rep)] = _verdict(1.0)
        judged[("weak", "easy_1", rep)] = _verdict(0.9)
        judged[("weak", "hard_1", rep)] = _verdict(0.3)  # the reliable judge separates it
    return records, judged


@pytest.mark.mocked
def test_with_judge_swaps_and_drops_missing():
    records = [_rec("m", "easy_1", 1, score=0.9)]
    swapped = with_judge(records, {("m", "easy_1", 1): _verdict(0.3)})
    assert swapped[0]["judge"]["score"] == 0.3
    assert records[0]["judge"]["score"] == 0.9  # original not mutated
    # a record with no paid verdict is dropped, not silently mispaired
    assert with_judge(records, {}) == []


@pytest.mark.mocked
def test_build_findings_has_both_suites_and_both_judges():
    records, judged = _fixture_cache()
    f = build_findings(records, judged, SMAP, paid_judge_model="test-judge")
    assert set(f["suites"]) == {"easy", "adversarial"}
    assert f["judges"] == [FREE_JUDGE, PAID_JUDGE]
    assert f["headline_judge"] == PAID_JUDGE
    for suite in ("easy", "adversarial"):
        for judge in (FREE_JUDGE, PAID_JUDGE):
            assert f["views"][suite][judge]["models"], f"{suite}/{judge} empty"


@pytest.mark.mocked
def test_paid_judge_separates_weak_only_on_hard_suite():
    """The capstone: the free judge hides weak's hard-suite gap; the paid judge
    exposes it. Easy stays a tie under both."""
    records, judged = _fixture_cache()
    f = build_findings(records, judged, SMAP, paid_judge_model="test-judge")

    def score(suite, judge, model):
        row = next(r for r in f["views"][suite][judge]["models"] if r["model"] == model)
        return row["mean_score"]

    # HARD: free judge rates weak 0.9 (looks fine); paid docks it to 0.3.
    assert score("adversarial", FREE_JUDGE, "weak") == pytest.approx(0.9)
    assert score("adversarial", PAID_JUDGE, "weak") == pytest.approx(0.3)
    # EASY: both judges agree weak is fine - no separation there.
    assert score("easy", FREE_JUDGE, "weak") == pytest.approx(0.9)
    assert score("easy", PAID_JUDGE, "weak") == pytest.approx(0.9)

    # Under the paid judge on hard, the strong model tops and the paired diff
    # against weak is a real gap (CI excludes 0); under the free judge it isn't.
    hard_paid = f["views"]["adversarial"][PAID_JUDGE]["paired"]
    assert hard_paid["baseline"] == "strong"
    weak_diff = next(d for d in hard_paid["diffs"] if d["model"] == "weak")
    assert weak_diff["mean_diff"] == pytest.approx(0.7)  # 1.0 - 0.3
    assert not (weak_diff["ci_lo"] <= 0.0 <= weak_diff["ci_hi"])  # gap, not tie


@pytest.mark.mocked
def test_separation_flags_identical_models_as_unseparable():
    """Two models that score identically get obs_needed=None (no rep count
    separates them); a model with a real, consistent gap gets a finite reps
    number. This is the 'the tie is a decision, not undersampling' calc."""
    records = []
    for i in range(6):
        records.append(_rec("A", f"q{i}", 1, score=1.0))
        records.append(_rec("B", f"q{i}", 1, score=1.0))  # identical to A
        records.append(_rec("C", f"q{i}", 1, score=0.8))  # consistently lower
        records.append(_rec("free", f"q{i}", 1, score=0.5, cost=0.0))
    sep = separation_analysis(records, paid_models=["A", "B", "C"], baseline="A")
    assert sep["n_items"] == 6
    # A/B are identical -> exactly one unseparable pair
    ab = next(p for p in sep["paid_pairs"] if set(p["pair"]) == {"A", "B"})
    assert ab["obs_needed"] is None and ab["reps_needed"] is None
    assert sep["identical_pairs"] == 1
    # A/C differ by a real 0.2 -> finite, small reps needed
    ac = next(p for p in sep["paid_pairs"] if set(p["pair"]) == {"A", "C"})
    assert ac["obs_needed"] is not None
    assert ac["reps_needed"] == pytest.approx(0.0, abs=1.0)  # ~0 spread -> tiny n
    # contrast against the weakest non-paid model is reported
    assert sep["contrast_weakest"]["pair"] == ["A", "free"]


@pytest.mark.mocked
def test_token_efficiency_orders_cheapest_input_first():
    records, _ = _fixture_cache()
    rows = token_efficiency(records)
    assert [r["model"] for r in rows] == ["strong", "weak"]  # 10 in-tokens < 40
    assert rows[0]["mean_tokens_in"] == pytest.approx(10.0)


@pytest.mark.mocked
def test_ttft_summary_reports_and_orders():
    records, _ = _fixture_cache()
    rows = ttft_summary(records)
    assert [r["model"] for r in rows] == ["strong", "weak"]  # 100ms < 500ms
    assert rows[0]["mean_ttft_ms"] == pytest.approx(100.0)
    assert rows[0]["n_ttft"] == 4


@pytest.mark.mocked
def test_ttft_summary_handles_missing_ttft():
    recs = [_rec("m", "easy_1", 1, score=1.0, ttft=None)]
    (row,) = ttft_summary(recs)
    assert row["n_ttft"] == 0
    assert row["mean_ttft_ms"] is None
    assert row["p50_ttft_ms"] is None


@pytest.mark.mocked
def test_prefix_opportunity_reports_no_shared_prefix():
    # distinct standalone prompts -> no common prefix -> caching does not apply
    recs = [
        _rec("m", "easy_1", 1, score=1.0, prompt="What is the capital of Peru?"),
        _rec("m", "hard_1", 1, score=1.0, prompt="A bat and a ball cost $1.10..."),
    ]
    p = prefix_opportunity(recs)
    assert p["distinct_prompts"] == 2
    assert p["common_prefix_chars"] == 0
    assert p["applies"] is False


def _cmp(suite, item, m_a, m_b, outcome, winner=None, o1=None, o2=None):
    """One cached pairwise comparison dict (the shape `run_pairwise` writes).
    `o1`/`o2` are the per-order winners (needed for the first-answer-bias read)."""
    return {
        "suite": suite,
        "item_id": item,
        "model_a": m_a,
        "model_b": m_b,
        "outcome": outcome,  # win | genuine_tie | flip
        "winner": winner,
        "order1_winner": o1,
        "order2_winner": o2,
    }


@pytest.mark.mocked
def test_pairwise_view_standings_flip_rate_and_tau():
    """standings rank by win rate, flip_rate counts the position-bias flips, and
    kendall_tau measures agreement with the injected absolute order - all reused
    from runners.pairwise, this only packages them."""
    models = ["A", "B", "C"]
    # A beats B and C (2 wins); B beats C (1 win); one A-vs-C game flips.
    comps = [
        _cmp("easy", "q1", "A", "B", "win", "A"),
        _cmp("easy", "q1", "A", "C", "win", "A"),
        _cmp("easy", "q1", "B", "C", "win", "B"),
        _cmp("easy", "q2", "A", "C", "flip"),
    ]
    view = pairwise_view(comps, models, absolute_order=["A", "B", "C"])
    assert view["pairwise_order"] == ["A", "B", "C"]  # A top by win rate
    assert view["flip_rate"] == pytest.approx(0.25)  # 1 of 4 flipped
    # pairwise order equals the absolute order here -> perfect agreement
    assert view["kendall_tau"] == pytest.approx(1.0)
    top = view["standings"][0]
    assert top["model"] == "A" and top["wins"] == 2 and top["flips"] == 1


@pytest.mark.mocked
def test_pairwise_view_first_answer_bias():
    """Of the flips, the fraction where the FIRST-shown answer (slot A) won BOTH
    times - the sharpest position-bias number. Two flips: one is pure slot-A bias
    (order1->A, order2->B, i.e. first-shown won both), one is not."""
    models = ["A", "B", "C"]
    comps = [
        # flip where the first-shown answer won both orders (A led then won; B led then won)
        _cmp("easy", "q1", "A", "B", "flip", o1="A", o2="B"),
        # a flip that is NOT first-answer bias (A won regardless of slot)
        _cmp("easy", "q2", "A", "C", "flip", o1="A", o2="A"),
    ]
    view = pairwise_view(comps, models, absolute_order=[])
    assert view["n_flips"] == 2
    assert view["first_answer_wins"] == 1
    assert view["first_answer_bias"] == pytest.approx(0.5)


@pytest.mark.mocked
def test_pairwise_view_disagrees_with_absolute_order():
    """When the head-to-head winner is the absolute LOSER, tau goes negative -
    the 'absolute and pairwise measure different things' signal."""
    models = ["A", "B"]
    comps = [_cmp("easy", "q1", "A", "B", "win", "B")]  # B wins head-to-head
    view = pairwise_view(comps, models, absolute_order=["A", "B"])
    assert view["pairwise_order"] == ["B", "A"]  # reversed vs absolute
    assert view["kendall_tau"] == pytest.approx(-1.0)


@pytest.mark.mocked
def test_pairwise_view_tau_is_none_without_absolute_order():
    view = pairwise_view([], ["A", "B"], absolute_order=[])
    assert view["kendall_tau"] is None


@pytest.mark.mocked
def test_build_pairwise_splits_by_suite_and_anchors_on_paid_absolute():
    """The block has one view per (suite, judge); each judge's comparisons are
    filtered to the suite, and every view's tau anchors on the PAID absolute
    ranking already in `views`."""
    models = ["A", "B"]
    views = {
        "easy": {PAID_JUDGE: {"models": [{"model": "A"}, {"model": "B"}]}},
        "adversarial": {PAID_JUDGE: {"models": [{"model": "B"}, {"model": "A"}]}},
    }
    comparisons_by_label = {
        FREE_JUDGE: [
            _cmp("easy", "q1", "A", "B", "win", "A"),
            _cmp("adversarial", "q1", "A", "B", "flip"),
        ],
        PAID_JUDGE: [_cmp("easy", "q1", "A", "B", "win", "B")],
    }
    block = build_pairwise(
        comparisons_by_label,
        models,
        views,
        suites=["easy", "adversarial"],
        judge_models={FREE_JUDGE: "llama3.2", PAID_JUDGE: "gpt-5.6-luna"},
    )
    assert block["anchor"] == PAID_JUDGE
    # easy free arm: A wins -> agrees with easy paid-absolute order [A, B] -> tau 1
    assert block["views"]["easy"][FREE_JUDGE]["kendall_tau"] == pytest.approx(1.0)
    # easy paid arm: B wins -> disagrees with the same absolute order -> tau -1
    assert block["views"]["easy"][PAID_JUDGE]["kendall_tau"] == pytest.approx(-1.0)
    # the adversarial free comparison was NOT leaked into the easy view
    assert block["views"]["easy"][FREE_JUDGE]["standings"][0]["flips"] == 0
    # the adversarial view saw only its own (flip) comparison
    assert block["views"]["adversarial"][FREE_JUDGE]["flip_rate"] == pytest.approx(1.0)


@pytest.mark.mocked
def test_build_findings_pairwise_null_when_no_arms_and_present_when_given():
    records, judged = _fixture_cache()
    # no pairwise arms -> the block is null, not fabricated
    f0 = build_findings(records, judged, SMAP, paid_judge_model="test-judge")
    assert f0["pairwise"] is None
    # with arms -> a block keyed by suite then judge label
    comparisons_by_label = {
        FREE_JUDGE: [_cmp("easy", "easy_1", "strong", "weak", "win", "strong")],
        PAID_JUDGE: [_cmp("easy", "easy_1", "strong", "weak", "win", "strong")],
    }
    f1 = build_findings(
        records,
        judged,
        SMAP,
        paid_judge_model="test-judge",
        pairwise_by_label=comparisons_by_label,
    )
    assert set(f1["pairwise"]["views"]) == {"easy", "adversarial"}
    assert set(f1["pairwise"]["views"]["easy"]) == {FREE_JUDGE, PAID_JUDGE}


@pytest.mark.mocked
def test_prefix_opportunity_detects_a_shared_prefix():
    recs = [
        _rec("m", "easy_1", 1, score=1.0, prompt="SYSTEM: be terse. Q: capital of Peru?"),
        _rec("m", "hard_1", 1, score=1.0, prompt="SYSTEM: be terse. Q: bat and ball?"),
    ]
    p = prefix_opportunity(recs)
    assert p["common_prefix_chars"] > 0
    assert p["applies"] is True


@pytest.mark.mocked
def test_batch_lane_summary_reads_evidence_and_skips_when_absent(tmp_path):
    import json

    from llm_benchmark.runners.findings import batch_lane_summary

    assert batch_lane_summary(tmp_path / "missing.json") is None  # graceful skip
    ev = tmp_path / "batch-lane-captures.json"
    ev.write_text(
        json.dumps(
            {
                "model": "claude-haiku-4-5-20251001",
                "batch_id": "msgbatch_test",
                "comparison_per_capture": {
                    "interactive_streamed": {
                        "n": 20,
                        "cost_per_capture_usd": 0.0005,
                        "mean_latency_ms": 1600.0,
                        "mean_judge_score": 0.965,
                    },
                    "batch_half_price": {
                        "n": 20,
                        "cost_per_capture_usd": 0.00025,
                        "mean_latency_ms": 49000.0,
                        "mean_judge_score": 0.9625,
                    },
                },
            }
        )
    )
    s = batch_lane_summary(ev)
    assert s["batch_id"] == "msgbatch_test"
    assert s["cost_ratio_batch_over_interactive"] == pytest.approx(0.5)  # half price
    assert s["quality_delta_batch_minus_interactive"] == pytest.approx(-0.0025)


@pytest.mark.mocked
def test_semantic_cache_summary_reports_the_false_hit_trade(tmp_path):
    import json

    from llm_benchmark.runners.findings import semantic_cache_summary

    assert semantic_cache_summary(tmp_path / "missing.json") is None  # graceful skip
    ev = tmp_path / "semantic-cache-receipts.json"
    ev.write_text(
        json.dumps(
            {
                "embed_model": "nomic-embed-text",
                "seeds": [{}, {}, {}, {}],
                "probes": [{}] * 12,
                "threshold_sweep": [
                    {"threshold": 0.75, "true_hits": 8, "false_hits": 4, "misses": 0},
                    {"threshold": 0.95, "true_hits": 3, "false_hits": 2, "misses": 7},
                ],
            }
        )
    )
    s = semantic_cache_summary(ev)
    assert s["max_false_hits"] == 4
    # every threshold in this fixture still admits at least one false hit - the finding
    assert s["has_false_hits_at_every_threshold"] is True
