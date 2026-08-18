"""Benchmark-validity stats: pure math off cached records, so $0 and mocked.

No provider, no HTTP - these read fabricated capture dicts of the same shape
`run_sweep` writes, so the whole module is exercised without spend. The
load-bearing checks are that the bootstrap is reproducible (seeded) and that a
paired interval collapses to a point when the inputs are identical (a tie is
reported as a tie, not a spurious gap)."""

from __future__ import annotations

import math

import pytest

from llm_benchmark.runners.stats import (
    bootstrap_ci,
    model_stats,
    paired_score_diffs,
    percentile,
)


def _record(model: str, item_id: str, rep: int, score: float, latency_ms: float) -> dict:
    """One capture dict, only the fields the stats read populated."""
    return {
        "model": model,
        "item_id": item_id,
        "rep": rep,
        "measurement": {"latency_ms": latency_ms, "tokens_out": 10, "ttft_ms": 5.0},
        "cost_usd": 0.001,
        "judge": {"score": score, "passed": score >= 0.7, "reason": "test"},
    }


@pytest.mark.mocked
def test_percentile_interpolates_and_handles_edges():
    xs = [10.0, 20.0, 30.0, 40.0]
    assert percentile(xs, 0.0) == 10.0
    assert percentile(xs, 1.0) == 40.0
    assert percentile(xs, 0.5) == 25.0  # linear interp between 20 and 30
    assert math.isnan(percentile([], 0.5))
    assert percentile([7.0], 0.95) == 7.0  # single point


@pytest.mark.mocked
def test_bootstrap_ci_is_seeded_and_ordered():
    xs = [0.8, 0.85, 0.95, 1.0, 1.0, 0.9]
    lo1, hi1 = bootstrap_ci(xs, n_resamples=2000, seed=0)
    lo2, hi2 = bootstrap_ci(xs, n_resamples=2000, seed=0)
    assert (lo1, hi1) == (lo2, hi2)  # reproducible from the same seed
    assert lo1 <= sum(xs) / len(xs) <= hi1  # mean sits inside its own interval


@pytest.mark.mocked
def test_bootstrap_ci_of_constant_is_a_point():
    lo, hi = bootstrap_ci([0.95, 0.95, 0.95], n_resamples=500)
    assert lo == hi  # no variance -> the honest interval is a point
    assert lo == pytest.approx(0.95)


@pytest.mark.mocked
def test_model_stats_rolls_up_and_orders_by_score():
    records = [
        _record("good", "q1", 1, 1.0, 100.0),
        _record("good", "q1", 2, 0.9, 300.0),
        _record("weak", "q1", 1, 0.6, 100.0),  # below 0.7 pass threshold
        _record("weak", "q1", 2, 0.8, 100.0),
    ]
    stats = model_stats(records)
    assert [s.model for s in stats] == ["good", "weak"]  # sorted by mean score desc

    good = stats[0]
    assert good.n == 2
    assert good.passes == 2
    assert good.mean_score == pytest.approx(0.95)
    # p95 near the top of the latency spread, above the median.
    assert good.lat_p95_ms > good.lat_p50_ms
    # Both good answers passed, so cost per pass == total / 2.
    assert good.cost_per_pass_usd == pytest.approx(good.total_cost_usd / 2)

    weak = stats[1]
    assert weak.passes == 1  # only the 0.8 cleared threshold
    assert weak.cost_per_pass_usd == pytest.approx(weak.total_cost_usd / 1)


@pytest.mark.mocked
def test_cost_per_pass_is_none_when_nothing_passes():
    records = [_record("bad", "q1", 1, 0.3, 100.0), _record("bad", "q1", 2, 0.4, 100.0)]
    (bad,) = model_stats(records)
    assert bad.passes == 0
    assert bad.cost_per_pass_usd is None  # undefined, not zero - no passed answer to price


@pytest.mark.mocked
def test_paired_diffs_pair_on_item_and_rep():
    # Identical scores -> mean diff 0 and the interval contains 0 (a tie).
    records = [
        _record("a", "q1", 1, 0.9, 100.0),
        _record("a", "q2", 1, 1.0, 100.0),
        _record("b", "q1", 1, 0.9, 100.0),
        _record("b", "q2", 1, 1.0, 100.0),
    ]
    (diff,) = paired_score_diffs(records, baseline_model="a")
    assert diff.model == "b"
    assert diff.n_pairs == 2
    assert diff.mean_diff == pytest.approx(0.0)
    assert not diff.ci_excludes_zero  # a real tie reads as a tie


@pytest.mark.mocked
def test_paired_diffs_detect_a_consistent_gap():
    # a beats b by 0.1 on every shared item -> interval should exclude 0.
    records = []
    for i in range(6):
        records.append(_record("a", f"q{i}", 1, 1.0, 100.0))
        records.append(_record("b", f"q{i}", 1, 0.9, 100.0))
    (diff,) = paired_score_diffs(records, baseline_model="a")
    assert diff.mean_diff == pytest.approx(0.1)
    assert diff.ci_excludes_zero


@pytest.mark.mocked
def test_paired_diffs_reject_unknown_baseline():
    with pytest.raises(KeyError):
        paired_score_diffs([_record("a", "q1", 1, 1.0, 100.0)], baseline_model="nope")
