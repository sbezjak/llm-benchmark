"""Aggregate the cached captures into `findings.json` - the analysis step.

This is the compute half of the compute/render seam (mirrors how
`runners/sweep.py` writes captures and `report.py` renders them): a pure
aggregator reads the two on-disk caches and the two golden sets and emits one
JSON artifact; a separate renderer (`report_dashboard.py`) turns that JSON into
the dashboard. Splitting them means the aggregation gets a $0 pure unit test and
the dashboard regenerates off the JSON for $0.

Everything here is FREE and pure - it reads `cache/` (the sweep's own free-local
judge, embedded in each record's `judge` field) and `cache/judged/` (the paid
gpt re-judge, a sibling verdict per answer) and never calls a model. The one
non-obvious reuse: the quality math (bootstrap CI, paired significance, tail
latency, cost-per-passed) already lives in `stats.py`, but it reads `r["judge"]`
- the free arm. To get the paid arm we swap each record's `judge` for its paid
verdict and feed the SAME functions through unchanged (`with_judge`). Cost and
latency are judge-independent and computed once.

Two axes structure the output: the SUITE (easy golden set vs the adversarial
set) and the JUDGE (free-local vs paid-gpt). The headline the numbers must
support is the capstone finding: on easy the two judges agree and the paid pack is a
tie, but on the hard suite only the reliable paid judge separates the weak model
out - the free judge compresses the gap away.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path

from llm_benchmark.dataset import load_golden_set
from llm_benchmark.runners.pairwise import (
    DEFAULT_PAIRWISE_DIR,
    flip_rate,
    kendall_tau,
    standings,
)
from llm_benchmark.runners.rejudge import DEFAULT_JUDGED_DIR
from llm_benchmark.runners.stats import model_stats, paired_score_diffs, percentile
from llm_benchmark.runners.sweep import DEFAULT_CACHE_DIR, load_cached_records, summarize

logger = logging.getLogger(__name__)

FREE_JUDGE = "free-local"
PAID_JUDGE = "paid-gpt"

# The pairwise lens was judged by two models, one per absolute-judge label:
# the free local Ollama baseline (llama3.2) and the paid mid-pack judge
# (gpt-5.6-luna, also the paid absolute judge - so the paid pairwise ranking is
# comparable to the paid absolute one). Each arm's verdicts live in its own
# `cache/pairwise/<model>/` tree.
FREE_PAIRWISE_MODEL = "llama3.2"
PAID_PAIRWISE_MODEL = "gpt-5.6-luna"

# The two suites, each backed by its golden set. `dataset.load_golden_set`
# validates them, so a typo in an id fails loud at load, not silently here.
SUITE_FILES = {
    "easy": Path("data") / "golden_set.yaml",
    "adversarial": Path("data") / "adversarial_set.yaml",
}


def suite_map(suite_files: dict[str, Path] = SUITE_FILES) -> dict[str, str]:
    """`item_id -> suite name`, built from the golden sets (not guessed from id
    prefixes). An id in two suites is a dataset bug and raises."""
    out: dict[str, str] = {}
    for suite, path in suite_files.items():
        for item in load_golden_set(path):
            if item.id in out:
                raise ValueError(f"item {item.id!r} appears in two suites")
            out[item.id] = suite
    return out


def load_judged_map(
    judge_model: str, judged_dir: Path = DEFAULT_JUDGED_DIR
) -> dict[tuple[str, str, int], dict]:
    """Read the paid re-judge tree into a `(answer_model, item_id, rep) -> verdict`
    map. The verdict dict is the `judge` block (`score`/`passed`/`reason`) that
    `stats.py` expects, so it drops straight into a swapped record."""
    out: dict[tuple[str, str, int], dict] = {}
    for path in sorted(judged_dir.glob(f"{judge_model}__*.json")):
        d = json.loads(path.read_text())
        out[(d["answer_model"], d["item_id"], int(d["rep"]))] = d["judge"]
    return out


def with_judge(records: list[dict], judged_map: dict[tuple[str, str, int], dict]) -> list[dict]:
    """Return copies of `records` with each `judge` block replaced by its paid
    verdict, so the free-arm stats functions produce the paid arm unchanged. A
    record with no paid verdict is dropped with a warning - the caller has
    already verified freshness, so a miss means a genuinely ungraded answer, not
    a silent wrong pairing."""
    out: list[dict] = []
    for r in records:
        key = (r["model"], r["item_id"], int(r["rep"]))
        verdict = judged_map.get(key)
        if verdict is None:
            logger.warning("no paid verdict for %s - dropping from paid arm", key)
            continue
        swapped = dict(r)
        swapped["judge"] = verdict
        out.append(swapped)
    return out


def _model_rows(records: list[dict]) -> list[dict]:
    """One combined per-model row: the judge-dependent quality stats (score with
    a bootstrap CI, passes, cost-per-passed) from `model_stats`, plus the
    judge-independent cost / latency / token means. Ordered by score desc, the
    same ranking the report uses."""
    stats = {s.model: s for s in model_stats(records)}
    summaries = {s.model: s for s in summarize(records)}
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    rows: list[dict] = []
    for model, rs in by_model.items():
        s = stats[model]
        summ = summaries[model]
        ins = [r["measurement"]["tokens_in"] for r in rs]
        rows.append(
            {
                "model": model,
                "n": s.n,
                # quality (judge-dependent)
                "mean_score": s.mean_score,
                "score_ci_lo": s.score_ci_lo,
                "score_ci_hi": s.score_ci_hi,
                "passes": s.passes,
                "pass_rate": summ.pass_rate,
                "cost_per_pass_usd": s.cost_per_pass_usd,
                # cost / latency / tokens (judge-independent)
                "mean_cost_usd": summ.mean_cost_usd,
                "total_cost_usd": s.total_cost_usd,
                "mean_latency_ms": summ.mean_latency_ms,
                "lat_p50_ms": s.lat_p50_ms,
                "lat_p95_ms": s.lat_p95_ms,
                "mean_ttft_ms": summ.mean_ttft_ms,
                "mean_tokens_in": sum(ins) / len(ins),
                "mean_tokens_out": summ.mean_tokens_out,
            }
        )
    rows.sort(key=lambda r: r["mean_score"], reverse=True)
    return rows


def _view(records: list[dict]) -> dict:
    """A (suite, judge) view: the per-model table plus the paired significance
    read against the top-ranked model - is a score gap real or noise?"""
    rows = _model_rows(records)
    baseline = rows[0]["model"] if rows else ""
    diffs = paired_score_diffs(records, baseline_model=baseline) if rows else []
    return {
        "models": rows,
        "paired": {"baseline": baseline, "diffs": [asdict(d) for d in diffs]},
    }


def token_efficiency(records: list[dict]) -> list[dict]:
    """Per-model mean input/output tokens across the whole grid - the tokenizer
    read. Identical questions cost different input tokens per model (a provider
    property, from its own usage object), so this is a real cost lever, not an
    artifact of the item mix. Ordered by input tokens asc (cheapest first)."""
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    rows = []
    for model, rs in by_model.items():
        ins = [r["measurement"]["tokens_in"] for r in rs]
        outs = [r["measurement"]["tokens_out"] for r in rs]
        rows.append(
            {
                "model": model,
                "mean_tokens_in": sum(ins) / len(ins),
                "mean_tokens_out": sum(outs) / len(outs),
            }
        )
    rows.sort(key=lambda r: r["mean_tokens_in"])
    return rows


def ttft_summary(records: list[dict]) -> list[dict]:
    """Per-model time-to-first-token (mean + p50), the felt-latency read the
    total-latency mean hides. Only captures that recorded a TTFT count; a model
    with none reports null."""
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    rows = []
    for model, rs in by_model.items():
        tt = [r["measurement"]["ttft_ms"] for r in rs if r["measurement"]["ttft_ms"] is not None]
        rows.append(
            {
                "model": model,
                "n_ttft": len(tt),
                "mean_ttft_ms": (sum(tt) / len(tt)) if tt else None,
                "p50_ttft_ms": percentile(tt, 0.5) if tt else None,
            }
        )
    rows.sort(key=lambda r: (r["mean_ttft_ms"] is None, r["mean_ttft_ms"] or 0.0))
    return rows


def prefix_opportunity(records: list[dict]) -> dict:
    """Is there a shared prompt prefix a provider's prefix cache could reuse?
    The honest answer for this suite is no: the items are standalone questions
    with no common preamble. Reporting the measured common-prefix length (rather
    than omitting the axis) is the finding - prefix caching pays off for a
    system-prompt-heavy workload, which this benchmark deliberately isn't."""
    import os

    prompts = sorted({r["prompt"] for r in records})
    lcp = os.path.commonprefix(prompts) if prompts else ""
    return {
        "distinct_prompts": len(prompts),
        "common_prefix_chars": len(lcp),
        "applies": len(lcp) > 0,
    }


# The two cost-lever candidates are self-contained experiments with their own
# committed evidence (not part of the sweep cache), so they are summarized FROM
# that evidence rather than recomputed here - one number per lever, plus the
# receipt that lets a reader open the full proof. Missing file -> candidate
# omitted (same graceful-skip contract as the pairwise agreement).
BATCH_EVIDENCE = Path("evidence") / "batch-lane-captures.json"
SEMANTIC_EVIDENCE = Path("evidence") / "semantic-cache-receipts.json"


def batch_lane_summary(path: Path = BATCH_EVIDENCE) -> dict | None:
    """The batch-lane cost/quality/latency trade, off the committed captures:
    ~half price for minutes of turnaround at tied quality. `None` if the batch
    evidence has not been generated (`python -m llm_benchmark.runners.batch_sweep`)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    c = d["comparison_per_capture"]
    inter, batch = c["interactive_streamed"], c["batch_half_price"]
    return {
        "model": d["model"],
        "batch_id": d.get("batch_id"),
        "n_cells": batch["n"],
        "interactive_cost_per_capture_usd": inter["cost_per_capture_usd"],
        "batch_cost_per_capture_usd": batch["cost_per_capture_usd"],
        "cost_ratio_batch_over_interactive": batch["cost_per_capture_usd"]
        / inter["cost_per_capture_usd"],
        "interactive_mean_latency_ms": inter["mean_latency_ms"],
        "batch_turnaround_ms": batch["mean_latency_ms"],
        "interactive_mean_judge_score": inter["mean_judge_score"],
        "batch_mean_judge_score": batch["mean_judge_score"],
        "quality_delta_batch_minus_interactive": batch["mean_judge_score"]
        - inter["mean_judge_score"],
    }


def semantic_cache_summary(path: Path = SEMANTIC_EVIDENCE) -> dict | None:
    """The semantic-cache false-hit trade, off the committed receipts: the
    threshold sweep (hit-rate vs false-hits) and the count of false hits at the
    focus threshold. `None` if the receipts have not been generated
    (`python -m llm_benchmark.semantic_cache_run`)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    sweep = d["threshold_sweep"]
    # The false-hit finding is that no threshold both keeps paraphrases AND
    # rejects the trap: report the sweep and the worst false-hit count.
    max_false = max((row["false_hits"] for row in sweep), default=0)
    return {
        "embed_model": d["embed_model"],
        "n_seeds": len(d["seeds"]),
        "n_probes": len(d["probes"]),
        "threshold_sweep": sweep,
        "max_false_hits": max_false,
        "has_false_hits_at_every_threshold": all(row["false_hits"] > 0 for row in sweep),
    }


# Two-sided alpha=0.05, power=0.80 - the standard z-values for a paired power
# calc. The scores are discrete (0.05 steps), so treat the output as an
# order-of-magnitude "reps you'd need", not a precise count - the point is that
# separating the paid pack needs an impractical n, so the tie is a decision, not
# undersampling.
_Z_ALPHA = 1.959963985
_Z_POWER = 0.841621234


def _reps_for_pair(records: list[dict], m1: str, m2: str, n_items: int) -> dict:
    """Paired power calc for one model pair, off the cached per-item score
    differences: how many reps (at `n_items` items/suite) it would take to detect
    the observed gap at 95% confidence / 80% power. `None` obs_needed means the
    two models score identically here - no experiment of any size separates
    them."""
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for r in records:
        by_key.setdefault((r["item_id"], r["rep"]), {})[r["model"]] = r["judge"]["score"]
    diffs = [v[m1] - v[m2] for v in by_key.values() if m1 in v and m2 in v]
    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0
    sd = (sum((x - mean) ** 2 for x in diffs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    if abs(mean) < 1e-9:
        obs_needed: float | None = None  # identical - unseparable at any n
    else:
        obs_needed = ((_Z_ALPHA + _Z_POWER) * sd / abs(mean)) ** 2
    return {
        "pair": [m1, m2],
        "n_obs": n,
        "mean_diff": mean,
        "sd": sd,
        "obs_needed": obs_needed,
        "reps_needed": (obs_needed / n_items) if obs_needed is not None else None,
    }


def separation_analysis(records: list[dict], paid_models: list[str], baseline: str) -> dict:
    """The "tie is a decision, not undersampling" finding for one view. Runs the
    paired power calc across every pair of paid models and, as a contrast, the
    baseline-vs-weakest pair. `identical_pairs` (mean diff exactly 0) is the
    sharpest result - those are unseparable by any rep count."""
    n_items = len({r["item_id"] for r in records})
    present = [m for m in paid_models if any(r["model"] == m for r in records)]
    pairs = [
        _reps_for_pair(records, a, b, n_items)
        for i, a in enumerate(present)
        for b in present[i + 1 :]
    ]
    finite = [p["reps_needed"] for p in pairs if p["reps_needed"] is not None]
    weakest = min(
        ({r["model"] for r in records} - set(paid_models)),
        default=None,
    )
    contrast = (
        _reps_for_pair(records, baseline, weakest, n_items)
        if weakest and baseline and weakest != baseline
        else None
    )
    return {
        "n_items": n_items,
        "paid_pairs": pairs,
        "identical_pairs": sum(1 for p in pairs if p["obs_needed"] is None),
        "max_paid_reps_needed": max(finite) if finite else None,
        "min_paid_reps_needed": min(finite) if finite else None,
        "contrast_weakest": contrast,
    }


def load_pairwise_comparisons(
    judge_model: str, pairwise_dir: Path = DEFAULT_PAIRWISE_DIR
) -> list[dict]:
    """Read one pairwise judge arm (`cache/pairwise/<judge_model>/`) into the flat
    list of comparison dicts the aggregators consume. Each file already holds one
    resolved both-orders comparison ({suite, item_id, model_a, model_b, outcome,
    winner}); this only reads them back, never re-judges. An absent arm is an
    empty list, not an error - the free arm may exist without the paid one."""
    arm = pairwise_dir / judge_model
    return [json.loads(p.read_text()) for p in sorted(arm.glob("*.json"))]


def pairwise_view(comparisons: list[dict], models: list[str], absolute_order: list[str]) -> dict:
    """One (suite, judge) pairwise view: the head-to-head standings, the judge's
    flip rate (its position-bias number), and the rank agreement (kendall_tau)
    with the paid-gpt absolute ranking for the same suite. All three reuse the
    pairwise functions in `runners.pairwise` unchanged - this only aggregates and
    packages, it does not re-derive the head-to-head math."""
    rows = standings(models, comparisons)
    pairwise_order = [s.model for s in rows]
    # Of the flips, how many just picked whichever answer was shown FIRST (slot A)
    # BOTH times? A flip whose order1 winner is model_a AND order2 winner is model_b
    # is the judge rewarding position, not quality - the sharpest read of the bias.
    flips = [c for c in comparisons if c["outcome"] == "flip"]
    first_answer_wins = sum(
        1
        for c in flips
        if c["order1_winner"] == c["model_a"] and c["order2_winner"] == c["model_b"]
    )
    return {
        "standings": [asdict(s) for s in rows],
        "flip_rate": flip_rate(comparisons),
        "n_flips": len(flips),
        "first_answer_wins": first_answer_wins,
        "first_answer_bias": (first_answer_wins / len(flips)) if flips else 0.0,
        "absolute_order": absolute_order,
        "pairwise_order": pairwise_order,
        "kendall_tau": kendall_tau(pairwise_order, absolute_order) if absolute_order else None,
    }


def build_pairwise(
    comparisons_by_label: dict[str, list[dict]],
    models: list[str],
    views: dict[str, dict],
    suites: list[str],
    judge_models: dict[str, str],
) -> dict:
    """Assemble the pairwise block from the cached verdicts, one view per
    (suite, judge). Every view's kendall_tau is measured against the SAME anchor -
    the paid-gpt absolute ranking for that suite (already in `views`) - so the four
    views answer one question: does forced head-to-head choice agree with
    independent absolute scoring? (The finding: tau=0 on all four - it doesn't,
    and the raw read shows why.) Pure: the caller injects the cached comparisons."""
    out: dict[str, dict] = {}
    for suite in suites:
        absolute_order = [r["model"] for r in views[suite][PAID_JUDGE]["models"]]
        per_judge: dict[str, dict] = {}
        for label, comparisons in comparisons_by_label.items():
            suite_comps = [c for c in comparisons if c.get("suite") == suite]
            per_judge[label] = pairwise_view(suite_comps, models, absolute_order)
        out[suite] = per_judge
    return {"judge_models": judge_models, "anchor": PAID_JUDGE, "views": out}


def build_findings(
    records: list[dict],
    judged_map: dict[tuple[str, str, int], dict],
    smap: dict[str, str],
    *,
    paid_judge_model: str,
    pairwise_by_label: dict[str, list[dict]] | None = None,
) -> dict:
    """Assemble the full findings artifact from the cached records, the paid
    verdicts, and the item->suite map. Pure: no I/O, so a fixture cache drives
    the unit test. Every number is off the caches, never a re-call."""
    suites = list(SUITE_FILES)
    models = sorted({r["model"] for r in records})
    # The free baseline is the model that never costs anything (local Ollama);
    # the paid pack is everyone else - derived, not hardcoded.
    cost_by_model: dict[str, float] = {}
    for r in records:
        cost_by_model[r["model"]] = cost_by_model.get(r["model"], 0.0) + r["cost_usd"]
    paid_models = sorted(m for m in models if cost_by_model.get(m, 0.0) > 0.0)

    def in_suite(rs: list[dict], suite: str) -> list[dict]:
        return [r for r in rs if smap.get(r["item_id"]) == suite]

    paid_records = with_judge(records, judged_map)

    def one_view(rs: list[dict]) -> dict:
        v = _view(rs)
        v["separation"] = separation_analysis(rs, paid_models, v["paired"]["baseline"])
        return v

    views: dict[str, dict] = {}
    for suite in suites:
        views[suite] = {
            FREE_JUDGE: one_view(in_suite(records, suite)),
            PAID_JUDGE: one_view(in_suite(paid_records, suite)),
        }

    pairwise = None
    if pairwise_by_label:
        judge_models = {
            FREE_JUDGE: FREE_PAIRWISE_MODEL,
            PAID_JUDGE: paid_judge_model,
        }
        pairwise = build_pairwise(pairwise_by_label, models, views, suites, judge_models)

    total_answer_cost = sum(r["cost_usd"] for r in records)
    return {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ"),
        "suites": suites,
        "judges": [FREE_JUDGE, PAID_JUDGE],
        "headline_judge": PAID_JUDGE,
        "paid_judge_model": paid_judge_model,
        "models": models,
        "n_captures": len(records),
        "views": views,
        "pairwise": pairwise,
        "candidates": {
            "token_efficiency": token_efficiency(records),
            "ttft": ttft_summary(records),
            "prefix_caching": prefix_opportunity(records),
            "batch_lane": batch_lane_summary(),
            "semantic_cache": semantic_cache_summary(),
        },
        "spend": {"total_answer_cost_usd": total_answer_cost},
    }


DEFAULT_FINDINGS_PATH = Path("reports") / "findings.json"


def write_findings(
    out_path: Path = DEFAULT_FINDINGS_PATH,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    judged_dir: Path = DEFAULT_JUDGED_DIR,
    pairwise_dir: Path = DEFAULT_PAIRWISE_DIR,
    paid_judge_model: str = "gpt-5.6-luna",
) -> Path:
    """Read the caches, build the findings, and write `reports/findings.json`.
    The thin I/O wrapper around `build_findings` - the aggregator's entry point,
    a real job, not a test. Returns the path written."""
    records = load_cached_records(cache_dir)
    if not records:
        raise SystemExit(f"no cached captures under {cache_dir} - run the sweep first")
    judged_map = load_judged_map(paid_judge_model, judged_dir)
    smap = suite_map()
    # The two pairwise arms, read back off disk - free-local and paid-gpt.
    # An arm with no cache is an empty list, so findings still builds if the
    # pairwise lens hasn't been run; the block is then null rather than fabricated.
    pairwise_by_label = {
        FREE_JUDGE: load_pairwise_comparisons(FREE_PAIRWISE_MODEL, pairwise_dir),
        PAID_JUDGE: load_pairwise_comparisons(paid_judge_model, pairwise_dir),
    }
    findings = build_findings(
        records,
        judged_map,
        smap,
        paid_judge_model=paid_judge_model,
        pairwise_by_label=pairwise_by_label,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2))
    logger.info(
        "wrote %s: %d captures, %d paid verdicts, %d free + %d paid pairwise, suites=%s",
        out_path,
        len(records),
        len(judged_map),
        len(pairwise_by_label[FREE_JUDGE]),
        len(pairwise_by_label[PAID_JUDGE]),
        findings["suites"],
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    write_findings()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
