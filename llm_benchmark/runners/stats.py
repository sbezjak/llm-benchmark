"""Benchmark-validity stats: the moves past a leaderboard-of-means.

Everything here is FREE and pure. It reads the already-cached sweep records (the
same `list[dict]` `run_sweep` returns and `summarize` rolls up) and never calls
a provider - so it can be recomputed off the artifact for $0, as many times as
you like. Three questions a production benchmark asks that a bare mean hides:

- **Are the score differences real, or noise on a small suite?** A 10-item suite
  run twice puts the paid models in a 0.960-0.975 band; a bare leaderboard would
  rank them as if that were signal. `bootstrap_ci` gives each model's mean a 95%
  interval, and `paired_score_diffs` asks the sharper question directly: resample
  the per-item *differences* between two models and see if 0 is inside the
  interval. If it is, the two are a statistical tie on this suite.
- **What is the TAIL latency, not the mean?** Production ships on p95, not the
  average - a reasoning model with a low mean can still hide a slow tail. Every
  raw latency is already in the cache; `percentile` reads it back.
- **What does a PASSED answer cost, not just any answer?** A cheap wrong answer
  is not cheap. `cost_per_pass` divides total spend by answers that cleared the
  judge threshold, not by all calls.

The judge scores this reads are discrete (the LLM-judge emits multiples of 0.05),
so a paired interval that *just* excludes 0 can be a single grading notch on one
item, not a durable quality gap - read the width and the mean_diff, not just the
sign. That caution is the point of reporting the interval instead of the mean.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# A record is one cached (model, item, rep) capture - the dicts run_sweep returns.
Record = dict


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile, `q` in [0, 1] (numpy's default method).
    p50 and p95 off the raw latencies are the tail read the mean cannot give."""
    if not xs:
        return math.nan
    ordered = sorted(xs)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def bootstrap_ci(
    xs: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval on the mean of `xs`.

    Resample `xs` with replacement `n_resamples` times, take each resample's
    mean, and cut the empirical `alpha/2` and `1-alpha/2` quantiles. Seeded so
    the reported interval is reproducible from the cache. A degenerate `xs` (all
    equal) collapses to a point interval, which is the honest answer."""
    if not xs:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    return (lo, hi)


def paired_bootstrap(
    diffs: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the mean of paired per-item differences. Returns
    `(mean_diff, lo, hi)`. When the interval contains 0 the two models are a
    tie on this suite; when it excludes 0, read the width before calling it a
    finding (see the module note on the judge's discrete scores)."""
    if not diffs:
        return (math.nan, math.nan, math.nan)
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return (mean_diff, lo, hi)


@dataclass(frozen=True)
class ModelStats:
    """Per-model validity roll-up: quality with an interval, tail latency, and
    the value-adjusted cost - the columns a bare `summarize` row omits."""

    model: str
    n: int
    mean_score: float
    score_ci_lo: float
    score_ci_hi: float
    lat_p50_ms: float
    lat_p95_ms: float
    passes: int
    total_cost_usd: float
    cost_per_pass_usd: float | None


def _by_model(records: list[Record]) -> dict[str, list[Record]]:
    out: dict[str, list[Record]] = {}
    for r in records:
        out.setdefault(r["model"], []).append(r)
    return out


def model_stats(records: list[Record], seed: int = 0) -> list[ModelStats]:
    """One `ModelStats` per model, ordered by mean score descending to match
    `summarize`. All math is off the cached records - no provider call."""
    stats: list[ModelStats] = []
    for model, rs in _by_model(records).items():
        scores = [r["judge"]["score"] for r in rs]
        lats = [r["measurement"]["latency_ms"] for r in rs]
        passes = sum(1 for r in rs if r["judge"]["passed"])
        total_cost = sum(r["cost_usd"] for r in rs)
        lo, hi = bootstrap_ci(scores, seed=seed)
        stats.append(
            ModelStats(
                model=model,
                n=len(rs),
                mean_score=sum(scores) / len(rs),
                score_ci_lo=lo,
                score_ci_hi=hi,
                lat_p50_ms=percentile(lats, 0.5),
                lat_p95_ms=percentile(lats, 0.95),
                passes=passes,
                total_cost_usd=total_cost,
                cost_per_pass_usd=(total_cost / passes) if passes else None,
            )
        )
    stats.sort(key=lambda s: s.mean_score, reverse=True)
    return stats


@dataclass(frozen=True)
class PairedDiff:
    """A baseline-minus-other paired score comparison over shared (item, rep)
    keys. `ci_excludes_zero` is a hint, not a verdict - weigh `mean_diff`."""

    model: str
    n_pairs: int
    mean_diff: float
    ci_lo: float
    ci_hi: float

    @property
    def ci_excludes_zero(self) -> bool:
        return not (self.ci_lo <= 0.0 <= self.ci_hi)


def paired_score_diffs(
    records: list[Record],
    baseline_model: str,
    seed: int = 0,
) -> list[PairedDiff]:
    """For each other model, pair its scores with `baseline_model`'s on the same
    (item, rep) and bootstrap the difference. Pairing controls for item
    difficulty - the sharper test than comparing two independent means."""
    by_model = _by_model(records)
    if baseline_model not in by_model:
        raise KeyError(f"baseline_model {baseline_model!r} not in records")

    def keyed(rs: list[Record]) -> dict[tuple[str, int], float]:
        return {(r["item_id"], r["rep"]): r["judge"]["score"] for r in rs}

    base = keyed(by_model[baseline_model])
    out: list[PairedDiff] = []
    for model, rs in by_model.items():
        if model == baseline_model:
            continue
        other = keyed(rs)
        keys = [k for k in base if k in other]
        diffs = [base[k] - other[k] for k in keys]
        mean_diff, lo, hi = paired_bootstrap(diffs, seed=seed)
        out.append(PairedDiff(model, len(keys), mean_diff, lo, hi))
    out.sort(key=lambda d: d.mean_diff)
    return out


def format_stats_table(stats: list[ModelStats]) -> str:
    """Fixed-width validity table: score with 95% CI, p50/p95 latency, and cost
    per passed answer side by side - the leaderboard with its error bars on."""
    header = (
        f"{'model':<28} {'n':>3} {'score':>6} {'95% CI':>16} "
        f"{'p50_ms':>8} {'p95_ms':>8} {'pass':>6} {'$/pass':>10}"
    )
    lines = [header, "-" * len(header)]
    for s in stats:
        ci = f"[{s.score_ci_lo:.3f},{s.score_ci_hi:.3f}]"
        per_pass = f"{s.cost_per_pass_usd:.6f}" if s.cost_per_pass_usd is not None else "n/a"
        lines.append(
            f"{s.model:<28} {s.n:>3} {s.mean_score:>6.3f} {ci:>16} "
            f"{s.lat_p50_ms:>8.0f} {s.lat_p95_ms:>8.0f} {s.passes:>6} {per_pass:>10}"
        )
    return "\n".join(lines)
