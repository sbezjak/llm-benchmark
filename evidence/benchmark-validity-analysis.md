# Benchmark validity - reading the sweep past a leaderboard of means

The comparison table (`full-sweep-comparison-table.md`) ranks five models on a tight
quality band. A rank alone invites a false read: that the fraction of a point between
the top models is signal. This analysis asks whether it is, and adds two axes a mean
hides - the interval around each score, and the latency tail. Everything here is
computed from the **same 100 easy captures** under `cache/` - no new call, no spend.

> **Finding vs sample.** Non-deterministic system: a re-roll moves every number and
> can reorder the leaderboard. The FINDING is the *shape* that survives - the paid
> models are a statistical tie on quality (every paired interval includes 0), so the
> differentiator is cost and latency, not score. The exact scores, the ranking, and
> which model tops the table are **one run's sample (2026-08-17)**. This roll proves
> the point: the leaderboard order here is different from the previous roll's, but the
> "it's a tie" conclusion is the same - because the tie is what is real. Regenerate
> off the fresh cache into `reports/findings.json` (`views.easy.free-local`).

## The table with its error bars on (easy suite, free-local judge)

Quality is the LLM-judge score in [0, 1]; the 95% interval is a percentile bootstrap
of the mean. Latency is the median and the 95th percentile, not just the mean. Cost
is per *passed* answer (judge threshold 0.7), not per call.

| model | score | 95% CI | lat p50 | lat p95 | passes | $/passed |
|---|---|---|---|---|---|---|
| deepseek-v4-pro | 0.970 | [0.948, 0.988] | 1949 ms | 5987 ms | 20/20 | $0.000138 |
| claude-haiku-4-5-20251001 | 0.967 | [0.950, 0.982] | 979 ms | 3379 ms | 20/20 | $0.000537 |
| gpt-5.6-luna | 0.962 | [0.920, 0.992] | 840 ms | 3484 ms | 19/20 | $0.000086 |
| claude-sonnet-5 | 0.937 | [0.897, 0.967] | 3471 ms | 7302 ms | 19/20 | $0.002554 |
| llama3.2 (local) | 0.922 | [0.872, 0.965] | 3211 ms | **26081 ms** | 19/20 | $0.000000 |

Paired significance - each pair's per-item score difference on the same item and rep,
bootstrapped. A CI that contains 0 is a statistical tie on this suite.

| pair | mean diff | 95% CI | read |
|---|---|---|---|
| all 6 paid-vs-paid pairs | -0.033 to +0.030 | every interval includes 0 | tie |
| deepseek-v4-pro minus llama3.2 (local) | +0.048 | [+0.004, +0.091] | gap, and it barely clears 0 |

## What the analysis says (at the level that survives)

**The paid pack is a statistical tie on quality - so the differentiator is cost and
latency, not score.** All six paid-vs-paid intervals overlap 0. The leaderboard
ordering is noise on ten items run twice; the honest headline is not "deepseek is
best" (it tops this roll; gpt topped last roll) but "the paid models are
indistinguishable on quality here, so choose on price and speed." That is a *stronger*
claim than a fake ranking, and the cost/latency columns then decide it - in favour of
the cheapest, fastest paid model (gpt-5.6-luna: $0.000086/pass, p50 840 ms).

**The only quality gap large enough to survive scrutiny is paid vs the free local
model, and even it barely clears 0** (+0.048, CI reaches down to +0.004). Read the
width and the size, not the sign: differences inside the paid pack are roughly one
grading notch on one or two items out of twenty - exactly the false precision the
statistical-tie framing exists to catch.

**p95 latency exposes a tail the mean flattens - and this run, the tail is the local
model alone.** llama3.2's p50 is 3.2 s but its p95 is **26 s** (the test machine, not
a hosted service). The paid models all stay under ~7.3 s at p95, gpt and haiku tightest
(~3.4 s). Note the finding-vs-sample caution: the *previous* roll had a large p95 tail
on deepseek too; this roll deepseek is tame (p95 6.0 s). So "a specific paid model has
a latency tail" did NOT survive the re-roll and is not claimed - what survives is the
general point that **p95, not the mean, is what a user feels, and it must be reported**;
which model happens to spike is a per-run sample.

**Cost per passed answer barely differs from cost per query - because this suite does
not separate the models on pass rate.** Most cells pass, so "value-adjusted" cost
collapses back to raw cost. That is itself the finding: the easy suite is too easy to
stress the quality axis, which is why more *reps* of it would spend without buying a
finding. The move that buys one is a harder suite or a stronger judge (see
`adversarial-suite-analysis.md` and `judge-ablation-analysis.md`) - not more of the
same. Cost-per-pass is the right metric to carry to a suite where models actually fail,
where a cheap wrong answer stops looking cheap.
