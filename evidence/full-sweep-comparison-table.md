# Full-sweep comparison table - five models, one fixed suite

The benchmark ran the same 10-item golden set through five models, twice each
(2 reps per model+item), and measured every call on three axes: cost per query,
latency, and quality. Quality is the LLM-judge score (a second model grades each
answer on correctness and relevance, combined into [0, 1]; pass threshold 0.7).

Every number below is computed from the raw per-call captures under `cache/`,
one JSON record per (model, item, rep), holding the prompt, the response, the
token counts reported by the provider itself, latency, time-to-first-token, the
dollar cost, the provider `request_id`, and the judge's verdict.

> **Finding vs sample.** These are a non-deterministic system; a re-roll moves every
> number and can reorder the leaderboard. The FINDING that survives is the *shape*:
> the paid models are a quality tie (their CIs overlap - see
> `benchmark-validity-analysis.md`), so quality does not pick among them; cost and
> latency do, and there gpt-5.6-luna is the cheapest and fastest. The exact scores and
> ranks in the table are **one run's sample (2026-08-17)** - do not read the ordering
> as "which model is best." This roll makes the point on its own: gpt-5.6-luna is the
> *third*-highest quality score here, and the previous roll it was the highest. The
> rank is noise; the tie is the finding.

100 captures = 10 items x 5 models x 2 reps. Total spend this run: $0.06365.

| model | mean $/query | total $ | mean latency | mean TTFT | mean out-tokens | quality | pass rate |
|---|---|---|---|---|---|---|---|
| deepseek-v4-pro | $0.000138 | $0.00277 | 2713 ms | 1869 ms | 112.7 | 0.970 | 100% |
| claude-haiku-4-5-20251001 | $0.000537 | $0.01073 | 1597 ms | 822 ms | 103.9 | 0.967 | 100% |
| gpt-5.6-luna | $0.000082 | $0.00163 | 1323 ms | 916 ms | 65.3 | 0.962 | 95% |
| claude-sonnet-5 | $0.002426 | $0.04852 | 4093 ms | 1552 ms | 238.6 | 0.937 | 95% |
| llama3.2 (local) | $0.000000 | $0.00000 | 7859 ms | 507 ms | 129.7 | 0.922 | 95% |

Rows ordered by this run's quality score. Latency and TTFT are wall-clock; the local
model's high latency reflects the test machine, not a hosted service. The quality
column spans 0.922-0.970 - a 0.048 band the bootstrap treats as a tie among the four
paid models (that analysis is `benchmark-validity-analysis.md`).

## What the table says (at the level that survives)

**On this QA workload, quality does not separate the paid models - so pick on cost
and speed, and there gpt-5.6-luna wins.** The four paid models land inside one narrow,
overlapping quality band; the ordering inside it flips between runs and is not signal.
What is stable is that gpt-5.6-luna is the cheapest per query and the fastest by mean
latency, so for a short factual / definition / reasoning QA workload the pick is the
small, cheap, fast model - not because it scores highest (it does not, reliably) but
because the ones that cost more do not reliably score higher. You do not need the
expensive one.

**The most expensive model is the majority of the spend for no quality edge.** Sonnet
is $0.0485 of the $0.0637 total (76% this run) and its quality sits inside the same
tie as models a fraction of the price. Its longer answers (239 output tokens average,
versus gpt's 65) buy latency and cost, not a separable score, on questions this suite
asks. A larger reasoning model earns its price on tasks where the extra reasoning
changes the answer; none of these ten did.

**Two reps caught a quality wobble a single run would have faked.** Three of the five
models sit at 19/20 passes rather than 20/20, and reading the individual runs shows
the misses are not stable - a model passes an item on one rep and fails it on the
other, right at the 0.7 threshold. A single-run sweep would have labelled quality by
chance. The honest reading is "these answers sit on the judge's pass threshold and
cross back and forth," not "this model reliably fails this item" - which is itself a
sample-vs-finding point: the *which* items wobble changes per roll; *that* near-
threshold answers wobble is the stable part, and it is visible only because the suite
was run more than once.
