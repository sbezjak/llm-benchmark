# Judge-choice ablation - is a cheap judge good enough to grade?

The sweep graded every answer with the FREE local llama3.2 judge - a small model
grading frontier ones, and the same weights as one of the answering models. That
makes its quality ranking suspect. So we re-graded the **same 200 cached answers**
(100 easy + 100 adversarial) with a paid frontier judge (gpt-5.6-luna) and asked
whether the ranking survives the judge swap. The answering models were never
re-called - the judge reads the answer text already on disk - so the whole ablation
cost **$0.02641** (200 short judge calls; receipt: `rejudge done: 200 verdicts, new
judge spend $0.02641` in `evidence/raw-logs/rejudge-gpt-full.log`). Verdicts live in
`cache/judged/`, never overwriting the free-judge arm. Regenerate with the tested
`rejudge_all` runner over the cache.

This is the project thesis turned on the eval itself: P5 asks "cheap model or
expensive one?"; the ablation asks the same of the *grader*.

> **Finding vs sample.** This is non-deterministic - a re-roll moves every number in
> the table and swaps which items disagree. The FINDING that survives re-rolls is the
> *direction*: swapping to the reliable judge pushes the paid pack toward a 1.000
> ceiling and pushes the weak local model DOWN, and the cheap judge's per-item errors
> concentrate as false PASSES on the weak model's wrong answers. The exact scores,
> the "8 of 200," and the specific quoted items are **one run's sample (2026-08-17)**
> - illustration, not the claim. (Vindication of exactly this discipline: last roll's
> leaderboard put gpt-5.6-luna #1 on quality; this roll it is #3. The only claim that
> held is the interval-level one - "the paid pack is a tie, don't rank it." The rank
> was noise; the tie was the truth.)

## The two judges, side by side (all 200 answers, mean score)

| answer model | free judge (llama3.2) | paid judge (gpt-5.6-luna) | delta | pass free | pass paid |
|---|---|---|---|---|---|
| claude-haiku-4-5-20251001 | 0.969 | 1.000 | +0.031 | 40/40 | 40/40 |
| gpt-5.6-luna | 0.966 | 1.000 | +0.034 | 39/40 | 40/40 |
| deepseek-v4-pro | 0.957 | 1.000 | +0.043 | 40/40 | 40/40 |
| claude-sonnet-5 | 0.938 | 0.997 | +0.060 | 39/40 | 40/40 |
| llama3.2 (local) | 0.936 | 0.919 | **-0.018** | 39/40 | **35/40** |

The swap does one clean thing: **every paid model rises toward a 1.000 ceiling, and
`llama3.2` is the only model that goes DOWN** (-0.018). Under the cheap judge the
local model looks a whisker off the pack (0.936 vs sonnet's 0.938); under the paid
judge the gap opens and its pass rate drops from 39/40 to **35/40** while every paid
model holds at 40/40. The cheap judge compresses the gap away; the reliable judge
separates the weak model out. That IS the capstone finding, seen at the grader.

## What the ablation says

**For the aggregate leaderboard call, the cheap judge is roughly good enough - but it
systematically flatters the weak model.** Under either grader the top-line story is
"the paid pack ties, the local model is the laggard." A benchmark reporting only the
ranking reaches the same conclusion with the cheap judge. That is the sense in which
the cheap judge is "good enough" - and it is a real sense. But the cheap judge puts
`llama3.2` a hair below a genuine frontier model, and the paid judge shows that gap
is real and wider.

**8 of 200 individual pass/fail calls flip between judges, and they do NOT cancel -
6 of the 8 are on the weak model's answers, 5 of those the cheap judge PASSED and the
paid judge FAILED.** The cheap judge is not noisy in both directions; it is
*specifically too soft on `llama3.2`'s wrong answers*:

- `adv_factual_002` (both reps) - "only sea with no land coastline?" (Sargasso). Llama
  answered **"the Mediterranean Sea"** - flat wrong. The **cheap judge scored it 0.85
  (correctness 8/10!)**, "correctly identified a sea without a direct land coastline
  but incorrectly stated its name"; the paid judge scored 0.40 (correctness 0/10),
  "gives the wrong sea." The cheap judge gave 8/10 correctness to a wrong answer.
- `adv_count_001` - "how many r's in strawberry?" (3). Llama answered **"twice."** The
  **cheap judge passed it at 0.75** ("partially correct... fails to account for a
  third"); the paid judge failed it at 0.50 (correctness 0/10), "incorrectly counts
  the three occurrences as two." A wrong count, waved through.
- `procedural_001` - "pytest command for slow OR network?" (`-m 'slow or network'`).
  Llama emitted `pytest -k "slow" --network` (invalid; `--network` is not a pytest
  flag). The **cheap judge passed it at 0.85**; the paid judge failed it at 0.60,
  "does not express an OR condition."

The other 3 disagreements run the other way (cheap judge too harsh on a correct
answer: `procedural_001`/sonnet, `reasoning_001`/llama, `reasoning_003`/gpt, all
free-fail / paid-pass) - but the weight is on the soft side, on the weak model. The
lesson is methodological: **aggregate agreement between judges does not imply
per-item agreement, and here the per-item disagreement is not random noise but a
consistent lenience toward the model you most need the judge to catch.** You have to
count the individual flips and look at whose answers they land on.

**gpt-as-judge did not uniquely favor its own answers** (it scored haiku, deepseek,
and sonnet at 1.000 too, and llama's `reasoning_003` up to 0.95) - so the pattern is
generosity toward genuinely strong answers plus rigor on wrong ones, not an own-model
preference. That rigor on wrong answers is exactly what the cheap judge lacks.

## The honest reading

A cheap judge is good enough for the *aggregate* "which model" call on this suite,
is never fully reliable per-item, and its errors are not symmetric: they concentrate
as false PASSES on the weakest model's factual and syntactic mistakes - the errors a
benchmark most needs its grader to catch. On the adversarial half, where answers can
actually be wrong, that lenience is what drives llama's pass rate down under the paid
judge (39/40 -> 35/40). Buy the reliable judge precisely because it catches what the
cheap one rubber-stamps. Every number here recomputes from `cache/` (free arm) and
`cache/judged/` (paid arm) at threshold 0.7; the two-judge means and pass rates are
also in `reports/findings.json` (`views.*.free-local` vs `views.*.paid-gpt`).
