# Judge position bias - the cheap judge grades on order, not quality

The pairwise lens shows each judge two answers to the same question and asks which
is better, then shows it the **same two swapped**. A preference only counts if it
survives the swap; a verdict that changes with the order was a vote on **position**,
not quality. Run over the free local judge (`llama3.2`), the swap catches it in the
act - and each verdict stores the judge's own reasoning, so you can read it
contradict itself in the cache file.

This is the project thesis turned on the evaluator: the benchmark asks "cheap model
or expensive one?"; this asks the same of the **grader**, and the answer is that a
cheap grader cannot be trusted on its own.

> **What is the finding vs what is a sample.** These are a non-deterministic
> system, so a re-roll moves every point number and swaps the specific examples.
> The FINDING is the *direction and the interval*, which survive re-rolls: the cheap
> judge flips on order 2-3x more than the paid judge, most of its flips are pure
> slot-A bias, and it inflates its own model. The exact percentages and the quoted
> pairs below are **one run's sample (the 2026-08-17 re-roll)** - they illustrate the
> finding, they are not the finding, and they will differ next roll. So read them as
> "this is what it looked like on this run," not "this is the number."
>
> All figures are regenerated straight off the cache by
> `python -m llm_benchmark.freeze_evidence` (pure, $0) into
> `evidence/judge-position-bias-stats.json`, which cross-checks against
> `reports/findings.json` (same cache, independent code path - they agree exactly).
> That is what a frozen file is FOR: an auditable receipt of one run, not a claim
> that the run's numbers are stable.

## The numbers (off `evidence/pairwise-verdicts/`, 200 both-orders verdicts / arm)

*The stable claim is the direction; the figures in bold are this run's sample.*

- **The cheap judge changes its winner on order for a large fraction of pairs** -
  this run, 78 / 200 (39%): 45 / 100 on the easy suite (45%), 33 / 100 on
  adversarial (33%). Easy is worse because every answer there is correct, so there
  is nothing but style and order left to grab.
- **Of the 78 flips, 70 (90%) picked whichever answer was shown FIRST, both times.**
  The judge is not disagreeing with itself about quality - it is reflexively
  rewarding slot A.
- **90% of the flip reasonings (140 / 156) literally begin with "Answer A."** The
  model decides on position, then writes a justification to fit.
- Contrast, same grid, the paid judge (`gpt-5.6-luna`): **42 / 200 flips (21%)**, and
  of those only **6 (14%)** were pure first-answer bias, with just **5% of its flip
  reasonings (4 / 84) opening "Answer A."** So the paid judge's flips are mostly
  *genuine disagreements*; the cheap judge's are mostly *position*. On the easy suite
  head to head the cheap judge flips 45% vs the paid judge's 19% - **2.4x more** on
  presentation order. Position bias is a property of the instrument, and a better
  instrument has far less of it - but "grade in both orders" is the control either way.
- Self-preference: in the arm it judges, `llama3.2` ranks **its own answers #3 of 5**
  (win rate 43% easy, 51% adversarial) while scoring answers on their own puts it
  last. An in-pool judge inflates itself - a second reason not to let a contestant
  grade.

## The judge indicting itself (its own reasoning, from the verdict files)

The reasoning is stored in each verdict, so these quotes are the judge's own replies
for the exact flip named - no reconstruction. Full set (all 78 flips with both-order
reasoning): `evidence/judge-position-bias-receipts.json`, built straight from the cache.

**`factual_004` - "How many planets are in our solar system?"** (both answers say 8,
list them, note Pluto - equivalent). `claude-sonnet-5` vs `llama3.2`,
`evidence/pairwise-verdicts/llama3.2/easy__factual_004__claude-sonnet-5__llama3.2.json`:
- shown `claude-sonnet-5` first -> picks sonnet: *"Answer A is more correct and
  relevant because it provides additional context about Pluto's reclassification as
  a dwarf planet."*
- shown `llama3.2` first -> picks llama: *"Answer A is more correct and relevant
  because it provides the most accurate number of planets in our solar system
  without referencing Pluto as a planet."*
- Pluto context wins one way, *not* mentioning Pluto wins the other - each reason
  praises whatever landed in slot A, and both open "Answer A."

**`procedural_001`**, `deepseek-v4-pro` vs `gpt-5.6-luna` - the purest tell,
**near-identical wording, opposite winner:**
- shown deepseek first -> picks deepseek: *"Answer A is better because it uses double
  quotes around the command to ensure correct shell syntax and prevent word splitting."*
- shown gpt first -> picks gpt: *"Answer A is better because it uses double quotes
  around the command to ensure correct syntax and prevent shell word splitting."*
- The same sentence, reordered, names a different winner - the reasoning is a fixed
  template with slot A dropped in.

**`adv_riddle_001`**, `claude-sonnet-5` vs `llama3.2` - contradictory criteria on the
hard suite:
- shown sonnet first -> picks sonnet: *"Answer A is more correct and relevant because
  it provides accurate information about the materials used to build a greenhouse and
  explains the origin of its name in a clear and concise manner."*
- shown llama first -> picks llama: *"Answer A is more correct and relevant because it
  provides a clear explanation of the materials used to build a greenhouse and
  addresses the linguistic trick in the question's name."*

## Where this is proven (all committed - checkable from a clone)

- **Verdicts WITH reasoning (ground truth):** `evidence/pairwise-verdicts/llama3.2/` -
  one JSON per pair, both orders and both reasonings in each file.
  `easy__factual_004__claude-sonnet-5__llama3.2.json` is the headline receipt
  (`order1_winner` != `order2_winner`, with `order1_reasoning` / `order2_reasoning`).
  The paid judge's arm is alongside it.
- **Distilled receipts:** `evidence/judge-position-bias-receipts.json` - all 78
  cheap-judge flips with both-order reasoning, plus
  `evidence/judge-position-bias-stats.json` - every number above, per suite and arm.
  Both are regenerated by `python -m llm_benchmark.freeze_evidence` (the committed
  generator, pure and $0), so the evidence can never silently drift from the cache
  again - the drift that forced this re-freeze.
- **Full run logs (belt and suspenders):** `evidence/raw-logs/pairwise-llama-full.log`
  (+ `-gpt-full.log`) - every prompt and reply verbatim.
- **Aggregate + the report:** flip counts and the computed `first_answer_bias` are in
  `reports/findings.json`; the scoreboard renders the flip chart and the 90% / 5%
  stat. The `factual_004` receipt with its two contradicting reasonings lives in the
  verdict file itself, under `evidence/pairwise-verdicts/llama3.2/`.

## The fix this drove (why reasoning is now in the cache)

The earlier run stored only the verdict, not the reasoning, so the evidence lived in
a `/tmp` log we had to race to recover, and a specific quote couldn't be pinned to
the exact cached call. `runners/pairwise.py` now persists `order1_reasoning` /
`order2_reasoning` in every verdict (see `compare_pair` + `parse_reasoning`, and
`test_verdict_records_the_judges_reasoning_both_orders`), so the raw output IS the
ground truth in the cache. The numbers here are the re-judged run that populated it
(cheap arm $0 local, paid arm $0.057).

## The limitation, named (how a real version fixes it)

`llama3.2` grades here because it is the free local baseline, and it is also one of
the answering models - both a weak grader and an in-pool one. Production would: use a
judge that is not a contestant, or better a small **panel** of judges with majority
vote (removes self-preference and single-judge bias); grade in both orders as
standard and average; and spot-check a sample against human judgement. The cheap
in-pool judge is the deliberate stand-in whose failure is the finding - not a result
to trust, a hazard to demonstrate.
