# Adversarial suite - the harder suite separated the models, but only a reliable judge could see it

The easy golden set could not tell the models apart (all pass, a narrow score band the
bootstrap called a tie). The disciplined reply was "harder items, not more items": a
10-item adversarial suite with knowable answers but real traps (bat-and-ball, a
letter-count, a common-misconception fact, a riddle, modular date arithmetic), re-swept
through all five models, 2 reps. A suite swap is itself a comparison. Spend this run:
$0.067 for the sweep, $0.026 to re-judge all 200 answers with the paid judge (below).

> **Finding vs sample.** Non-deterministic: a re-roll moves the numbers and swaps which
> items disagree. The FINDING that survives is the direction: on the hard suite the free
> local model makes mistakes the frontier models do not, that gap is real, and it is
> **visible only through a reliable judge** - the cheap local judge rates the weak model
> right alongside the pack and hides it. The exact scores and the specific disagreeing
> items are one run's sample (2026-08-17). One old claim did NOT survive and is dropped,
> not renumbered - see the note at the end.

## First result: under the cheap judge, the harder suite ALSO looked like a tie

Under the free local llama judge - the one the sweep uses - the adversarial suite made
the models look about as alike as the easy one:

| model | adversarial score (free judge) | pass rate |
|---|---|---|
| claude-haiku-4-5-20251001 | 0.970 | 20/20 |
| gpt-5.6-luna | 0.970 | 20/20 |
| llama3.2 (local) | **0.950** | **20/20** |
| deepseek-v4-pro | 0.945 | 20/20 |
| claude-sonnet-5 | 0.937 | 20/20 |

Taken at face value this says "even hard items can't separate these models, and the
free local model (0.950, a perfect 20/20 pass) is right in the mix." That reading is
wrong, and the raw trace is what exposes it: the cheap judge waved through llama's wrong
trap answers.

## Reading the raw answers: the cheap judge passed the weak model's wrong trap answers

The scores hid what the answers show. Every case verbatim from the captures (all
`llama3.2`, all cheap-judge PASS / paid-judge FAIL):

- **"strawberry" letter count** (`adv_count_001`): llama answered *"twice"* (it is 3).
  Cheap judge **0.75, PASS** - its own reasoning admits *"fails to account for a third
  occurrence"* - paid judge 0.50, FAIL, correctness 0/10.
- **the Sargasso Sea** (`adv_factual_002`, both reps): llama answered *"the Mediterranean
  Sea"* (flatly wrong). Cheap judge **0.85, PASS** (correctness 8/10 for a wrong sea);
  paid judge 0.40, FAIL, correctness 0/10, *"gives the wrong sea."*

Partial-correctness plus full-relevance clears the cheap judge's 0.7 bar even when the
factual answer is wrong. The models did diverge on the traps; the cheap judge smeared it
away by scoring a wrong answer as mostly right.

## Quantifying it: re-judge the hard suite with the paid judge

Re-grading the same adversarial answers with the paid gpt judge (answers not re-called)
turns the anecdote into a count. On the adversarial suite, **all 3 judge disagreements
are the cheap judge passing a wrong `llama3.2` answer** the paid judge fails (the
strawberry miscount and the Sargasso error, twice). None run the other way.

The aggregate effect is the finding: under the reliable paid judge, `llama3.2`'s
adversarial score drops from **0.950 to 0.903** and its pass rate from **20/20 to
17/20 (85%)**, while the paid pack sits at **~1.000 / 100%**. The quality gap between
the free local model and the paid pack was real on hard items - the cheap judge was
hiding it behind a 0.950 and a perfect pass rate.

## What the adversarial suite says (at the level that survives)

**The suite swap worked - it surfaced a real gap the easy suite could not.** On hard
items the free local model makes mistakes the frontier models do not (a miscount, a
confidently wrong sea), and that is the "which model for which use case" signal the
benchmark exists to find. It was invisible on the easy suite because nothing there was
hard enough to be wrong.

**But the gap is only visible through a reliable judge - which sharpens the cheap-judge
question.** The cheap judge scored llama 0.950 / 20-of-20 on the adversarial suite, tied
with the pack; the reliable judge pulled it down to 0.903 / 17-of-20. The production
reading: spend on the strong judge when the suite is hard enough to matter, because that
is exactly when the cheap judge silently mis-ranks the weak model as good enough.

---

**One claim did NOT survive the re-roll (finding-vs-sample, kept honest).** A previous
roll reported "judge disagreements *double* on the hard suite (2 easy -> 4 adversarial)."
This roll the split is the reverse - 5 easy, 3 adversarial - so that specific claim was a
sample artifact and is dropped, not restated with new numbers. What survives is the
directional part that held both rolls: on the adversarial suite the cheap judge's
disagreements are one-directional lenience on the weak model, and they move its aggregate
score and pass rate down once a reliable judge is used. See `judge-ablation-analysis.md`
for the full both-suite disagreement breakdown (8 of 200 this run).
