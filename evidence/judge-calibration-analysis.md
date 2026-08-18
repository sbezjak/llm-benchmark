# Judge calibration - human vs judge agreement

**What this is.** The judge (gpt-5.6-luna) is a model standing in for a human grader.
This validates that proxy: a human (Sara) graded a 24-answer sample **blind** - without
seeing the judge's scores - into Good / Weak / Wrong. We then compared. This is the one
production-grade move in P5 that costs $0, only reading time.

**Sample.** 24 answers from the 2026-08-17 re-roll, stratified: every one of the 14
non-perfect paid-judge verdicts (5 fails, 9 borderline) + 2 perfect controls per model.
Blind sheet: `docs/judge-calibration-grading-sheet.md`; sealed key:
`evidence/judge-calibration-key.json`. Regenerate the sheet/key with
`python -m llm_benchmark.calibration` (pure, $0; it refuses to overwrite a graded sheet).

## Headline

- **Blind 3-band agreement: 20/24 = 83%.** This is the reportable number - the
  independent one.
- **Pass/fail agreement: 24/24 = 100%** (Good+Weak = pass, Wrong = fail). Human and
  judge never once disagreed on whether an answer passes.
- **On review, 3-band rises to 24/24 = 100%** - but this is *post-hoc*, after the human
  saw the judge's verdicts and we discussed them, so it is NOT independent and is not the
  calibration figure. It is recorded only to show the disagreements were all soft calls
  the human herself resolved toward the judge, not hard conflicts. **Report the blind 83%.**

## The four disagreements were all the same thing: a low-confidence "Weak" on a correct answer

Every one of the 4 blind disagreements was **Human=Weak, Judge=Good (score 1.0)**, and
every one was the **bat-and-ball** item (`adv_arith_001`): G01 (llama3.2), G04
(gpt-5.6-luna), G13 (claude-haiku), G20 (claude-sonnet). All four answers reach the
correct $0.05 with valid algebra - so on correctness they are right, which is why the
judge scored them 1.0.

Crucially, this was **not** a blanket "this item is weak" rule: the human marked **6
other** bat-and-ball answers **Good** (G05, G11, G15, G18, G21, G24). So these four were
fine-grained, genuinely borderline presentation calls - correct answers she was unsure
whether to dock for how they were written (one, G01, buries the answer at the very end
after all the steps). On review she moved all four to Good. The human was uncertain
exactly where the judge was confident, and on reflection agreed with it.

## What this says (at the level that survives)

**The judge did not miss anything the human was sure about.** On every confident call -
all 5 Wrong (the judge failed the same 5, the weak model's real errors: strawberry
miscount, Sargasso->"Mediterranean", the invalid pytest command) and the 15 clear Goods
- human and judge agreed. The entire gap was the human's *uncertain* middle: "correct,
but is it a good answer?" - a severity the judge structurally cannot express, because it
scores correctness+relevance and has no "Weak" band (its band is Good if it passes 0.7,
Wrong if not). So the 3-band / pass-fail split (83% vs 100%) is exactly that missing
middle, and here the middle was four low-confidence calls the human revised away.

This is a **weaker, more honest** result than the previous roll's writeup (which had the
judge forgiving a real science error). This roll, on this sample, the judge and a human
essentially agree - the only friction was the human's own uncertainty on borderline
presentation, not the judge waving through anything wrong. That is a finding-vs-sample
point in itself: the *specific* disagreement pattern changed completely between rolls
(science-error lenience last time, borderline-presentation uncertainty this time), so
neither is a stable property of the judge - what is stable across both is that the two
never disagree on pass/fail.

## Full comparison

| ID  | item            | answer model              | Human (blind) | Human (review) | Judge | score |
|-----|-----------------|---------------------------|---------------|----------------|-------|-------|
| G01 | adv_arith_001   | llama3.2                  | **Weak**      | Good           | Good  | 1.00  |
| G02 | reasoning_003   | llama3.2                  | Good          | Good           | Good  | 0.95  |
| G03 | adv_logic_001   | llama3.2                  | Good          | Good           | Good  | 0.95  |
| G04 | adv_arith_001   | gpt-5.6-luna              | **Weak**      | Good           | Good  | 1.00  |
| G05 | adv_arith_001   | deepseek-v4-pro           | Good          | Good           | Good  | 1.00  |
| G06 | adv_riddle_001  | llama3.2                  | Good          | Good           | Good  | 0.95  |
| G07 | procedural_001  | llama3.2                  | Wrong         | Wrong          | Wrong | 0.60  |
| G08 | reasoning_003   | llama3.2                  | Good          | Good           | Good  | 0.95  |
| G09 | reasoning_001   | llama3.2                  | Good          | Good           | Good  | 0.85  |
| G10 | adv_count_001   | llama3.2                  | Wrong         | Wrong          | Wrong | 0.50  |
| G11 | adv_arith_001   | llama3.2                  | Good          | Good           | Good  | 1.00  |
| G12 | adv_logic_001   | llama3.2                  | Good          | Good           | Good  | 0.85  |
| G13 | adv_arith_001   | claude-haiku-4-5-20251001 | **Weak**      | Good           | Good  | 1.00  |
| G14 | procedural_001  | llama3.2                  | Wrong         | Wrong          | Wrong | 0.40  |
| G15 | adv_arith_001   | claude-sonnet-5           | Good          | Good           | Good  | 1.00  |
| G16 | adv_factual_002 | llama3.2                  | Wrong         | Wrong          | Wrong | 0.40  |
| G17 | adv_riddle_001  | claude-sonnet-5           | Good          | Good           | Good  | 0.95  |
| G18 | adv_arith_001   | claude-haiku-4-5-20251001 | Good          | Good           | Good  | 1.00  |
| G19 | adv_factual_002 | llama3.2                  | Wrong         | Wrong          | Wrong | 0.40  |
| G20 | adv_arith_001   | claude-sonnet-5           | **Weak**      | Good           | Good  | 1.00  |
| G21 | adv_arith_001   | gpt-5.6-luna              | Good          | Good           | Good  | 1.00  |
| G22 | reasoning_001   | llama3.2                  | Good          | Good           | Good  | 0.95  |
| G23 | procedural_001  | claude-sonnet-5           | Good          | Good           | Good  | 0.95  |
| G24 | adv_arith_001   | deepseek-v4-pro           | Good          | Good           | Good  | 1.00  |

Human blind band totals: Good 15, Weak 4, Wrong 5. Judge: Good 19, Wrong 5, Weak 0.

## Limits (do not overclaim)

- **The 83% is the number; the 100%-on-review is not independent.** The revision happened
  after the human saw the judge and we discussed it, so it cannot be used as a blind
  agreement rate. It only tells us the disagreements were soft.
- **n=24, and all 4 disagreements are one item.** This is one borderline pattern (correct
  answer, uncertain presentation) observed four times, not four independent signals.
- **The disagreement pattern is a per-run sample.** It was science-error lenience last
  roll, borderline-presentation uncertainty this roll. Do not read either as a stable
  judge property; the stable part is the 100% pass/fail agreement.
- The judge has **no Weak band** - it is pass/fail around 0.7. So 3-band agreement will
  always understate alignment and pass/fail overstate it; report both.

## For the scoreboard

Reportable line: **"Graded blind against a human, the judge agreed on 20 of 24 answers
(83%) by 3-band and 24 of 24 (100%) on pass/fail; all four disagreements were the human
being unsure whether to dock a *correct* answer for its presentation - cases she revised
toward the judge on review. The judge didn't pass anything the human judged wrong."**
Pair the 83% with the 100% so the reader sees both the strict and lenient reading.
