# Walkthrough

Live HTML reports: [scoreboard](https://sbezjak.github.io/llm-benchmark/reports/report-benchmark-scoreboard-2026-08-18.html) - the aggregate table, start here · [golden sweep](https://sbezjak.github.io/llm-benchmark/reports/report-golden-sweep-2026-08-17.html) and [adversarial sweep](https://sbezjak.github.io/llm-benchmark/reports/report-adversarial-sweep-2026-08-17.html) - the full per-row transcripts (every prompt, answer, and judge verdict)

I'm an automation tester - my normal job is checking an app does the same thing every
time. This is a different kind of testing: I run the same questions through five language
models at once and compare them on the three things a team pays for - cost per query,
speed, and answer quality.

The surprise is that the scoreboard is the least trustworthy thing it produces. The models
land in a tiny quality band, so the ranking is mostly noise - and every quality number
came from a grader that is itself one of the five models graded, a contestant scoring its
own group. So the real work wasn't ranking the models but checking whether I could trust
the number that ranks them.

## What this is

A pytest harness that points Project 1's scorers and LLM-as-judge at five models side
by side: a free local Llama baseline, plus GPT, DeepSeek, and two Claude tiers, all
behind one `Provider.generate` seam. It runs one fixed suite through each, twice, and
records what every call cost, how long it took, and how the judge scored it.

It's also the first project in the series that spends real money - every one before it
ran locally, for free. Here each call has a price, and the whole comparison came to about
21 cents. That's a small price, but it changed how I tested, and not in the way I expected
(see "What the price tag changed").

There's one rule the whole thing runs on: **run each call once, save it, and do all the
math off the saved copy - never a second call.** You never pay twice, and the saved answer
is the ground truth every number has to agree with.

### The two suites

The benchmark ships two hand-written suites:

- **The golden set** (`data/golden_set.yaml`) - 10 easy items from Project 1, one fact
  each (*"capital of Slovenia?"* - Ljubljana). All five models pass near 100%, so the
  scores collapse into a tie and can't separate them.

- **The adversarial set** (`data/adversarial_set.yaml`) - 10 harder items, each with one
  knowable answer but a trap a weak model falls into: the bat-and-ball problem (the ball
  is $0.05, not $0.10), *"how many r's in strawberry?"* (three), the only sea with no
  coastline (the Sargasso).

The main finding lives in the adversarial set, because it's the only place the weak model
is actually wrong - the only place you can catch the judge passing a wrong answer.

### Vocabulary

| In this repo | What it means |
|---|---|
| Sweep | One run of the whole grid: every question, through every model, N times, each call measured and saved. |
| Capture | The saved record of a single call - the prompt, the answer, the provider's own token counts, latency, cost, the request id, and the judge's verdict. One JSON file per (model, item, rep - this project runs 2 reps). |
| Judge (LLM-as-judge) | A second model that reads an answer and scores it on correctness and relevance, combined into a 0-1 number, pass line 0.7. The whole quality axis is the judge's output. |
| Judge swap (rejudge) | Re-grading the same saved answers with a *different* judge, to see how much a score depends on who graded it. No model is called again - the new judge reads answers already on disk. |
| Pairwise / flip | Showing the judge two answers to one question and asking which is better, then showing it the same two *swapped*. If the winner changes, that's a flip - the judge went on order, not quality. |
| Position bias | The judge favouring an answer for where it sits (usually the one shown first) instead of what it says. Flips are how you measure it. |
| Self-preference | A judge scoring its own answers up when it's grading a pool it competes in. |
| Bootstrap range | A confidence range around a score, built by resampling the items. A range that crosses zero on a *difference* means the two models are a tie on this suite. |
| Batch lane | Running the same sweep asynchronously through the provider's batch API - about half price, minutes of turnaround instead of seconds. |
| Semantic cache | A cache that serves a stored answer when a new question *looks* similar. A false hit is when "looks similar" and "same correct answer" come apart. |
| Calibration | A blind human grade of a sample of answers, compared to the judge, to check the judge is a decent stand-in for a person. |
| `mocked` / `live` / `billed` | Three test tiers: `respx`-mocked (fast, no network), live local Ollama (free, slow), and paid provider calls (real spend, off by default). |

### How it works

The harness is built out of seams that don't know about each other, so a failure in one
can't be mistaken for a failure in another:

- **providers** - the only place an HTTP call happens. One `Provider.generate` interface
  hides five backends (Ollama local, OpenAI, DeepSeek on the OpenAI-shaped seam, and two
  Anthropic tiers). Tests mock here, so the default run touches no network and spends
  nothing. Every prompt and reply is logged in full.
- **scorers** - the quality axis, carried over from Project 1. Pure functions except the
  judge, which is itself a model call.
- **runners** - drive the grid and, separately, do the math. `sweep.py` runs and saves
  each call; `findings.py` reads all the saved calls and aggregates them into
  `findings.json`; `stats.py` computes the ranges and paired differences. The math never
  calls a provider.
- **dataset.py** - loads and validates whichever suite you point it at.

On top of capture-once, there's a **compute / render split**: `findings.py` does the
analysis and writes `findings.json`; the report files only draw it. So the analysis gets
a $0 unit test, and any report regenerates from the saved calls for free.

To read the raw calls, open a sweep report (golden or adversarial, linked up top): every
test row carries the full prompt, answer, and judge verdict, untruncated. The scoreboard
is the aggregate view built from those same calls.

## The three axes, and how each is measured

| Axis | The question it asks | How it's measured |
|---|---|---|
| **Cost** | What does one answer cost? | The provider's own token counts times its published price, saved per call. Also reported per *passed* answer, so a cheap wrong answer stops looking cheap. |
| **Latency** | How long does a user wait? | Wall-clock per call, plus time-to-first-token. Reported at the median *and* the 95th percentile, because the mean hides the tail a user actually feels. |
| **Quality** | Is the answer any good? | The LLM-judge's 0-1 score, pass line 0.7 - the number the rest of this project checks, because the judge is a model too. |

### The scoreboard those three axes produce

The 2026-08-17 golden-set run, ordered by quality score - clean and easy to read, which is why I didn't stop here:

```
model                    quality   mean $/query   mean latency   out-tokens
deepseek-v4-pro          0.970     $0.000138      2713 ms        113
claude-haiku-4-5         0.967     $0.000537      1597 ms        104
gpt-5.6-luna             0.962     $0.000082      1323 ms         65
claude-sonnet-5          0.937     $0.002426      4093 ms        239
llama3.2 (local)         0.922     $0.000000      7859 ms        130
```

Read straight, it looks done: the whole quality column sits in a 0.92-0.97 band. The cheapest, fastest model (GPT) scores as high as the rest, and the priciest (Sonnet, ~30x the cost per query) scores no higher - its answers are just longer (239 tokens to GPT's 65). So the easy takeaway is: use the small cheap model, skip the expensive one. Two findings below say don't stop there (F1, F2).

## Findings

The seven findings, each with the easy read you'd take at first and the one that holds
up when you run it again. "Where" points at the committed evidence - a written analysis
(`evidence/*.md`) backed by a machine-checkable receipt (`*.json`), or a test.

| # | Finding | The easy read | What holds up on a re-run | Where |
|---|---|---|---|---|
| F1 | The paid models tie on quality | "Model X is best, it tops the board" | Every paid-vs-paid range crosses zero; the ranking is noise, so pick on cost and speed | `benchmark-validity-analysis.md`, `full-sweep-comparison-table.md` |
| F2 | The cheap judge hides the gap (main) | "The local model scores with the rest, 0.950 / 20-of-20" | A reliable judge pulls it to 0.903 / 17-of-20 while the paid models hold - the cheap judge was passing wrong answers | `judge-ablation-analysis.md`, `adversarial-suite-analysis.md` |
| F3 | Position bias | "The judge has a view on which answer is better" | The cheap judge flips on order 2-3x more than the paid one, and most flips just keep the first-shown answer | `judge-position-bias.md` (+ `-receipts.json`, `-stats.json`) |
| F4 | Self-preference | "The local model ranks a respectable 3rd head-to-head" | It's grading its own pool and marking itself up - it scores last when graded on its answers alone | `judge-position-bias.md`, `freeze_evidence.py` |
| F5 | Human calibration | "You can't check a model judge without more models" | A blind human grade of 24 answers agreed with the paid judge on pass/fail every time | `judge-calibration-analysis.md`, `evidence/judge-calibration-grading-sheet.md` |
| F6 | Cost levers, measured | "Batching and caching obviously save money" | The batch lane does (~half price, tied quality); a semantic cache serves wrong answers at every threshold; prefix caching doesn't apply here at all | `batch-lane-captures.json`, `semantic-cache-false-hits.md`, `findings.json` |
| F7 | Two passes caught an unstable score | "This model fails that item, it scored 19/20" | The miss isn't stable - the answer sits on the 0.7 line and crosses it between passes; one pass would have recorded a coin flip as a fact | `full-sweep-comparison-table.md` |

## The findings, up close

**F1 - the tie under the leaderboard.** I put a range around each score, and around the
*difference* between each pair of models. Every paid-vs-paid range crossed zero - a tie,
not a close finish. The proof: GPT came third this run and first on the previous one, with
nothing changed but the dice.

**F2 - the cheap judge hides the gap (the main finding).** The whole sweep was graded by
the free local Llama - a weak model, and the same weights as one of the answers it grades.
So I re-graded the same 200 saved answers with a paid judge instead - no new calls, just
re-reading text on disk, about three cents. The swap did one thing, over and over:

```
answer model       cheap judge   paid judge   pass rate (cheap -> paid)
claude-haiku-4-5      0.969         1.000        40/40 -> 40/40
gpt-5.6-luna          0.966         1.000        39/40 -> 40/40
deepseek-v4-pro       0.957         1.000        40/40 -> 40/40
claude-sonnet-5       0.938         0.997        39/40 -> 40/40
llama3.2 (local)      0.936         0.919        39/40 -> 35/40   <- the only one that drops
```

Every paid model rose to the top; Llama was the only one that dropped. The gap is on the
hard suite - the only place the weak model is actually wrong: there it fell from a perfect
20/20 under the cheap judge to 17/20 under the paid one, while the paid models held near
100%. (The full-suite numbers above are softer because the ten easy items everyone gets
right dilute it.) Reading where the two judges disagreed shows the drop is real, not noise:

- *"Which is the only sea with no land coastline?"* (Sargasso). Llama answered **"the
  Mediterranean Sea"** - wrong. The cheap judge gave it 8 out of 10 for correctness and
  passed it; the paid judge gave it 0.
- *"How many r's in strawberry?"* (three). Llama answered **"twice."** The cheap judge
  passed it as partly right; the paid judge failed it.

The cheap judge is soft in the one place a benchmark most needs it strict: catching the
weak model's wrong answers. Trust its totals and you ship a scoreboard that scores real
mistakes as passes.

**F3 and F4 - the judge graded on order, and it liked itself.** Show it two answers, then
the same two swapped, and count a win only if it holds. The cheap judge changed its winner
on 45% of the easy pairs - and 42 of those 45 flips just kept whichever answer was shown
first. Its own reasoning gives it away: on the planets question it praised one answer for
*mentioning* Pluto when that answer was on top, then praised the other for *not* mentioning
Pluto when *it* was on top. The paid judge flipped far less. And grading a pool it competes
in, the cheap judge ranked its own answers third of five, though they score last on their own.

**F5 - checking the judge against a human.** The judge stands in for a human grader, so the
last step is to check the stand-in. I graded a 24-answer sample **blind**, then compared:
on pass/fail the human and judge agreed on all 24 - the judge passed nothing I marked
wrong. The only gaps were four *correct* answers I was unsure whether to dock for how they
were written - my uncertainty, not the judge letting something wrong through. Small sample,
one run, but it's the $0 move any team can make: read a sample yourself and see where the
grader and a person come apart.

**F6 - cost levers, measured not assumed.** Three ways to spend less per answer, each
checked rather than assumed:

- **The batch lane works** - the sweep through the provider's batch API cost about half
  price for minutes of turnaround, at quality that ties the live lane.
- **The semantic cache is a trap at every setting** - serving a stored answer when a new
  question *looks* similar has false hits at every threshold I tried; at one it served
  "Ljubljana" to "what is the capital of Slovakia?" (Bratislava) at similarity 1.000.
- **Prefix caching doesn't apply here** - the items share no common preamble, so there's
  nothing to cache. Reporting that null is the point.

**F7 - the pass that wasn't stable.** Three of five models scored 19/20 on the easy suite
instead of a clean sweep. The easy read: "this model fails that item." But those answers
sit right on the 0.7 line and cross it between the two passes - pass once, fail the next,
same question. One pass would have stamped a coin flip as a fact; only the row-by-row trace
shows it. That the "wasteful" second pass is the one telling the truth is the story behind
"What the price tag changed" below.

## The judge is the whole project

Pull the findings together and most are the same problem one level down. The benchmark
grades answers, but the grader is a model too, with the same faults: it passes wrong answers (F2),
goes on order not content (F3), and likes its own work (F4) - while the scoreboard reads
clean the whole time. The only way to see it is to grade the same answers a second way and
*read where the two graders disagree* - the "read the raw output, don't trust the summary"
rule from the earlier projects, pointed at the grader this time.

## What the price tag changed

When every call has a price, there's a constant pull to test less - one pass instead of
two, the free judge instead of a paid one - and the instinct is to push the bill toward
zero. That instinct is backwards, and F7 is the proof: the second pass, the first thing
save-money logic cuts, is exactly the pass that caught answers crossing the 0.7 line
between runs. The few cents you save can buy you the wrong model, or a judge you can't
trust - multiplied by every request you serve.

So the money didn't make me test less, it made me test on purpose - every call had to earn
its place. Normal automation hides the cost, so nobody asks whether a test is worth
running; a price per call puts that in front of you. Same lesson as the earlier projects
("ten sharp items beat fifty easy ones"), except here it shows up on an invoice.

## Known limitations

- **No single best model.** The paid models are too close to rank on this suite, and the
  default judge is biased, so the honest output is "tie, choose on cost and speed," not a
  winner.
- **Small n.** Two passes of a 10-item suite can't resolve a fraction-of-a-point gap. The
  tie is a decision made from that, and stated as one.
- **The judge is one of the tested models.** Llama grades because it's the free local
  baseline - a weak, in-pool grader. That failure *is* the main finding, not a result to
  trust.
- **The numbers are one run.** Every score here is the 2026-08-17 roll. Re-run and they
  move, sometimes enough to reorder the board. The committed `evidence/*.json` files are
  receipts of that run, not claims that the numbers are stable.

**What production would add:** a judge from outside the pool, or a small panel voting;
larger, standard, public suites; extra passes only where scores are close; and human
spot-checks with drift monitoring over time. This harness is built to show the hazards a
real setup is designed to remove.

## What I'd reuse

- **Put a range on a number before you rank on it.** A list of bare averages invites you
  to read noise as signal. A range around the *difference* between two models turns "X
  wins" into the truer "these are a tie, decide on cost." A few lines of resampling, and
  it changed the headline of this whole project.
- **Don't let the thing you're testing grade the test.** The entire quality axis ran
  through a grader that was one of the contestants. The judge swap - grade the same saved
  answers a second way and read the disagreements - is the cheap check that caught it, and
  it costs a few cents because nothing is regenerated.
- **Capture once, math off the saved copy.** One rule that gives you spend control and a
  checkable ground truth at once - every number traces back to a saved answer, and
  re-running the analysis is free.
- **Give every committed proof a generator; don't freeze it by hand.** I learned this the
  hard way: re-calling the models to chase real receipts re-rolls every answer, so my
  hand-built evidence files kept quoting answers that no longer existed. Now every proof
  recomputes off the cache for free - if a claim can't be recomputed, it can't be checked.
- **Re-run non-deterministic tests; don't test them once.** F7's lesson. A score that sits
  on the pass line is a coin flip, and one pass hides it. Budget the second pass where the
  answer is close, and read the passes side by side.
