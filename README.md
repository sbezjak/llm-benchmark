# llm-benchmark

> Part of a [5-project AI/QA testing portfolio](https://github.com/sbezjak/sbezjak) - all projects and write-ups.

A pytest-based harness that runs one fixed suite through several LLM providers at
once and compares them on the three axes a team actually pays for: **cost per
query, latency, and answer quality**. It reuses Project 1's scorers and
LLM-as-judge as the quality axis and points them at five models side by side.

The headline is not "which model won." The paid models tie on quality, so the
pick falls to cost and speed - and the more useful finding is about the **grader**:
a cheap judge rates a weak model right alongside frontier ones and hides the gap
that a reliable judge exposes. The benchmark asks "cheap model or expensive one?";
this project turns the same question on the evaluator.

Built as project 5 of 5 exploring AI/LLM testing. It is the first that **spends
real money** (projects 0-4 were local + mocked); the whole comparison here cost
about $0.21 - answers plus the paid judging that graded them. A writeup is in
progress.

Live reports (published after a push, self-contained HTML):
[scoreboard](https://sbezjak.github.io/llm-benchmark/reports/report-benchmark-scoreboard-2026-08-17.html) ·
[golden sweep](https://sbezjak.github.io/llm-benchmark/reports/report-golden-sweep-2026-08-17.html) ·
[adversarial sweep](https://sbezjak.github.io/llm-benchmark/reports/report-adversarial-sweep-2026-08-17.html)

Python 3.11+, managed with [`uv`](https://docs.astral.sh/uv/).

## The idea

Non-deterministic output makes "which model is best" a testing question, not a
spec-sheet one. This harness sends the *same* items to each model, scores the
answers the same way, and records what each call cost and how long it took - so
the comparison is apples to apples. Then it does the honest second step most
leaderboards skip: it puts an error bar on every score and checks whether the
grader that produced those scores can be trusted.

**Finding vs sample.** These are non-deterministic systems - a re-roll moves every
number and can reorder the leaderboard. The claims below are the *shape* that
survives re-rolls (a direction, or an interval); the exact scores are one run's
sample (the 2026-08-17 roll), committed as receipts, not as stable truths. Every
number regenerates from the on-disk cache for $0 (`reports/findings.json`).

Models compared - one local free baseline plus four paid, behind one
`Provider.generate` seam:

| Model | Backend | Role |
|---|---|---|
| `llama3.2` | local [Ollama](https://ollama.com/) | free baseline **and** the cheap judge (a contestant grading its own pool) |
| `gpt-5.6-luna` | OpenAI | paid; also the reliable paid judge in the ablation |
| `deepseek-v4-pro` | DeepSeek (OpenAI-compatible seam) | paid |
| `claude-haiku-4-5-20251001` | Anthropic | paid |
| `claude-sonnet-5` | Anthropic | paid |

## Findings

Each finding is written up with its evidence; the right column is where to read
it - a committed analysis (`evidence/*.md`, a read) backed by a machine-checkable
receipt (`*.json`, the recompute), or a test.

| # | Finding | Where |
|---|---|---|
| 1 | The paid pack is a statistical **tie on quality** - every paired interval includes 0 - so the differentiator is cost and latency, not score. The cheapest, fastest paid model is the pick precisely because the pricier ones don't reliably score higher. | `evidence/benchmark-validity-analysis.md`, `full-sweep-comparison-table.md` |
| 2 | **The cheap judge hides the real gap** (the capstone). On the hard suite the free local judge scores the weak model right in the pack (a perfect pass rate); re-graded by a reliable paid judge, its score and pass rate drop while the paid pack holds. The cheap judge compresses the gap away; a better instrument separates the weak model out. | `evidence/judge-ablation-analysis.md`, `adversarial-suite-analysis.md` |
| 3 | **Position bias.** Show the cheap judge two answers, then the same two swapped, and count a win only if it holds: it flips on order 2-3x more than the paid judge, and most of its flips just keep whichever answer was shown first. Order, not quality, drove the vote - and its own stored reasoning contradicts itself pair by pair. | `evidence/judge-position-bias.md` (+ `-receipts.json`, `-stats.json`) |
| 4 | **Self-preference.** Grading a pool it competes in, the cheap local judge ranks its *own* answers up (third of five head-to-head) while it scores last on its own. A model should not grade a pool it is in. | `evidence/judge-position-bias.md`, `llm_benchmark/freeze_evidence.py` |
| 5 | **Human calibration.** A blind human grade of a 24-answer sample agreed with the paid judge on pass/fail every time; the judge passed nothing the human called wrong. The only friction was the human's own uncertainty on borderline *presentation* of correct answers. | `evidence/judge-calibration-analysis.md`, `docs/judge-calibration-grading-sheet.md` |
| 6 | **Cost levers, measured not assumed.** A batch lane buys ~half price for minutes of turnaround at tied quality; a semantic answer cache has false hits at *every* similarity threshold (no setting both keeps paraphrases and rejects a trap); prefix caching does not apply to this suite (the items share no preamble) - and reporting that null is the finding. | `evidence/batch-lane-captures.json`, `semantic-cache-false-hits.md`, `findings.json` (`candidates`) |
| 7 | **Two reps caught a wobble one run would have faked.** Several answers sit right on the judge's 0.7 pass threshold and cross back and forth between reps; the aggregate hides it, the row-by-row trace shows it. A single-run sweep would have labelled quality by chance. | `evidence/full-sweep-comparison-table.md` |

## Quickstart

```bash
uv sync                                   # install runtime + dev deps
ollama pull llama3.2                      # ~2 GB, one-time (the free baseline + cheap judge)

uv run pytest -m mocked                   # fast unit tests, no network (~1 s)
uv run pytest -m "not billed"             # FREE: mocked + live local Ollama
uv run pytest -m billed                   # PAID sweep smoke - real spend, opt in explicitly

uv run ruff check .                       # lint
uv run ruff format .                      # format
```

Paid providers read their keys from a local `.env` (never committed):

```
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
ANTHROPIC_API_KEY=...
```

The `billed` marker is the spend gate: the free run excludes it with
`-m "not billed"`, and each billed test also skips itself when its key is absent -
so routine runs cost nothing unless you opt in explicitly. Real spend for a full
comparison is single-digit dollars.

### The sweep as a job (not a test)

A benchmark run is a measurement *job*, not a pass/fail unit test - you don't want
CI to go red because a model scored 0.93. So the grid runs from its own entry
point, and the report renders off the cache:

```bash
# regenerate any report from already-paid captures ($0, never calls a provider):
uv run python -m llm_benchmark.sweep --report-only --items data/golden_set.yaml \
  --html reports/report-golden-sweep-2026-08-17.html --title "Model benchmark - golden set"

# run the free local baseline only (no paid keys touched):
uv run python -m llm_benchmark.sweep --no-billed --html reports/report-<name>.html

# run the full paid grid (real spend - the cache makes a re-run $0):
uv run python -m llm_benchmark.sweep --html reports/report-<name>.html --max-spend 3.0

# rebuild findings.json off the cache, then render the scoreboard ($0):
uv run python -m llm_benchmark.runners.findings
uv run python -m llm_benchmark.report_dashboard --out reports/report-benchmark-scoreboard-2026-08-17.html
```

The cache is the spend control: `run_sweep` returns a cached capture without
calling the provider, so re-running spends nothing. `--report-only` never even
constructs a provider - the safe path when the intent is analysis, not measurement.

## How it's built

The seams, each kept deliberately narrow:

```
llm_benchmark/
├── providers/         # one Provider.generate per backend; the ONLY place HTTP happens
│   ├── ollama.py            #   local baseline (vendored from project 1) + ollama_embed
│   ├── openai_compat.py     #   OpenAI + DeepSeek (base_url swap) behind one class
│   ├── anthropic.py         #   Anthropic messages + anthropic_batch (the batch lane)
├── scorers/           # the quality axis (from project 1); I/O-free except the judge
│   └── judge.py             #   LLM-as-judge: correctness + relevance -> [0,1]
├── runners/           # drive the grid; compute stats/findings OFF the cache
│   ├── sweep.py             #   (model, item, rep) -> captured record on disk
│   ├── stats.py             #   bootstrap CI, paired significance, tail latency
│   ├── findings.py          #   aggregate caches -> findings.json (pure, $0)
│   ├── pairwise.py          #   A-vs-B both-orders lens (the position-bias control)
│   └── rejudge.py           #   re-grade cached answers with a second judge ($0 re-call)
├── sweep.py           # the JOB entry point (python -m llm_benchmark.sweep)
├── report.py          # cached records -> the full-trace HTML report
├── report_dashboard.py# findings.json -> the scoreboard
├── pricing.py         # per-provider token pricing (tokenizers differ)
└── dataset.py         # YAML loader + validation
data/
├── golden_set.yaml            # the easy suite (from project 1)
├── adversarial_set.yaml       # harder items with knowable answers but real traps
└── semantic_cache_probes.yaml # paraphrase + trap probes for the cache false-hit test
docs/, evidence/               # the reads (.md) and the receipts (.json / raw logs)
```

Two architecture rules make the numbers trustworthy and cheap:

- **Capture once, math off the artifact.** Every `(model, item, rep)` call is
  written to disk once - prompt, response, provider token counts, latency, TTFT,
  cost, `request_id`, and the judge's verdict. All cost/latency/quality math runs
  off that cached record, never a re-call. This is both the spend control and the
  reason the raw output is the ground truth.
- **Compute / render split.** `findings.py` aggregates the caches into
  `findings.json`; `report.py` and `report_dashboard.py` only render. So the
  analysis gets a $0 unit test and the reports regenerate for free whenever the
  reading changes - the bootstrap is seeded, so a regenerate is byte-identical bar
  its timestamp.

## Testing

Markers gate environment-dependent tests (configured in `pyproject.toml`):

| Marker | Purpose | Cost |
|---|---|---|
| `mocked` | `respx`-mocked providers; provider/scorer/stats logic, dataset validation | free, ~1 s |
| `live` | requires local Ollama (the free baseline) | free, slow |
| `billed` | hits a PAID provider API - the spend gate, **excluded from default runs** | real $ |

`asyncio_mode = "auto"` is set, so async tests don't need `@pytest.mark.asyncio`.
Every component that calls a model logs the full prompt and response at `INFO`, so
a report's trace shows exactly what each model saw and said. Each new provider
adapter is smoke-tested on 1-2 fabricated items before any full sweep.

## Reports

Two kinds, both self-contained HTML:

- **Sweep + scoreboard** (committed, named by topic): `report.py` renders the full
  verbatim trace - one collapsible block per `(model, item, rep)`, with a preview
  of each answer in the always-visible summary so a wrong answer is scannable
  without expanding it. `report_dashboard.py` renders the plain-table scoreboard.
  Both regenerate off the cache for $0.
- **pytest-html** (auto, gitignored): every `uv run pytest` writes a unique
  `reports/report-<UTC timestamp>.html` so no run overwrites another.

The raw trace is the ground truth - not the pass/fail count, which is only a
summary. Read the verbatim answers, not just the tallies.

## Limits (do not overclaim)

- **No single best model.** The paid pack is too close to rank, and the grader is
  biased - the honest output is "tie, pick on price/speed," not a leaderboard.
- **Small n.** Two reps of a 20-item suite can't resolve a fraction-of-a-point gap;
  the tie is a decision from that, stated as such.
- **The judge is one of the tested models.** `llama3.2` grades because it is the
  free local baseline - a weak, in-pool grader. That failure *is* the finding, not
  a result to trust.

**What production would add:** a grader outside the pool (or a panel by majority);
larger, standard, public suites; extra reps only where scores are close; and human
spot-checks with drift monitoring. This harness demonstrates the hazards a
real setup is built to remove.

## How this was built

Built session by session with Claude Code against a written scope plan, so it
stayed small and the spend stayed in single-digit dollars. The work ran both ways:
I drove the model, but it also kept a queue of tasks for *me* - the calls only a
person should make (the blind human calibration grade, deciding when a re-rolled
number was a sample not a finding, supplying the real repo URL). `CLAUDE.md`,
committed next to this README, is the standing instruction set the assistant works
from: the seams to keep, what counts as evidence, the working style.

## Further reading

The canonical sources behind this project, worth reading first for the overview a
hands-on build does not give:

- Zheng et al. 2023, [Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena](https://arxiv.org/abs/2306.05685) - the reference study for using a model
  to grade model output. It names, by controlled experiment, the exact judge
  biases this project reproduces by hand: **position** bias and **self-enhancement**
  (Findings 3-4).
- Chiang et al. 2024, [Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference](https://arxiv.org/abs/2403.04132) - the large-scale pairwise / Elo
  approach the A-vs-B lens here mirrors in miniature, and the argument for
  preference-based ranking the absolute-score tie motivates.
- Anthropic [Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)
  and OpenAI [Batch API](https://platform.openai.com/docs/guides/batch) - the ~50%
  off, higher-latency lane measured as a cost lever in Finding 6.
- Efron & Tibshirani 1993, *An Introduction to the Bootstrap* - the resampling
  behind every 95% interval here; the reason a leaderboard of bare means invites a
  false read.
