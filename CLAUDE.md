# CLAUDE.md

## Project

Pytest-based **model benchmarking harness** (Project 5 of the AI/QA portfolio).
Unlike P4 (which built its own agent SUT), P5 **consumes** the earlier work: it
takes P1's eval harness (scorers/judge) and points it at **several models at
once** - gpt-5.6-luna, deepseek-v4-pro, claude-haiku-4-5, claude-sonnet-5, and a
local Ollama baseline (llama3.2) - running the *same* fixed suite through each,
then compares them on three axes:
**cost per query, latency, and quality score**. The deliverable is a report that
says *which model for which use case* - where cheap/fast is good enough, where you
actually need the expensive one.

P5 is the first project that **spends real money** (P0-P4 were local + mocked).
The scope-discipline lessons that keep the suite small are also the cost control;
spend stays in low single-digit dollars. See `plan-benchmarks.md` for the full
cost model and the production-practices table.

Python 3.11+, managed with `uv`. Package under `llm_benchmark/`. The seams:

- `providers/`, adapters over each LLM backend behind one `Provider.generate`
  interface. Only place that issues HTTP calls; tests mock here with `respx`.
  Ollama is vendored from P1 (the free baseline); OpenAI- and Anthropic-shaped
  adapters get added behind the same seam.
- `scorers/`, the quality axis, vendored from P1 (`Scorer.score -> ScoreResult`).
  I/O-free except the LLM-judge, which calls a provider.
- `runners/`, drive the suite across models and capture the raw measurement per
  (model, item): text, tokens in/out, latency, cost.
- `dataset.py`, loads + validates the benchmark suite under `data/` (P1's
  `golden_set.yaml` to start; swappable later - a suite swap + rerun is itself a
  comparison).

## Commands

Use `uv` for all environment and execution tasks:

- Install / sync deps: `uv sync`
- Run all FREE tests (mocked + local Ollama): `uv run pytest -m "not billed"`
- Run only fast (mocked) tests: `uv run pytest -m mocked`
- Run a PAID sweep (real spend, opt-in): `uv run pytest -m billed`
- HTML report: every `uv run pytest` writes a UNIQUE `reports/report-<UTC
  timestamp>.html` (conftest auto-rewrites the default so no run overwrites
  another). A report to COMMIT/host is generated with an explicit descriptive name.
- Lint: `uv run ruff check .`  Format: `uv run ruff format .`

## Test conventions (configured in pyproject.toml)

- `asyncio_mode = "auto"`, async tests do not need `@pytest.mark.asyncio`.
- Markers gate environment-dependent tests:
  - `@pytest.mark.mocked`, uses `respx` to mock; default for unit tests.
  - `@pytest.mark.live`, requires the local Ollama backend (free, slow).
  - `@pytest.mark.billed`, hits a PAID provider API - real spend. Excluded from
    default runs; the spend gate. Never fire a billed sweep by accident; smoke
    test on 1-2 items before any full sweep.
- `testpaths = ["tests"]`; `ruff` line length 100, target `py311`.

## Architecture intent

Preserve these seams when adding code:

- HTTP calls only inside `providers/`. Tests mock here with `respx` (no live
  network in unit tests, no billed calls in default runs).
- Every (model, item) call is captured to disk once (raw prompt + response +
  tokens + latency); all cost/latency/quality math runs off the cached artifact,
  never a re-call. Satisfies both cost control and "raw output is ground truth."
- Scorers stay I/O-free except the LLM-judge-on-answer.
- Smoke test each new provider adapter on 1-2 fabricated items before a full
  sweep; arm a log monitor on any non-instant run.

---

## Working style with this user

- **`human-tasks.md` is the channel for anything only the user can do.**
  Whenever the next step needs the user (post an update, paste a real URL,
  decide a squishy ground-truth call, run an interactive login), append a
  checkbox item to `human-tasks.md` instead of only mentioning it in chat -
  chat scrolls, the file persists. Read it when picking up work. Named
  `human-tasks`, not `tasks`, so it isn't confused with a generic task tool.
- **Prepare drafts/templates for any task the user has to do by hand.**
  When the next step is something only the user can do, prepare a
  fill-in-the-blanks file with the structure pre-built. Don't make the user
  start from a blank page. Reduce the user's task to filling in the squishy
  parts.
- **Capture explanations to `notes.md` when teaching.** When the user asks
  "explain this to me" and the answer is non-trivial (why an agent failure
  mode happens, how a trace checker decides, OWASP LLM Top 10 / LLM06
  mechanics), mirror it (lightly cleaned up) into `notes.md` as reference.
  The chat scrolls; the article stays.
- **Two writing registers, no duplicate copies.** `notes.md` is the dense,
  finding-first record. The teaching / front-door artifacts (the narrated
  walkthrough, `docs/` explainers, README) use the plain, layered voice the
  user likes. Same explanation in two registers drifts out of sync, so the
  registers serve different artifacts, never two copies of one thing.
- **Ground each front-door artifact in the previous projects' versions
  before drafting.** The article, walkthrough, and LinkedIn post come out
  measurably better when the model first reads the *same* artifact from the
  earlier projects (A/B confirmed). Links live in `PORTFOLIO.md`; read the
  LOCAL sibling repos (`../llm-eval-harness/`, `../llm-rag/`,
  `../llm-api-testing/`, `../llm-red/`) - faster, and LinkedIn URLs are
  auth-walled. For a LinkedIn post, match the shipped voice: curiosity /
  first-person hook, one finding told as a story, an easy-case-vs-hard-case
  contrast, no arrows / hashtags / stacked numbers, closing line exactly
  `Automation engineer learning AI testing. Project N of 5. More from the
  series: <link>`.
- **Default to production / best-practice solutions; take the pragmatic
  shortcut only when the trade-off is justified for this project's scope,
  and call the trade-off out explicitly.** Name what the production-grade
  pattern would be, name why we're not doing it here, and write the
  trade-off into `notes.md` so the writeup shows it was a deliberate choice.
- **Validate a new integration cheaply before any long or expensive run.**
  Prove it works on the smallest possible input first (1-2 items, fabricated
  inputs are fine) and confirm the output parses. ALWAYS arm a monitor on
  the log for any run that is not near-instant - grepping for success AND
  failure signatures
  (`500|Internal Server Error|Traceback|Error|Exception|Killed|OOM|NaN|Timeout|Failed`)
  so a mid-run failure surfaces at the failing job. Smoke test first, monitor
  second, both every time. When a run fails, fix the root cause and
  re-validate cheaply before re-running full.
- **Always make model prompts and responses visible in test reports.**
  Every component that calls a model (providers, LLM-judge checkers) must log
  the prompt going in and the response coming out at `INFO` level via the
  stdlib `logging` module, not truncated. The always-on pytest-html report
  captures these. For an agent project this is how you read the whole trace,
  not just the final answer.
- **Read the report/logs thoroughly - the raw model output / trace is the
  only ground truth, never conclude from a summary boolean, a count, or a
  keyword/length heuristic when the actual text is available.** Open the
  trace and read it end to end; that reading IS the work, not an optional
  check.
- **Preserve the FULL run log for anything a claim rests on - never truncate
  it through `tail`/`head`/`tee | tail`.** The raw log carries the receipts
  (provider request_id / batch_id, verbatim usage, per-item traces) that let a
  claim be verified or recomputed; a curated `.md` summary is a read, not proof.
  Three enforced tiers: (1) MECHANISM - redirect the whole stream to a durable
  file: `scripts/run-evidence.sh <name> -- <cmd>` (or `> evidence/raw-logs/<name>.log
  2>&1`); (2) TIER-2 DEMAND - before stating a finding from a run, quote its
  receipt line verbatim from the committed `evidence/raw-logs/<name>.log` (you
  cannot quote a batch_id you never saved); (3) TIER-3 GUARD - a PreToolUse Bash
  hook (`scripts/guard-truncated-logs.sh`) warns when a run is piped through
  tail/head. Also emit a machine-checkable receipts JSON next to any summarized
  finding (e.g. `evidence/semantic-cache-receipts.json`) so the summary is
  recomputable, not taken on faith. (Lost the batch run log once to `tee | tail`;
  this is the fix.)
- **Prefer the smallest solution that solves the problem; after writing,
  re-read and cut machinery the task didn't ask for.** Treat the re-read as
  a step: "is half of this scaffolding I invented but nobody asked for?" If
  yes, rewrite it small.
- In all prose (docs, comments, commit messages, PR descriptions), join
  clauses with a single hyphen `-`, a comma, a period, or parentheses. The
  only dash character in written text is a single `-`.
- **Name committed artifacts by finding-id / topic, never by build stage.**
  Evidence files, reports, and code comments use the durable public vocabulary
  (`F1-ablation-schema-example.md`, `baseline-trace-calculator.md`, dated report
  names) from creation - session labels (`S1` / `S3a` / `S5c-P2`) live only in
  the gitignored working notes. Stage labels in committed names force a later
  cleanup pass; set the naming convention before the first evidence/report file
  exists.
- End commit messages at the body. The user is the sole author.
