# Tracing a benchmark run: live vs backfill

This project can send its results to Langfuse (an LLM observability platform) in
two different ways. They look similar in the dashboard but one of them quietly
gets a key number wrong, and knowing why is the whole point of this note.

## The two ways

**Backfill** (`export_records`, used by `--report-only`). Read the finished
records off the `cache/` disk artifact and replay them into Langfuse after the
fact. Nothing is re-run - it reads numbers the models were already paid for and
mirrors them up. This is the right tool when you just want to populate a dashboard
from captures you already have, for free.

**Live instrumentation** (the `Tracer` seam, used by a real sweep run). Wrap the
*actual* model call as it happens, so the trace is recorded in real time. This is
the standard, idiomatic way to use a platform like Langfuse.

## Why backfill gets latency wrong

Langfuse measures how long an operation took from the wall-clock gap between when
a span *starts* and when it *ends*. That works when the span wraps a real call
that genuinely takes a few seconds.

A backfilled span doesn't wrap anything - it's created and closed in the same
instant, just to carry already-finished data. So its start and end are the same
moment, and Langfuse reads its duration as **zero**. Every latency in the
dashboard shows 0 ms, even though the real calls took seconds. The true latency is
sitting in the record's metadata, but the latency column never looks there.

Cost, token counts, and quality scores all backfill faithfully. Latency is the one
number that can't survive being replayed, because latency is a property of *time
passing*, and a replay has no time passing.

## Why live instrumentation fixes it

When you wrap the real call, the span is open for exactly as long as the call
takes. Langfuse's start-to-end measurement is now measuring the real thing, so the
latency is real and correct - no extra work, it falls out of the timing.

Live tracing also gives you the **shape** of the run that backfill can't
reconstruct: a nested tree. One capture becomes a root span with two children -
the answer generation and the judge evaluation - each timed on its own. Backfill
flattens that into separate, unrelated entries.

## When each is used here

Both are kept, because they're good at different things:

- **A real run** (`python -m llm_benchmark.sweep ...` with keys present) uses the
  live tracer. Real calls, real latency, the nested tree.
- **`--report-only`** replays the cache and uses backfill. There are no live calls
  to wrap in that mode, so backfill is the only option - and its latency-is-0
  limitation doesn't matter, because report-only is for regenerating the HTML
  report, not the latency dashboard.

The choice is made once, in the job, by which mode you ran - you never think about
it at the call site.

## The design rule that made this safe

The tracer is injected at the **runner**, never inside `providers/`, and it
defaults to a no-op (`NullTracer`). Only the sweep *job* builds a real
`LangfuseTracer`, and only when the keys are present. So the test suite and any
free/local run stay completely offline without any special flags - they simply get
the no-op tracer and never touch the network. Tracing is something the job opts
into, not something baked into the code that talks to the models.
