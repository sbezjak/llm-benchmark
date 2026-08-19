"""The benchmark sweep as a JOB, not a test - the production shape.

A benchmark run is a measurement job that writes an artifact store and a report,
not a pass/fail unit test (you do not want CI to go red because a model scored
0.93). This module is the job's entry point: `python -m llm_benchmark.sweep`.
pytest keeps what it is good at - the mocked harness unit tests and a thin
`billed` smoke - and this drives the real grid and renders the report off the
cache.

  # regenerate the report from already-paid captures ($0, no provider call):
  python -m llm_benchmark.sweep --report-only --html reports/report-<name>.html

  # run the free local baseline only (no paid keys touched):
  python -m llm_benchmark.sweep --no-billed --html reports/report-<name>.html

  # run the full paid grid (real spend - the cache makes a re-run $0):
  python -m llm_benchmark.sweep --html reports/report-<name>.html

The cache is the spend control: `run_sweep` returns a cached capture without
calling the provider, so re-running the job re-spends nothing. `--report-only`
never constructs a provider at all - the safe path when the intent is analysis,
not measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from llm_benchmark.config import load_dotenv
from llm_benchmark.dataset import load_golden_set
from llm_benchmark.report import render_report
from llm_benchmark.runners.stats import format_stats_table, model_stats
from llm_benchmark.runners.sweep import (
    build_default_providers,
    default_scorer,
    format_table,
    load_cached_records,
    run_sweep,
    summarize,
)
from llm_benchmark.scorers.judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL
from llm_benchmark.tracing import NullTracer, Tracer, build_tracer, export_records

logger = logging.getLogger("llm_benchmark.sweep")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m llm_benchmark.sweep",
        description="Run the model benchmark sweep and render its report from the cache.",
    )
    p.add_argument(
        "--html", required=True, help="Report output path, e.g. reports/report-<name>.html"
    )
    p.add_argument("--title", default="Model benchmark - cost / latency / quality")
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="The grader named in the report + used as the per-trace fallback for older "
        "captures that predate the stored judge field. Default: the free local judge.",
    )
    p.add_argument("--reps", type=int, default=2, help="Repetitions per (model, item). Default 2.")
    p.add_argument("--items", default=None, help="Golden set path. Default: the bundled set.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Grade only the first N items - the cheap smoke before a full sweep.",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Render from cached captures only - never construct a provider or call one ($0).",
    )
    p.add_argument(
        "--no-billed",
        action="store_true",
        help="Free local baseline only; paid providers are not built.",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-call providers, ignoring the cache (spends)."
    )
    p.add_argument(
        "--max-spend",
        type=float,
        default=3.0,
        # POST-HOC backstop, not a hard cap: it flags a runaway loudly AFTER the
        # run (non-zero exit), it cannot abort mid-sweep. The real spend control
        # is the cache (a re-run re-spends nothing); this catches a FIRST-time
        # runaway (pricing typo, a model streaming far more tokens than expected).
        # A true pre-flight token estimate would be more machinery than a
        # 10-item, ~$3-ceiling suite warrants.
        help="Loud backstop: exit non-zero if the run's total cost exceeds this. Default $3.",
    )
    return p.parse_args(argv)


def _load_records(args: argparse.Namespace, tracer: Tracer) -> list[dict]:
    """Report-only reads the artifact store; otherwise drive the grid (which
    still reads the cache first and only calls on a miss). The live `tracer` is
    threaded into the grid run; report-only never uses it (there is no call to
    time - that path traces via `export_records` backfill in `main`)."""
    if args.report_only:
        records = load_cached_records()
        # Filter to one suite when --items is given, so a per-suite report renders
        # from the cache without a re-call (and without re-judging - a cache hit is
        # returned verbatim, so the frozen scores are unchanged).
        if args.items:
            suite_ids = {it.id for it in load_golden_set(args.items)}
            records = [r for r in records if r["item_id"] in suite_ids]
        if not records:
            logger.error(
                "--report-only: no matching cached captures under cache/. Run the sweep first."
            )
            sys.exit(2)
        logger.info("report-only: loaded %d cached captures (no provider call)", len(records))
        return records

    items = load_golden_set(args.items) if args.items else load_golden_set()
    if args.limit is not None:
        items = items[: args.limit]
    providers = build_default_providers(include_billed=not args.no_billed)
    logger.info(
        "sweep: %d items x %d models x %d reps%s",
        len(items),
        len(providers),
        args.reps,
        " (FORCE re-call)" if args.force else "",
    )
    return asyncio.run(
        run_sweep(
            items, providers, default_scorer(), reps=args.reps, force=args.force, tracer=tracer
        )
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()  # paid providers read their keys from the env; mirror the pytest path
    args = _parse_args(argv)

    # Live tracing is a real-run concern only: a report-only replay has no call
    # to time, so it stays on the NullTracer and mirrors via `export_records`
    # backfill below. A real run builds a live tracer (no-op without keys).
    tracer = build_tracer() if not args.report_only else NullTracer()

    records = _load_records(args, tracer)
    summaries = summarize(records)
    stats = model_stats(records)
    total = sum(r["cost_usd"] for r in records)

    logger.info(
        "cost / latency / quality (%d captures, total spend $%.5f):\n%s",
        len(records),
        total,
        format_table(summaries),
    )
    logger.info(
        "benchmark validity (score CI / tail latency / cost-per-pass):\n%s",
        format_stats_table(stats),
    )

    out = render_report(records, args.html, title=args.title, judge_model=args.judge_model)
    logger.info("wrote report -> %s", out)

    # Langfuse mirroring, two complementary paths:
    #  - report-only: no live call happened, so backfill the cached records
    #    (no-op without keys/extra; latency reads 0, the known backfill limit).
    #  - a real run: the injected tracer already timed each LIVE call (real
    #    latency + the nested tree), so just flush its buffer; do NOT also
    #    backfill or the same captures would be double-traced.
    if args.report_only:
        export_records(records)
    else:
        tracer.flush()

    if total > args.max_spend:
        logger.error(
            "SPEND BACKSTOP: total $%.5f exceeded --max-spend $%.5f", total, args.max_spend
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
