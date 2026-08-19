"""Drive the golden set through the Anthropic Batch lane, capture once, half price.

The interactive sweep (`runners/sweep.py`) runs each `(item, model, rep)` through
`Provider.measure` one streamed call at a time. This runner submits every
`(item, rep)` for ONE model as a single batch job, waits for it to end, then
retrieves the answers keyed by custom_id - the async job lifecycle, made literal.

It writes the SAME record schema as the interactive sweep, so the same report,
stats, and pricing read it unchanged - with two honest additions per record:
`mode: "batch"` and `batch_turnaround_s` (the whole-job wall-clock), because a
batch has no per-request latency to compare against the streamed sweep. Cost is
booked at the Batch API's half rate (`cost_usd(..., batch=True)`).

The cache is the spend control, exactly as in the interactive sweep: before
submitting, the runner drops every `(item, rep)` already captured on disk and
submits only the misses - so a re-run, or a resume after a mid-job failure, never
re-spends. Records live under `cache-batch/` so a batch capture never collides
with the interactive capture of the same `(model, item, rep)`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path

from llm_benchmark.dataset import GoldenItem
from llm_benchmark.pricing import cost_usd
from llm_benchmark.providers.anthropic_batch import AnthropicBatchProvider
from llm_benchmark.runners.sweep import capture_path
from llm_benchmark.scorers.base import Scorer

logger = logging.getLogger(__name__)

DEFAULT_BATCH_CACHE_DIR = Path("cache-batch")


def _custom_id(item_id: str, rep: int) -> str:
    """A batch is per-model, so the custom_id only needs to re-identify the
    (item, rep); the model is implied by the job. Stays inside the API's
    `[A-Za-z0-9_-]{1,64}` custom_id rule (item ids use underscores, no dots)."""
    return f"{item_id}__rep{rep}"


async def run_batch_sweep(
    items: list[GoldenItem],
    model: str,
    provider: AnthropicBatchProvider,
    scorer: Scorer,
    reps: int = 2,
    cache_dir: Path = DEFAULT_BATCH_CACHE_DIR,
    force: bool = False,
) -> list[dict]:
    """Run the whole grid for one model as a single batch job.

    Idempotent per capture: an existing capture on disk is loaded and returned
    WITHOUT any submission unless `force=True`. Only the missing `(item, rep)`
    pairs are batched, so the paid job only ever carries fresh work. Returns
    every record (cached + fresh), in item/rep order."""
    cache_dir.mkdir(exist_ok=True)

    # Split the grid into hits (already on disk) and misses (need the job).
    by_custom_id: dict[str, tuple[GoldenItem, int]] = {}
    cached: dict[str, dict] = {}
    prompts: dict[str, str] = {}
    for item in items:
        for rep in range(1, reps + 1):
            cid = _custom_id(item.id, rep)
            by_custom_id[cid] = (item, rep)
            path = capture_path(cache_dir, model, item.id, rep)
            if path.exists() and not force:
                logger.info(
                    "batch.cache HIT model=%s item=%s rep=%d (no submit) -> %s",
                    model,
                    item.id,
                    rep,
                    path,
                )
                cached[cid] = json.loads(path.read_text())
            else:
                prompts[cid] = item.question

    measurements: dict = {}
    errors: dict[str, str] = {}
    turnaround_s = 0.0
    if prompts:
        logger.info(
            "batch.submit model=%s misses=%d (of %d cells)", model, len(prompts), len(by_custom_id)
        )
        measurements, errors = await provider.run(prompts)
        # Every fresh Measurement shares the job turnaround; read it off any one.
        turnaround_s = next((m.latency_ms / 1000.0 for m in measurements.values()), 0.0)
    else:
        logger.info("batch.submit model=%s: all %d cells cached, no job", model, len(by_custom_id))

    for cid, reason in errors.items():
        item, rep = by_custom_id[cid]
        logger.error("batch.result FAILED model=%s item=%s rep=%d: %s", model, item.id, rep, reason)

    # Score each fresh answer with the free local judge and write its record,
    # same schema as the interactive sweep so the report reads it unchanged.
    for cid, m in measurements.items():
        item, rep = by_custom_id[cid]
        verdict = await scorer.score(item.question, m.text, item.expected)
        cost = cost_usd(m, batch=True)
        record = {
            "captured_at": dt.datetime.now(dt.UTC).isoformat(),
            "model": model,
            "item_id": item.id,
            "rep": rep,
            "prompt": item.question,
            "expected": item.expected,
            "measurement": asdict(m),
            "cost_usd": cost,
            "judge": {
                "score": verdict.score,
                "passed": verdict.passed,
                "reason": verdict.reason,
            },
            # Batch-lane provenance: the half-price flag, the one honest latency
            # a batch has (the whole-job turnaround, not per-request), and the
            # provider's own batch_id - the receipt that cross-references the
            # Anthropic console, committed to the record so it can't be lost.
            "mode": "batch",
            "batch_turnaround_s": turnaround_s,
            "batch_id": provider.last_batch_id,
        }
        path = capture_path(cache_dir, model, item.id, rep)
        path.write_text(json.dumps(record, indent=2))
        cached[cid] = record
        logger.info(
            "batch.wrote model=%s item=%s rep=%d cost_usd=%.6g score=%.3f -> %s",
            model,
            item.id,
            rep,
            cost,
            verdict.score,
            path,
        )

    # Return in a stable item/rep order regardless of the batch's result order.
    return [cached[cid] for cid in by_custom_id if cid in cached]


# --- evidence generation (the committed, recomputable proof behind the finding) ---

DEFAULT_EVIDENCE_PATH = Path("evidence/batch-lane-captures.json")


def _lane_stats(records: list[dict]) -> dict:
    """Cost/latency/quality means for one lane over a set of records. Every
    number is read straight off the captured record - never re-derived - so the
    evidence recomputes from the same artifacts the report reads."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    costs = [r["cost_usd"] for r in records]
    lats = [r["measurement"]["latency_ms"] for r in records]
    scores = [r["judge"]["score"] for r in records]
    return {
        "n": n,
        "total_cost_usd": sum(costs),
        "cost_per_capture_usd": sum(costs) / n,
        "mean_latency_ms": sum(lats) / n,
        "mean_judge_score": sum(scores) / n,
    }


def _cell(record: dict) -> tuple[str, int]:
    return (record["item_id"], record["rep"])


def build_batch_evidence(
    batch_records: list[dict],
    model: str,
    batch_id: str | None,
    request_counts: dict | None,
    interactive_cache_dir: Path = Path("cache"),
) -> dict:
    """Assemble the durable batch-lane evidence: the half-price captures, their
    provider batch_id receipt, and a FAIR head-to-head against the interactive
    lane on the SAME (item, rep) cells - not batch-easy vs interactive-everything,
    which would compare quality across different suites."""
    cells = {_cell(r) for r in batch_records}
    interactive: list[dict] = []
    for item_id, rep in sorted(cells):
        path = capture_path(interactive_cache_dir, model, item_id, rep)
        if path.exists():
            interactive.append(json.loads(path.read_text()))
    turnaround = next((r["batch_turnaround_s"] for r in batch_records), 0.0)
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "note": (
            "Ground-truth captures behind the batch-lane claim. cache-batch/ is a "
            "gitignored local spend-control cache (like cache/), so these captures "
            "are committed here as the durable proof. Each record carries the "
            "provider's own token counts; cost is those tokens at half the rate "
            "card (Batch API). The comparison is over the SAME (item, rep) cells "
            "in both lanes, so quality is compared on identical items. The "
            "batch_id below is the Anthropic console cross-reference for this job."
        ),
        "model": model,
        "batch_id": batch_id,
        "request_counts": request_counts,
        "batch_turnaround_s": turnaround,
        "comparison_per_capture": {
            "interactive_streamed": _lane_stats(interactive),
            "batch_half_price": _lane_stats(batch_records),
        },
        "batch_captures": batch_records,
    }


async def _amain(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from llm_benchmark.config import load_dotenv
    from llm_benchmark.dataset import load_golden_set
    from llm_benchmark.providers.anthropic_batch import AnthropicBatchProvider
    from llm_benchmark.runners.sweep import default_scorer

    p = argparse.ArgumentParser(prog="python -m llm_benchmark.runners.batch_sweep")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="smoke: first N items only")
    p.add_argument("--force", action="store_true", help="re-submit even if cached (fresh receipt)")
    p.add_argument("--evidence", default=str(DEFAULT_EVIDENCE_PATH))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set - the batch lane is a PAID call, aborting")
        return 2

    items = load_golden_set()
    if args.limit:
        items = items[: args.limit]
    provider = AnthropicBatchProvider(model=args.model, api_key=api_key)
    records = await run_batch_sweep(
        items, args.model, provider, default_scorer(), reps=args.reps, force=args.force
    )
    evidence = build_batch_evidence(
        records, args.model, provider.last_batch_id, provider.last_request_counts
    )
    out = Path(args.evidence)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2))
    logger.info(
        "batch evidence -> %s  (batch_id=%s, n_batch=%d)",
        out,
        provider.last_batch_id,
        len(records),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
