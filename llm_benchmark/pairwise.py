"""The pairwise / Elo ranking lens as a JOB - `python -m llm_benchmark.pairwise`.

The absolute lens scored each answer on its own and ranked by the mean.
This is the other production pattern - the LMSYS-Arena / Elo shape: show a judge
two models' answers to the SAME item and ask which is better, then aggregate the
head-to-head wins into a ranking and check whether it AGREES with the absolute
one (`kendall_tau`). It answers "which model when" directly - A beats B - without
needing a calibrated absolute scale.

No model is re-called: the judge reads the cached answer TEXT already on disk, so
the only spend is the judge's own calls (a re-judged pair is loaded, not re-paid -
the same cache spend-control the sweep uses). Two judges are meant to be run:

  # $0 free-local arm - its flip rate MEASURES the cheap judge's position bias:
  python -m llm_benchmark.pairwise --judge llama3.2

  # paid arm (the authoritative ranking, comparable to the gpt-judged absolute):
  python -m llm_benchmark.pairwise --judge gpt-5.6-luna --limit 1   # smoke first
  python -m llm_benchmark.pairwise --judge gpt-5.6-luna             # full

Each judge's verdicts land in its own `cache/pairwise/<judge>/` tree, so the two
arms never collide and either re-runs for $0. The paid judge is deliberately NOT
the strongest model (claude-sonnet-5): that model tops the absolute ranking, so
judging with it would let it grade its own answers (self-preference bias) and
break comparability with the absolute ranking. gpt-5.6-luna is mid-pack and the paid absolute judge.
The production-grade choice would be a judge OUTSIDE the contestant pool (or a
panel); every paid provider here is also a contestant, so we use a single in-pool
judge and report its self-preference as a caveat.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from llm_benchmark.config import load_dotenv
from llm_benchmark.runners.findings import PAID_JUDGE, suite_map
from llm_benchmark.runners.pairwise import (
    DEFAULT_PAIRWISE_DIR,
    flip_rate,
    kendall_tau,
    run_pairwise,
    standings,
)
from llm_benchmark.runners.rejudge import CostCapturingJudge
from llm_benchmark.runners.sweep import build_default_providers, load_cached_records

logger = logging.getLogger("llm_benchmark.pairwise")

DEFAULT_FINDINGS_PATH = Path("reports") / "findings.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m llm_benchmark.pairwise",
        description="Rank the models head-to-head (pairwise/Elo) off cached answers.",
    )
    p.add_argument(
        "--judge",
        default="gpt-5.6-luna",
        help="Judge model (a provider; a paid judge's key must be in .env). Default gpt-5.6-luna.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N items PER SUITE - the cheap smoke before the full run.",
    )
    p.add_argument("--force", action="store_true", help="Re-judge even cached pairs (spends).")
    p.add_argument(
        "--max-spend",
        type=float,
        default=1.0,
        help="Loud backstop: exit non-zero if judge spend exceeds this. Default $1.",
    )
    return p.parse_args(argv)


def _canonical_answers(records: list[dict]) -> dict[tuple[str, str], dict]:
    """One answer per (model, item_id) for the pairwise comparison - the LOWEST
    rep, so the choice is deterministic (the sweep captures reps 1..N; pairwise
    ranks a single canonical answer per model, not a rep average)."""
    best: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["model"], r["item_id"])
        cur = best.get(key)
        if cur is None or r["rep"] < cur["rep"]:
            best[key] = r
    return {k: {"text": r["measurement"]["text"]} for k, r in best.items()}


def _items_by_suite(records: list[dict], smap: dict[str, str]) -> dict[str, list[dict]]:
    """Per-suite ordered item list ({item_id, prompt, expected}), deduped from the
    records so we depend on the cache, not on re-reading the golden-set field
    names. Sorted by item_id for a stable round-robin order."""
    by_suite: dict[str, dict[str, dict]] = {}
    for r in records:
        suite = smap.get(r["item_id"])
        if suite is None:
            logger.warning("item %s has no suite in the golden sets - skipping", r["item_id"])
            continue
        items = by_suite.setdefault(suite, {})
        if r["item_id"] not in items:
            items[r["item_id"]] = {
                "item_id": r["item_id"],
                "prompt": r["prompt"],
                "expected": r["expected"],
            }
    return {suite: [items[i] for i in sorted(items)] for suite, items in by_suite.items()}


def _absolute_ranking(suite: str, findings_path: Path = DEFAULT_FINDINGS_PATH) -> list[str] | None:
    """The paid-judge absolute ranking for a suite (models, best first), read
    off `findings.json`. Returns None if the artifact is absent, so the pairwise
    run still produces standings - the agreement check is a bonus, not a gate."""
    if not findings_path.exists():
        logger.warning("no %s - skipping absolute-vs-pairwise agreement", findings_path)
        return None
    findings = json.loads(findings_path.read_text())
    view = findings.get("views", {}).get(suite, {}).get(PAID_JUDGE)
    if not view:
        logger.warning("no %s/%s view in findings - skipping agreement", suite, PAID_JUDGE)
        return None
    return [row["model"] for row in view["models"]]


def _report_suite(suite: str, comparisons: list[dict], models: list[str]) -> None:
    """Log one suite's standings, its judge flip rate (the position-bias number),
    and its agreement (kendall_tau) with the absolute ranking."""
    rows = standings(models, comparisons)
    lines = [
        (
            f"{'model':<28} {'games':>5} {'win':>4} {'loss':>4} {'tie':>4} {'flip':>4} "
            f"{'win%':>6} {'elo':>6}"
        ),
        "-" * 70,
    ]
    for s in rows:
        lines.append(
            f"{s.model:<28} {s.games:>5} {s.wins:>4} {s.losses:>4} {s.genuine_ties:>4} "
            f"{s.flips:>4} {s.win_rate:>6.1%} {s.elo:>6.0f}"
        )
    logger.info("[%s] pairwise standings:\n%s", suite, "\n".join(lines))
    logger.info("[%s] judge flip rate (position bias): %.1f%%", suite, 100 * flip_rate(comparisons))

    pairwise_order = [s.model for s in rows]
    absolute_order = _absolute_ranking(suite)
    if absolute_order is not None:
        tau = kendall_tau(pairwise_order, absolute_order)
        logger.info(
            "[%s] absolute-vs-pairwise agreement kendall_tau=%.3f\n  absolute: %s\n  pairwise: %s",
            suite,
            tau,
            absolute_order,
            pairwise_order,
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()  # a paid judge reads its key from the env; mirror the other jobs
    args = _parse_args(argv)

    providers = build_default_providers()
    if args.judge not in providers:
        logger.error(
            "judge %r has no provider (missing key in .env?). Available: %s",
            args.judge,
            list(providers),
        )
        return 2
    capture = CostCapturingJudge(providers[args.judge])

    async def judge_fn(prompt: str) -> str:
        # Make the judge's prompt+reply visible in the report (CLAUDE.md: every
        # model call logs its full prompt in and response out at INFO).
        resp = await capture(prompt)
        logger.info("pairwise.judge PROMPT:\n%s\nRESPONSE:\n%s", prompt, resp)
        return resp

    records = load_cached_records()
    if not records:
        logger.error("no cached answers under cache/. Run the sweep first.")
        return 2

    smap = suite_map()
    answers = _canonical_answers(records)
    items_by_suite = _items_by_suite(records, smap)
    models = sorted({r["model"] for r in records})
    pairwise_dir = DEFAULT_PAIRWISE_DIR / args.judge  # per-judge tree; arms never collide

    logger.info(
        "pairwise: judge=%s, %d models, suites=%s -> %s",
        args.judge,
        len(models),
        {s: len(its) for s, its in items_by_suite.items()},
        pairwise_dir,
    )

    for suite in sorted(items_by_suite):
        items = items_by_suite[suite]
        if args.limit is not None:
            items = items[: args.limit]
        comparisons = asyncio.run(
            run_pairwise(answers, items, models, suite, judge_fn, pairwise_dir, args.force)
        )
        _report_suite(suite, comparisons, models)

    logger.info(
        "pairwise done: judge=%s, %d calls, new judge spend $%.5f",
        args.judge,
        capture.calls,
        capture.total_cost_usd,
    )
    if capture.total_cost_usd > args.max_spend:
        logger.error(
            "SPEND BACKSTOP: judge spend $%.5f exceeded --max-spend $%.5f",
            capture.total_cost_usd,
            args.max_spend,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
