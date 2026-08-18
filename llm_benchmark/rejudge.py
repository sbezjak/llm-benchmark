"""The judge-choice ablation as a JOB - `python -m llm_benchmark.rejudge`.

Re-grade the cached answers with a different judge and compare the two graders.
The answering models are never re-called (the judge reads cached answer text), so
the only spend is the judge's calls. Same job shape as the sweep: a real entry
point, not a pytest test, with a smoke-sized `--limit` and the cache as the spend
control (a re-judged capture is loaded, not re-paid).

  # smoke: re-judge the first 10 cached answers with the paid gpt judge (~cents)
  python -m llm_benchmark.rejudge --judge gpt-5.6-luna --limit 10

  # full: re-judge every cached answer, then print the cheap-vs-strong comparison
  python -m llm_benchmark.rejudge --judge gpt-5.6-luna

The judge model is one of the answer providers (its key must be in `.env`). The
comparison at the end is the finding: per-model score under each judge, and the
PASS/FAIL disagreements - the cases where the choice of judge changes the verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from llm_benchmark.config import load_dotenv
from llm_benchmark.runners.rejudge import judge_comparison, rejudge_all
from llm_benchmark.runners.sweep import build_default_providers, load_cached_records

logger = logging.getLogger("llm_benchmark.rejudge")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m llm_benchmark.rejudge",
        description="Re-grade cached answers with another judge and compare graders.",
    )
    p.add_argument(
        "--judge",
        default="gpt-5.6-luna",
        help="Judge model (an answer provider; key must be in .env). Default gpt-5.6-luna.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Re-judge only the first N cached answers - the cheap smoke before the full pass.",
    )
    p.add_argument("--force", action="store_true", help="Re-judge even cached verdicts (spends).")
    p.add_argument(
        "--max-spend",
        type=float,
        default=1.0,
        help="Loud backstop: exit non-zero if judge spend exceeds this. Default $1.",
    )
    return p.parse_args(argv)


def _print_comparison(records: list[dict], judged: list[dict]) -> None:
    rows, disagreements = judge_comparison(records, judged)
    lines = [
        f"{'model':<28} {'local':>7} {'judge':>7} {'localP':>8} {'judgeP':>8}",
        "-" * 62,
    ]
    for r in rows:
        lines.append(
            f"{r.model:<28} {r.local_mean:>7.3f} {r.other_mean:>7.3f} "
            f"{r.local_pass:>4}/{r.n:<3} {r.other_pass:>4}/{r.n:<3}"
        )
    logger.info("judge comparison (sweep-local vs re-judge):\n%s", "\n".join(lines))

    logger.info("PASS/FAIL disagreements: %d of %d compared", len(disagreements), len(judged))
    for d in disagreements:
        logger.info(
            "  %-26s %-16s rep%d  local=%.2f/%s  judge=%.2f/%s",
            d["model"],
            d["item_id"],
            d["rep"],
            d["local_score"],
            "P" if d["local_passed"] else "F",
            d["other_score"],
            "P" if d["other_passed"] else "F",
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()
    args = _parse_args(argv)

    providers = build_default_providers()
    if args.judge not in providers:
        logger.error(
            "judge %r has no provider (missing key in .env?). Available: %s",
            args.judge,
            list(providers),
        )
        return 2
    judge_provider = providers[args.judge]

    records = load_cached_records()
    if not records:
        logger.error("no cached answers under cache/. Run the sweep first.")
        return 2
    if args.limit is not None:
        records = records[: args.limit]
    logger.info("rejudge: %d cached answers, judge=%s", len(records), args.judge)

    judged, spend = asyncio.run(
        rejudge_all(records, judge_provider, judge_model=args.judge, force=args.force)
    )
    logger.info("rejudge done: %d verdicts, new judge spend $%.5f", len(judged), spend)

    _print_comparison(records, judged)

    if spend > args.max_spend:
        logger.error(
            "SPEND BACKSTOP: judge spend $%.5f exceeded --max-spend $%.5f", spend, args.max_spend
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
