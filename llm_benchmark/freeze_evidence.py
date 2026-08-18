"""Freeze the pairwise-judge evidence off the on-disk cache - the missing generator.

  python -m llm_benchmark.freeze_evidence

The position-bias finding rests on the pairwise verdict cache (`cache/pairwise/`),
but the committed *evidence* copies under `evidence/pairwise-verdicts/` were frozen
once by hand and then went stale when the 2026-08-17 rerun re-rolled every answer.
Hand-copied evidence has no generator, so it silently drifts from the cache it
claims to quote. This module is that generator: it re-freezes the raw verdicts and
emits the machine-checkable receipts, all pure and $0 (reads the cache, never calls
a model), mirroring how `semantic_cache_run` writes a read (.md) beside a receipt
(.json).

Three artifacts, in the same compute/render spirit as the rest of the harness:

  1. `evidence/pairwise-verdicts/<arm>/*.json` - a verbatim copy of every cached
     verdict (the RAW receipt: both winners + both reasonings per pair). This is
     the ground truth the .md quotes; freezing it means the quote is checkable.
  2. `evidence/judge-position-bias-receipts.json` - the free-local judge's FLIPS
     reshaped into both-order receipts (shown_first / picked / reason), each
     pointing at its frozen verdict file. The full set behind the .md's examples.
  3. `evidence/judge-position-bias-stats.json` - every number the .md prose cites,
     computed here per (arm, suite): flip rate, first-answer-bias, the
     "Answer A ..." opener rate, self-preference standing. So the .md is a read and
     this is the recompute (cross-checks against `findings.json`, same cache).

The narrative .md itself (`judge-position-bias.md`) is written by hand from these
numbers - prose is a read, not proof - the way CLAUDE.md splits the two registers.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from llm_benchmark.runners.pairwise import DEFAULT_PAIRWISE_DIR, standings

logger = logging.getLogger("llm_benchmark.freeze_evidence")

# The two judge arms live in sibling subdirs of the pairwise cache, named by the
# judging model. The free local judge is the finding's subject; the paid judge is
# the contrast (a better instrument still needs the both-orders control).
FREE_JUDGE_ARM = "llama3.2"
PAID_JUDGE_ARM = "gpt-5.6-luna"

DEFAULT_EVIDENCE_DIR = Path("evidence")


def iter_verdicts(arm_dir: Path) -> list[dict]:
    """Every cached verdict under one arm, sorted for a deterministic freeze."""
    return [json.loads(p.read_text()) for p in sorted(arm_dir.glob("*.json"))]


def _opens_with_answer_a(reason: str) -> bool:
    # The tell: the judge names slot A first, then reasons to fit it. Matches the
    # .md's "% of flip reasonings that begin with 'Answer A'".
    return reason.strip().lower().startswith("answer a")


def receipt_for(verdict: dict, arm: str, evidence_dir: Path) -> dict:
    """Reshape one verdict into a both-orders receipt. order1 shows model_a first
    (slot A); order2 shows model_b first - so `shown_first` is model_a then model_b,
    and `picked` is the winner the judge named in that order (see
    `runners.pairwise.compare_pair`)."""
    a, b = verdict["model_a"], verdict["model_b"]
    name = f"{verdict['suite']}__{verdict['item_id']}__{a}__{b}.json"
    return {
        "suite": verdict["suite"],
        "item": verdict["item_id"],
        "model_a": a,
        "model_b": b,
        "order1": {
            "shown_first": a,
            "picked": verdict["order1_winner"],
            "reason": verdict["order1_reasoning"],
        },
        "order2": {
            "shown_first": b,
            "picked": verdict["order2_winner"],
            "reason": verdict["order2_reasoning"],
        },
        "file": str(evidence_dir / "pairwise-verdicts" / arm / name),
    }


def freeze_verdicts(pairwise_dir: Path, evidence_dir: Path) -> dict[str, int]:
    """Mirror-copy every cached verdict into evidence/, arm by arm. Returns the
    per-arm count so a caller (and the test) can assert the freeze is complete."""
    counts: dict[str, int] = {}
    for arm in (FREE_JUDGE_ARM, PAID_JUDGE_ARM):
        src = pairwise_dir / arm
        dst = evidence_dir / "pairwise-verdicts" / arm
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(src.glob("*.json")):
            # Re-dump (not shutil.copy) so the frozen copy is canonical JSON even if
            # the cache file's formatting ever drifts.
            (dst / p.name).write_text(json.dumps(json.loads(p.read_text()), indent=2))
            n += 1
        counts[arm] = n
    return counts


def position_bias_receipts(pairwise_dir: Path, evidence_dir: Path, arm: str) -> list[dict]:
    """The free-local judge's FLIPS as both-order receipts (the .md's example set)."""
    out = [
        receipt_for(v, arm, evidence_dir)
        for v in iter_verdicts(pairwise_dir / arm)
        if v["outcome"] == "flip"
    ]
    out.sort(key=lambda r: (r["suite"], r["item"], r["model_a"], r["model_b"]))
    return out


def _arm_stats(verdicts: list[dict]) -> dict:
    """Per-suite bias numbers for one judge arm, computed straight off its verdicts."""
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        by_suite[v["suite"]].append(v)

    stats: dict[str, dict] = {}
    for suite, vs in sorted(by_suite.items()):
        flips = [v for v in vs if v["outcome"] == "flip"]
        # first-answer bias: on a flip, the judge picked whoever sat in slot A BOTH
        # times (order1 -> model_a, order2 -> model_b).
        first_answer = [
            v
            for v in flips
            if v["order1_winner"] == v["model_a"] and v["order2_winner"] == v["model_b"]
        ]
        flip_reasons = [v["order1_reasoning"] for v in flips] + [
            v["order2_reasoning"] for v in flips
        ]
        answer_a_openers = [r for r in flip_reasons if _opens_with_answer_a(r)]
        stats[suite] = {
            "n_pairs": len(vs),
            "n_flips": len(flips),
            "flip_rate": round(len(flips) / len(vs), 4) if vs else 0.0,
            "first_answer_bias_n": len(first_answer),
            "first_answer_bias_rate": round(len(first_answer) / len(flips), 4) if flips else 0.0,
            "answer_a_opener_n": len(answer_a_openers),
            "answer_a_opener_total": len(flip_reasons),
            "answer_a_opener_rate": round(len(answer_a_openers) / len(flip_reasons), 4)
            if flip_reasons
            else 0.0,
        }
    return stats


def _self_preference(verdicts: list[dict], judged_model: str) -> dict[str, dict]:
    """Where the free judge ranks its OWN model in each suite's standings - an
    in-pool judge inflating itself (reuses the tested `standings` reducer)."""
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        by_suite[v["suite"]].append(v)
    out: dict[str, dict] = {}
    for suite, vs in sorted(by_suite.items()):
        models = sorted({v["model_a"] for v in vs} | {v["model_b"] for v in vs})
        table = standings(models, vs)
        for rank, s in enumerate(table, start=1):
            if s.model == judged_model:
                out[suite] = {
                    "rank": rank,
                    "of": len(table),
                    "win_rate": round(s.win_rate, 4),
                }
                break
    return out


def build_stats(pairwise_dir: Path) -> dict:
    free = iter_verdicts(pairwise_dir / FREE_JUDGE_ARM)
    paid = iter_verdicts(pairwise_dir / PAID_JUDGE_ARM)
    return {
        "arms": {
            "free-local": {"model": FREE_JUDGE_ARM, "suites": _arm_stats(free)},
            "paid-gpt": {"model": PAID_JUDGE_ARM, "suites": _arm_stats(paid)},
        },
        "self_preference_free_judge": _self_preference(free, FREE_JUDGE_ARM),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m llm_benchmark.freeze_evidence")
    p.add_argument("--pairwise-dir", default=str(DEFAULT_PAIRWISE_DIR))
    p.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    args = p.parse_args(argv)

    pairwise_dir = Path(args.pairwise_dir)
    evidence_dir = Path(args.evidence_dir)

    counts = freeze_verdicts(pairwise_dir, evidence_dir)
    logger.info("froze pairwise verdicts: %s", counts)

    receipts = position_bias_receipts(pairwise_dir, evidence_dir, FREE_JUDGE_ARM)
    receipts_path = evidence_dir / "judge-position-bias-receipts.json"
    receipts_path.write_text(json.dumps(receipts, indent=2))
    logger.info("wrote %d flip receipts -> %s", len(receipts), receipts_path)

    stats = build_stats(pairwise_dir)
    stats_path = evidence_dir / "judge-position-bias-stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("wrote position-bias stats -> %s", stats_path)
    logger.info(
        "free-local flips: easy %s, adversarial %s",
        stats["arms"]["free-local"]["suites"].get("easy", {}).get("n_flips"),
        stats["arms"]["free-local"]["suites"].get("adversarial", {}).get("n_flips"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
