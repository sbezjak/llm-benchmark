"""Regenerate the judge-calibration blind grading sheet + answer key - the generator.

  python -m llm_benchmark.calibration

The judge is a model standing in for a human grader; calibration validates that
proxy by having a human grade a sample BLIND (Good / Weak / Wrong) and comparing to
the judge. The one production move in P5 that costs $0 - only reading time. The paid
gpt judge is the one calibrated, because it is the one the benchmark recommends.

This used to be a hand-built sheet with no generator, so the 2026-08-17 re-roll left
it stale (it quoted answers that no longer exist). This module is the generator:
pure and $0 (reads `cache/judged/`, the paid-judge verdicts already on disk), it
picks a stratified, deterministic 24-answer sample and writes two files:

  1. `docs/judge-calibration-grading-sheet.md` - the BLIND sheet: question, expected,
     answer, three empty band boxes. Carries NO judge score or band (that is the point).
  2. `evidence/judge-calibration-key.json` - the key: per-id judge score, band, and
     the judge's own reasoning, kept sealed until the human grading is done.

Sample design: include every non-perfect verdict (the answers with any grading signal
- the borderline and the wrong ones) plus a fixed number of perfect answers per model
so the human also sees clear-Good cases. Deterministic (sorted + fixed-seed shuffle)
so the sheet regenerates identically and the ids are stable. The judge has no "Weak"
band - it is pass/fail around 0.7 - so `judge_band` is Good (passed) or Wrong (failed)
by construction; the human's "Weak" is the middle the judge cannot express, and the
gap between the two is itself the finding.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger("llm_benchmark.calibration")

DEFAULT_JUDGED_DIR = Path("cache") / "judged"
DEFAULT_SHEET = Path("docs") / "judge-calibration-grading-sheet.md"
DEFAULT_KEY = Path("evidence") / "judge-calibration-key.json"

PERFECT_PER_MODEL = 2  # clear-Good cases per model, on top of every non-perfect answer
SEED = 5  # fixed so the sheet/key regenerate identically (project 5)


def judge_band(passed: bool) -> str:
    """The judge outputs pass/fail around 0.7 - no middle band. So its band is Good
    (passed) or Wrong (failed); the human supplies the Weak the judge cannot."""
    return "Good" if passed else "Wrong"


def load_judged(judged_dir: Path) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(judged_dir / "*.json")))]


def sample(
    records: list[dict], perfect_per_model: int = PERFECT_PER_MODEL, seed: int = SEED
) -> list[dict]:
    """Every non-perfect verdict + `perfect_per_model` perfect ones per model,
    deterministically. Then shuffle (fixed seed) so the band order does not leak
    down the sheet (all the Wrong answers must not cluster at the top)."""
    non_perfect = [r for r in records if r["judge"]["score"] < 1.0]
    perfect = [r for r in records if r["judge"]["score"] >= 1.0]

    by_model: dict[str, list[dict]] = {}
    for r in sorted(perfect, key=lambda r: (r["answer_model"], r["item_id"], r["rep"])):
        by_model.setdefault(r["answer_model"], []).append(r)
    perfect_pick = [r for rs in by_model.values() for r in rs[:perfect_per_model]]

    chosen = sorted(
        non_perfect + perfect_pick,
        key=lambda r: (r["judge"]["score"], r["answer_model"], r["item_id"], r["rep"]),
    )
    random.Random(seed).shuffle(chosen)
    return chosen


def _key_entry(r: dict) -> dict:
    j = r["judge"]
    return {
        "answer_model": r["answer_model"],
        "item_id": r["item_id"],
        "rep": r["rep"],
        "judge_score": j["score"],
        "judge_band": judge_band(j["passed"]),
        "judge_passed": j["passed"],
        "judge_reason": j["reason"],
    }


def build_key(chosen: list[dict]) -> dict[str, dict]:
    return {f"G{ix:02d}": _key_entry(r) for ix, r in enumerate(chosen, start=1)}


def build_sheet(chosen: list[dict]) -> str:
    n = len(chosen)
    n_below = sum(1 for r in chosen if r["judge"]["score"] < 1.0)
    lines = [
        "# Judge calibration - grade these yourself, blind",
        "",
        (
            "You are the ground truth. Grade each answer **without** knowing what the model "
            "judge (gpt-5.6-luna) gave it - that is the whole point. Afterwards we compare "
            "your grades to the judge's and report how often they agree."
        ),
        "",
        "**How to grade** - for each answer, mark one:",
        "",
        "- **Good** - correct and answers the question well",
        "- **Weak** - partly right, missing something, or flawed but not wrong",
        "- **Wrong** - incorrect, off-topic, or unusable",
        "",
        (
            "Put an `x` in one box. Add a short note if you want (optional). Do **not** "
            "open `evidence/judge-calibration-key.json` until you are done."
        ),
        "",
        (
            f"Sample: **{n} answers** ({n_below} the judge scored below perfect, the rest "
            "it scored perfect - you don't get told which, or in what order)."
        ),
        "",
        (
            "> These are the 2026-08-17 re-roll answers. Regenerate this sheet with "
            "`python -m llm_benchmark.calibration` (pure, $0)."
        ),
        "",
        "---",
        "",
    ]
    for ix, r in enumerate(chosen, start=1):
        gid = f"G{ix:02d}"
        lines += [
            f"### {gid} · item `{r['item_id']}`",
            "",
            f"**Question:** {r['prompt']}",
            "",
            f"**Expected:** {r['expected']}",
            "",
            "**Answer:**",
            "",
            "~~~~",
            r["answer_text"].rstrip(),
            "~~~~",
            "",
            "- [ ] Good   - [ ] Weak   - [ ] Wrong",
            "",
            "Note: ",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m llm_benchmark.calibration")
    p.add_argument("--judged-dir", default=str(DEFAULT_JUDGED_DIR))
    p.add_argument("--sheet", default=str(DEFAULT_SHEET))
    p.add_argument("--key", default=str(DEFAULT_KEY))
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the sheet even if it already carries human grades (`[x]`). "
        "Off by default so a regen never wipes in-progress grading.",
    )
    args = p.parse_args(argv)

    records = load_judged(Path(args.judged_dir))
    chosen = sample(records)
    logger.info(
        "sampled %d answers (%d below perfect, %d perfect) across %d models",
        len(chosen),
        sum(1 for r in chosen if r["judge"]["score"] < 1.0),
        sum(1 for r in chosen if r["judge"]["score"] >= 1.0),
        len({r["answer_model"] for r in chosen}),
    )

    sheet_path = Path(args.sheet)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    # A graded sheet is human work, not a regenerable artifact - never clobber it by
    # accident. The key regenerates identically (deterministic), so the pairing holds.
    if sheet_path.exists() and "[x]" in sheet_path.read_text() and not args.force:
        logger.warning(
            "sheet %s already has grades ([x]) - NOT overwriting (use --force to). "
            "Key still (re)written; it is deterministic so it stays aligned.",
            sheet_path,
        )
    else:
        sheet_path.write_text(build_sheet(chosen))
        logger.info("wrote BLIND grading sheet -> %s", sheet_path)

    key_path = Path(args.key)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(build_key(chosen), indent=2))
    logger.info("wrote sealed key -> %s (do not open until grading is done)", key_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
