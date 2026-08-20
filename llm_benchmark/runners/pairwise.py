"""Pairwise ranking lens: rank the models by head-to-head preference.

The absolute analysis scores each answer on its own and ranks by the mean.
This is the other production pattern - the LMSYS-Arena / Elo shape: show a judge
two models' answers to the SAME item and ask which is better. Aggregate the
head-to-head wins into a ranking, then ask whether it agrees with the absolute
one. It answers "which model when" directly - A beats B - without needing a
calibrated absolute scale.

$0: the judge is the local Ollama baseline reading answer TEXT already on disk;
no model is re-called and no paid provider is touched. The cost is time (a
round-robin is N-choose-2 pairs x 2 orders per item), not money.

Position bias is the whole reason this is careful. LLM judges systematically
favor whichever answer is shown FIRST, regardless of content - so a naive "A vs
B" makes the ranking an artifact of presentation order. The fix (a production
requirement, not polish): judge every pair in BOTH orders - (A,B) and (B,A) - and
count a win only when it SURVIVES the swap. Three outcomes fall out, and keeping
them distinct is half the finding:

- **win**: both orders pick the same model - a real, order-independent preference.
- **genuine tie**: both orders say TIE - the judge saw them as equal.
- **flip**: the two orders disagree - the judge voted on POSITION, not quality.
  Counted as a non-decision (half credit) in the ranking, but the flip RATE is
  reported on its own as a direct measurement of how biased the judge was.

Ranking is by win rate (wins + half the non-decisions, over all games) - it is
order-independent and needs no iteration, unlike a sequential Elo whose numbers
depend on game order. An Elo-scale number is derived from the win rate for
familiarity (`elo_from_win_rate`), clearly a transform, not a separate fit.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PAIRWISE_DIR = Path("cache") / "pairwise"

# Symmetric on purpose: nothing in the wording hints which slot is favoured, so
# the only asymmetry left is position - which the both-orders swap cancels.
PAIRWISE_PROMPT = """You are comparing two answers to the same question and choosing which is better.

QUESTION: {question}
REFERENCE (a correct answer, for your judgment - the answers need not match it word-for-word): {expected}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which answer is better overall - more correct and more relevant? If they are equally good or equally bad, say TIE.
Respond with ONLY a JSON object, no other text:
{{"reasoning": "<one sentence>", "winner": "A" or "B" or "TIE"}}"""

TIE = "TIE"


def build_pairwise_prompt(question: str, expected: str, answer_a: str, answer_b: str) -> str:
    return PAIRWISE_PROMPT.format(
        question=question, expected=expected, answer_a=answer_a, answer_b=answer_b
    )


def parse_winner(raw: str) -> str:
    """Pull "A" / "B" / "TIE" out of the judge's reply. Strict JSON first, then a
    braces block, then a bare-word fallback; an unreadable reply is a TIE (a
    non-decision, never a silent win for one side)."""
    blob = re.search(r"\{.*\}", raw, re.DOTALL)
    for candidate in (raw, blob.group(0) if blob else None):
        if not candidate:
            continue
        try:
            w = str(json.loads(candidate)["winner"]).strip().upper()
            if w in ("A", "B", TIE):
                return w
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    m = re.search(r'"?winner"?\s*[:=]\s*"?(A|B|TIE)"?', raw, re.IGNORECASE)
    return m.group(1).upper() if m else TIE


def parse_reasoning(raw: str) -> str:
    """The judge's one-line reasoning from its reply (empty string if unreadable).
    Stored next to the verdict so the cache keeps the judge's OWN WORDS, not just
    the winner - the raw output is the ground truth for the position-bias finding,
    and dropping it (the earlier behaviour) sent the only copy to a transient log."""
    blob = re.search(r"\{.*\}", raw, re.DOTALL)
    for candidate in (raw, blob.group(0) if blob else None):
        if not candidate:
            continue
        try:
            r = json.loads(candidate).get("reasoning")
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            continue
        if r:
            return str(r).strip()
    return ""


def classify(order1: str, order2: str) -> tuple[str, str | None]:
    """Combine the two ordered verdicts into an outcome. `order1`/`order2` are the
    WINNING MODEL (or TIE) after mapping each call's A/B back to the real model.

    - same model both times -> ("win", model)
    - TIE both times        -> ("genuine_tie", None)
    - anything else         -> ("flip", None)   (position-dependent, a non-decision)
    """
    if order1 != TIE and order1 == order2:
        return ("win", order1)
    if order1 == TIE and order2 == TIE:
        return ("genuine_tie", None)
    return ("flip", None)


@dataclass(frozen=True)
class Standing:
    """One model's head-to-head record for a suite."""

    model: str
    games: int
    wins: int
    losses: int
    genuine_ties: int
    flips: int
    win_rate: float
    elo: float


def elo_from_win_rate(win_rate: float, anchor: float = 1000.0) -> float:
    """Map a win rate in (0,1) onto the familiar Elo scale relative to `anchor`.
    A pure transform of the win rate (400*logit), so it never re-orders the
    ranking - it just puts it in points people recognize. Clamped so a clean
    sweep or shutout stays finite."""
    wr = min(max(win_rate, 0.01), 0.99)
    return anchor + 400.0 * math.log10(wr / (1.0 - wr))


async def compare_pair(
    judge_fn: Callable[[str], Awaitable[str]],
    question: str,
    expected: str,
    model_a: str,
    answer_a: str,
    model_b: str,
    answer_b: str,
) -> dict:
    """Judge one pair in BOTH orders and classify. `judge_fn` is prompt -> raw
    text (the local Ollama judge in the job; a stub in tests)."""
    # order 1: A = model_a, B = model_b
    raw1 = await judge_fn(build_pairwise_prompt(question, expected, answer_a, answer_b))
    w1 = parse_winner(raw1)
    winner1 = model_a if w1 == "A" else model_b if w1 == "B" else TIE
    # order 2: A = model_b, B = model_a (swap)
    raw2 = await judge_fn(build_pairwise_prompt(question, expected, answer_b, answer_a))
    w2 = parse_winner(raw2)
    winner2 = model_b if w2 == "A" else model_a if w2 == "B" else TIE

    outcome, winner = classify(winner1, winner2)
    return {
        "model_a": model_a,
        "model_b": model_b,
        # The exact inputs the judge saw, so the verdict is a self-contained receipt
        # - the raw answers a claim rests on live IN the evidence file, not only in a
        # 44k-line run log you have to grep. (question/expected are per-item; the two
        # answers are per-model.)
        "question": question,
        "reference": expected,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "order1_winner": winner1,
        "order2_winner": winner2,
        # The judge's own reasoning for each order. A flip's two reasonings are the
        # self-contradiction.
        "order1_reasoning": parse_reasoning(raw1),
        "order2_reasoning": parse_reasoning(raw2),
        # The judge's FULL raw reply for each order, verbatim - never parse-and-drop
        # the ground truth. The reasoning fields above are extracted from these.
        "order1_raw_reply": raw1,
        "order2_raw_reply": raw2,
        "outcome": outcome,  # win | genuine_tie | flip
        "winner": winner,  # the model, or None
    }


def _pairwise_path(pairwise_dir: Path, suite: str, item_id: str, m1: str, m2: str) -> Path:
    a, b = sorted((m1, m2))  # canonical order so a pair maps to one file
    return pairwise_dir / f"{suite}__{item_id}__{a}__{b}.json"


async def run_pairwise(
    answers: dict[tuple[str, str], dict],
    items: list[dict],
    models: list[str],
    suite: str,
    judge_fn: Callable[[str], Awaitable[str]],
    pairwise_dir: Path = DEFAULT_PAIRWISE_DIR,
    force: bool = False,
) -> list[dict]:
    """Round-robin every model pair over every item for one suite. `answers` maps
    (model, item_id) -> answer record ({prompt, expected, text}); `items` is the
    ordered item list. Idempotent: a cached comparison is loaded, not re-judged
    (the same spend/time control the sweep uses)."""
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for item in items:
        iid = item["item_id"]
        for m1, m2 in combinations(models, 2):
            path = _pairwise_path(pairwise_dir, suite, iid, m1, m2)
            if path.exists() and not force:
                out.append(json.loads(path.read_text()))
                continue
            a = answers.get((m1, iid))
            b = answers.get((m2, iid))
            if a is None or b is None:
                logger.warning("missing answer for %s/%s on %s - skipping pair", m1, m2, iid)
                continue
            res = await compare_pair(
                judge_fn, item["prompt"], item["expected"], m1, a["text"], m2, b["text"]
            )
            res.update({"suite": suite, "item_id": iid})
            path.write_text(json.dumps(res, indent=2))
            logger.info(
                "pairwise %s %s: %s vs %s -> %s (%s)",
                suite,
                iid,
                m1,
                m2,
                res["winner"] or "-",
                res["outcome"],
            )
            out.append(res)
    return out


def standings(models: list[str], comparisons: list[dict]) -> list[Standing]:
    """Aggregate the head-to-head comparisons into a per-model record ranked by
    win rate. A non-decision (genuine tie or flip) is half credit to each side;
    the flip and genuine-tie counts are kept so the position-bias rate is
    visible, not smeared into the win rate."""
    rec = {m: {"games": 0, "wins": 0, "losses": 0, "genuine_ties": 0, "flips": 0} for m in models}
    for c in comparisons:
        a, b = c["model_a"], c["model_b"]
        rec[a]["games"] += 1
        rec[b]["games"] += 1
        if c["outcome"] == "win":
            w = c["winner"]
            loser = b if w == a else a
            rec[w]["wins"] += 1
            rec[loser]["losses"] += 1
        else:
            key = "genuine_ties" if c["outcome"] == "genuine_tie" else "flips"
            rec[a][key] += 1
            rec[b][key] += 1

    out: list[Standing] = []
    for m, r in rec.items():
        non_decisions = r["genuine_ties"] + r["flips"]
        wr = (r["wins"] + 0.5 * non_decisions) / r["games"] if r["games"] else 0.0
        out.append(
            Standing(
                model=m,
                games=r["games"],
                wins=r["wins"],
                losses=r["losses"],
                genuine_ties=r["genuine_ties"],
                flips=r["flips"],
                win_rate=wr,
                elo=elo_from_win_rate(wr),
            )
        )
    out.sort(key=lambda s: s.win_rate, reverse=True)
    return out


def flip_rate(comparisons: list[dict]) -> float:
    """Fraction of comparisons whose verdict flipped with presentation order -
    the direct measurement of the judge's position bias."""
    if not comparisons:
        return 0.0
    return sum(1 for c in comparisons if c["outcome"] == "flip") / len(comparisons)


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Rank-agreement between two orderings of the same models, in [-1, 1]
    (1 = identical order, -1 = reversed). Used to ask whether the pairwise
    ranking agrees with the absolute one."""
    common = [m for m in order_a if m in order_b]
    rank_b = {m: i for i, m in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s = rank_b[common[i]] - rank_b[common[j]]
            if s < 0:
                concordant += 1
            elif s > 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0
