"""Judge-choice ablation: re-grade the cached answers with a different judge.

The sweep graded every answer with the FREE local llama judge. That judge is the
same weights as one of the answering models, and it is a small model grading
frontier ones - so the quality ranking it produced is itself suspect. This runner
re-grades the *same cached answer text* with a second judge and asks whether the
ranking survives the judge swap. It is the project thesis turned on the eval:
*is a cheap judge good enough, or do you need the expensive one to grade?*

The answering models are NEVER re-called - the judge reads the answer TEXT that is
already on disk. So the only spend is the judge's own calls. Verdicts land in a
sibling `cache/judged/` tree keyed by judge model, never overwriting the sweep's
original verdicts (those are the free-judge arm of the comparison). Re-running is
idempotent: an existing judged capture is loaded without a call, the same spend
control the sweep uses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.pricing import cost_usd
from llm_benchmark.providers.base import Measurement, Provider
from llm_benchmark.scorers.judge import DEFAULT_THRESHOLD, LLMJudgeScorer

logger = logging.getLogger(__name__)

DEFAULT_JUDGED_DIR = Path("cache") / "judged"


class CostCapturingJudge:
    """Adapt a `Provider` into the judge's `str -> str` callable while keeping
    the `Measurement` the plain `generate` view throws away - so each judge
    call's token cost is captured for the spend total, not estimated."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.total_cost_usd = 0.0
        self.calls = 0
        self.last: Measurement | None = None

    async def __call__(self, prompt: str) -> str:
        m = await self.provider.measure(prompt)
        self.last = m
        self.total_cost_usd += cost_usd(m)
        self.calls += 1
        return m.text


def rejudge_path(
    judged_dir: Path, judge_model: str, answer_model: str, item_id: str, rep: int
) -> Path:
    """One re-judged verdict's location - keyed by BOTH the judge and the
    answering model, so two judges' verdicts on the same answer never collide."""
    return judged_dir / f"{judge_model}__{answer_model}__{item_id}__rep{rep}.json"


async def rejudge_all(
    records: list[dict],
    judge_provider: Provider,
    judge_model: str,
    threshold: float = DEFAULT_THRESHOLD,
    judged_dir: Path = DEFAULT_JUDGED_DIR,
    force: bool = False,
) -> tuple[list[dict], float]:
    """Re-grade every cached answer with `judge_provider`. Returns the judged
    records and the NEW spend (cache hits add nothing). Idempotent per capture."""
    judged_dir.mkdir(parents=True, exist_ok=True)
    capture = CostCapturingJudge(judge_provider)
    scorer = LLMJudgeScorer(threshold=threshold, judge_fn=capture)

    out: list[dict] = []
    for rec in records:
        answer_model = rec["model"]
        path = rejudge_path(judged_dir, judge_model, answer_model, rec["item_id"], rec["rep"])
        if path.exists() and not force:
            logger.info(
                "rejudge.cache HIT judge=%s answer=%s item=%s rep=%d (no call)",
                judge_model,
                answer_model,
                rec["item_id"],
                rec["rep"],
            )
            out.append(json.loads(path.read_text()))
            continue

        verdict = await scorer.score(rec["prompt"], rec["measurement"]["text"], rec["expected"])
        judge_cost = cost_usd(capture.last) if capture.last is not None else 0.0
        judged = {
            "judge_model": judge_model,
            "answer_model": answer_model,
            "item_id": rec["item_id"],
            "rep": rec["rep"],
            "prompt": rec["prompt"],
            "expected": rec["expected"],
            "answer_text": rec["measurement"]["text"],
            "judge_cost_usd": judge_cost,
            "judge": {"score": verdict.score, "passed": verdict.passed, "reason": verdict.reason},
        }
        path.write_text(json.dumps(judged, indent=2))
        logger.info(
            "rejudge.wrote judge=%s answer=%s item=%s rep=%d score=%.3f judge_cost=$%.6f",
            judge_model,
            answer_model,
            rec["item_id"],
            rec["rep"],
            verdict.score,
            judge_cost,
        )
        out.append(judged)

    return out, capture.total_cost_usd


@dataclass(frozen=True)
class JudgeComparisonRow:
    """One answering model's quality under two graders, side by side - the shape
    of the cheap-vs-strong judge finding."""

    model: str
    n: int
    local_mean: float
    other_mean: float
    local_pass: int
    other_pass: int


def judge_comparison(
    records: list[dict], judged: list[dict]
) -> tuple[list[JudgeComparisonRow], list[dict]]:
    """Compare the sweep's own (local) verdicts against a re-judge pass. Returns
    per-model rows plus the list of PASS/FAIL disagreements - the cases where the
    two judges reach a different decision, which is where a cheap judge earns or
    loses its keep. Only captures graded by BOTH judges are compared."""
    local = {(r["model"], r["item_id"], r["rep"]): r["judge"] for r in records}
    other = {(j["answer_model"], j["item_id"], j["rep"]): j["judge"] for j in judged}
    shared = sorted(set(local) & set(other))

    by_model: dict[str, list[tuple]] = {}
    disagreements: list[dict] = []
    for key in shared:
        model = key[0]
        lj, oj = local[key], other[key]
        by_model.setdefault(model, []).append((lj, oj))
        if lj["passed"] != oj["passed"]:
            disagreements.append(
                {
                    "model": model,
                    "item_id": key[1],
                    "rep": key[2],
                    "local_score": lj["score"],
                    "local_passed": lj["passed"],
                    "other_score": oj["score"],
                    "other_passed": oj["passed"],
                }
            )

    rows: list[JudgeComparisonRow] = []
    for model, pairs in by_model.items():
        n = len(pairs)
        rows.append(
            JudgeComparisonRow(
                model=model,
                n=n,
                local_mean=sum(lj["score"] for lj, _ in pairs) / n,
                other_mean=sum(oj["score"] for _, oj in pairs) / n,
                local_pass=sum(1 for lj, _ in pairs if lj["passed"]),
                other_pass=sum(1 for _, oj in pairs if oj["passed"]),
            )
        )
    rows.sort(key=lambda r: r.other_mean, reverse=True)
    return rows, disagreements
