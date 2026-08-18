from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of a single scorer applied to a single (output, expected) pair.

    `score` is in [0.0, 1.0]. `passed` is the scorer's binary verdict — for
    threshold-based scorers (semantic, judge), it's `score >= threshold`.
    `reason` is a short human-readable explanation, surfaced in reports.

    The `judge_*` fields are the grader's OWN resource use, populated only by
    the LLM-judge (an LLM call has a model and token counts); I/O-free scorers
    leave them None. They let a trace/dashboard show the evaluator's cost the
    same way it shows the answer's, so eval spend is visible, not hidden.
    """

    passed: bool
    score: float
    reason: str
    judge_model: str | None = None
    judge_tokens_in: int | None = None
    judge_tokens_out: int | None = None


class Scorer(ABC):
    """Scoring function over (question, output, expected).

    `score` is async because some scorers (LLM-as-judge) make HTTP calls at
    score time. Scorers without I/O (exact match, semantic similarity) just
    don't await anything. The harness runs under `asyncio_mode = "auto"`,
    so every test is async by default — async-only is the consistent
    choice given a single runtime context.

    Production frameworks (DeepEval, Ragas) expose dual sync+async
    interfaces to avoid async contagion across many runtime contexts.
    """

    name: str

    @abstractmethod
    async def score(self, question: str, output: str, expected: str) -> ScoreResult: ...
