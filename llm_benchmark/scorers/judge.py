"""LLM-as-judge: a second model grades the first model's answer.

Plain-English version of what this file does, end to end:

1. Take the question, the expected answer, and the model's actual
   answer. Drop them into a grading-instructions template (see
   `RUBRIC_PROMPT`).
2. Send that filled-in prompt to an LLM (a *separate* call from
   whatever produced the answer being graded).
3. The LLM replies with a small JSON blob containing two integer
   scores (correctness 0-10, relevance 0-10) and a one-line
   explanation.
4. Parse the JSON, average the two scores, divide by 10 to get a
   score in [0, 1], compare against a threshold, return PASS or FAIL.

Why bother. Text-shape scorers (exact match, BLEU/ROUGE, embedding
similarity — P1 has the family; P5 keeps only the judge) all compare
*shapes*, so they get fooled two ways: a right answer phrased very
differently from the expected (a bulleted list of all 8 planets when
the expected was just "8" — no string overlap), and a wrong answer
that reuses the right vocabulary (high similarity, still wrong). A
judge LLM reads both texts and can sometimes tell "yes, this answers
the question" from "no, this is plausible-sounding nonsense" where
pure-text comparison can't.

Important caveat. When the judge and the answering model are the same
weights (the case in this project — both are llama3.2), the judge has
a blind spot for its own hallucinations. It produced the same
nonsense; it can't tell it apart from a real answer. Surfacing this
"self-grading bias" is one of the things this harness is here to
demonstrate — see `tests/test_calibration.py` and the
`procedural_001` item in `data/human_labels.yaml`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable

from llm_benchmark.providers.base import Measurement
from llm_benchmark.providers.ollama import OllamaProvider
from llm_benchmark.scorers.base import Scorer, ScoreResult

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL = "llama3.2"

# The grading instructions sent to the judge. See LLMJudgeScorer's
# docstring for why the rubric looks the way it does.
RUBRIC_PROMPT = """You are a strict but fair grader for a question-answering system.

You will be given:
- A QUESTION
- An EXPECTED answer (a reference — the model's answer does NOT need to
  match it word-for-word to be correct)
- The MODEL'S answer to grade

Grade the model's answer on two dimensions, each 0-10:
- correctness: does the answer correctly answer the question?
  10 = fully correct (even if phrased differently from the expected answer)
   5 = partially correct, or correct but with significant inaccuracies
   0 = wrong, contradicts the expected answer, or hallucinated
- relevance: is the answer on-topic and responsive to the question?
  10 = directly addresses the question
   5 = partially on-topic, drifts
   0 = off-topic or refuses to answer

Respond with ONLY a JSON object, no other text:
{{"reasoning": "<one or two sentences explaining your scores>",
 "correctness": <int>,
 "relevance": <int>}}

QUESTION: {question}
EXPECTED: {expected}
MODEL'S ANSWER: {output}"""


class JudgeParseError(RuntimeError):
    """Raised when the judge's response cannot be parsed into scores.

    A malformed judge response is an infrastructure problem (the judge
    model isn't following the rubric format), not "the model's answer
    was bad." Silently coercing to a 0 score would hide bugs in the
    judge prompt or the parsing logic.
    """


def _parse_judge_response(raw: str) -> tuple[int, int, str]:
    """Extract (correctness, relevance, reasoning) from the judge's text.

    Strategy: try strict JSON first. If the model wrapped the JSON in
    prose or markdown, try to find the first {...} block. As a last
    resort, regex-extract the two integer scores. If even that fails,
    raise JudgeParseError with the raw text — better than silently
    returning a 0.
    """
    try:
        data = json.loads(raw)
        return (
            int(data["correctness"]),
            int(data["relevance"]),
            str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    json_blob = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_blob:
        try:
            data = json.loads(json_blob.group(0))
            return (
                int(data["correctness"]),
                int(data["relevance"]),
                str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    correctness_match = re.search(r'"?correctness"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    relevance_match = re.search(r'"?relevance"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    if correctness_match and relevance_match:
        return (
            int(correctness_match.group(1)),
            int(relevance_match.group(1)),
            "(reasoning unparseable, scores extracted by regex fallback)",
        )

    raise JudgeParseError(f"Could not parse judge response: {raw!r}")


class LLMJudgeScorer(Scorer):
    """Calls a grading LLM and turns its reply into a PASS/FAIL.

    Three rubric design choices are worth knowing about, since they
    explain why `RUBRIC_PROMPT` looks the way it does:

    - **Reference is a hint, not a string to match.** The judge is told
      the expected answer is a reference — the model's answer can be
      phrased completely differently and still be correct. This is what
      lets the judge pass right-but-different-shape answers (a bulleted
      list when the expected was a single number, etc).
    - **Two scores, not one.** Asking for *correctness* and *relevance*
      separately keeps two distinct failure modes from being smeared
      into a single number: a right-but-rambling answer scores high on
      correctness and low on relevance; a confident hallucination
      scores low on correctness and high on relevance. A single
      "quality" score loses that signal.
    - **Reasoning before score.** The rubric asks the judge to justify
      its score *before* writing the number down. A model asked for
      the score first tends to commit to a number and rationalize it.

    Construction.

    - `threshold` — pass/fail cutoff on the [0, 1] combined score.
    - `judge_measure_fn` — async callable prompt -> `Measurement`. The default,
      built from a fresh `OllamaProvider` at temperature 0 (judge wants
      determinism). Returning a full `Measurement` (not just text) is what lets
      the verdict carry the judge's OWN model + token counts, so a trace can
      show evaluator cost. When set, it is the path `score` uses.
    - `judge_fn` — the older text-only path: prompt -> raw text. Kept for
      callers that inject a plain string callable (e.g. the re-judge flow, and
      tests that avoid HTTP). It yields no token usage, so the `judge_*` fields
      stay None. Ignored when `judge_measure_fn` is provided.
    - `model` — the Ollama model name to use when the default judge is built.
    """

    name = "judge"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        judge_fn: Callable[[str], Awaitable[str]] | None = None,
        model: str = DEFAULT_MODEL,
        judge_measure_fn: Callable[[str], Awaitable[Measurement]] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.model = model
        # Default to the measure path (gives token usage); an injected judge_fn
        # keeps the text-only path for backward compatibility.
        if judge_fn is None and judge_measure_fn is None:
            provider = OllamaProvider(model=model, timeout=180.0, temperature=0.0)
            judge_measure_fn = provider.measure
        self._judge_fn = judge_fn
        self._judge_measure_fn = judge_measure_fn

    def build_prompt(self, question: str, output: str, expected: str) -> str:
        """Render the rubric prompt the judge will see. Public so tests and
        debug paths can inspect what was actually sent without scoring."""
        return RUBRIC_PROMPT.format(question=question, expected=expected, output=output)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        prompt = self.build_prompt(question, output, expected)
        # Prefer the measure path so the verdict carries the judge's own usage;
        # fall back to the text-only judge_fn (no usage) when that was injected.
        if self._judge_measure_fn is not None:
            jm = await self._judge_measure_fn(prompt)
            raw = jm.text
            judge_model: str | None = jm.model
            judge_tokens_in: int | None = jm.tokens_in
            judge_tokens_out: int | None = jm.tokens_out
        else:
            raw = await self._judge_fn(prompt)
            judge_model = judge_tokens_in = judge_tokens_out = None
        correctness, relevance, reasoning = _parse_judge_response(raw)

        # Clamp into 0-10 in case the judge invented an out-of-range score.
        correctness = max(0, min(10, correctness))
        relevance = max(0, min(10, relevance))

        score = (correctness + relevance) / 20.0
        passed = score >= self.threshold
        reason = (
            f"correctness={correctness}/10 relevance={relevance}/10 "
            f"score={score:.3f} {'>=' if passed else '<'} threshold={self.threshold} "
            f"| reasoning: {reasoning}"
        )
        # The provider call above already logged the prompt in + raw judge JSON
        # out. This logs the DERIVED verdict so the report shows PASS/FAIL and
        # the combined score without the reader doing (c+r)/20 arithmetic.
        logger.info("judge.score verdict=%s %s", "PASS" if passed else "FAIL", reason)
        return ScoreResult(
            passed=passed,
            score=score,
            reason=reason,
            judge_model=judge_model,
            judge_tokens_in=judge_tokens_in,
            judge_tokens_out=judge_tokens_out,
        )
