"""Drive the golden set across every model, capture once, math off the cache.

The sweep is the benchmark's engine: it runs each `(item, model, rep)`
through the same `Provider.measure -> Measurement` seam the provider adapters proved, scores
each answer with the free local LLM-judge, and writes one self-describing
record per capture to `cache/`.

The cache is the spend control, not a nicety. Before any paid call the runner
checks for an existing capture on disk; a hit is returned WITHOUT calling the
provider. So a re-run, a resume after a mid-sweep failure, or later re-analysis
never re-spends - this is the architecture-intent "every (model, item) call is
captured to disk once; all cost/latency/quality math runs off the cached
artifact, never a re-call" made literal.

The paid `measure` is the immutable, expensive part of a record. The judge
verdict is a re-runnable snapshot: it reads the cached answer text and runs on
free local Ollama, so it can be recomputed off the same artifact without
spending anything. It is stored inline for convenience, not because it is
ground truth the way the measurement is.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_benchmark.dataset import GoldenItem
from llm_benchmark.pricing import cost_components_usd, cost_usd
from llm_benchmark.providers.anthropic import AnthropicProvider
from llm_benchmark.providers.base import Measurement, Provider
from llm_benchmark.providers.ollama import OllamaProvider
from llm_benchmark.providers.openai_compat import OpenAICompatProvider
from llm_benchmark.scorers.base import Scorer
from llm_benchmark.scorers.judge import LLMJudgeScorer
from llm_benchmark.tracing import NullTracer, Tracer

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("cache")
# Module-level singleton so the runner's default is a hard no-op WITHOUT a
# function call in the argument default (NullTracer is stateless - one is enough).
_NULL_TRACER = NullTracer()


def capture_path(cache_dir: Path, model: str, item_id: str, rep: int) -> Path:
    """On-disk location of one capture. Named by durable topic vocabulary
    (model, item id, rep) - distinct from the smoke `__smoke-arithmetic.json`
    files, so a sweep never collides with a smoke."""
    return cache_dir / f"{model}__{item_id}__rep{rep}.json"


async def run_capture(
    item: GoldenItem,
    model: str,
    rep: int,
    provider: Provider,
    scorer: Scorer,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    tracer: Tracer = _NULL_TRACER,
) -> dict:
    """One `(item, model, rep)` capture. Idempotent: an existing capture is
    loaded and returned WITHOUT a provider call unless `force=True`. That skip
    is the spend control - the paid call only fires on a cache miss.

    `tracer` defaults to a no-op; the sweep job injects a real `LangfuseTracer`
    for a live run. It is opened only on a cache MISS - a hit does no call and no
    scoring, so there is nothing whose duration is worth timing. The tracer hook
    lives here, at the runner, never inside `providers/` (seam intent)."""
    path = capture_path(cache_dir, model, item.id, rep)
    if path.exists() and not force:
        logger.info(
            "sweep.cache HIT model=%s item=%s rep=%d (no call) -> %s", model, item.id, rep, path
        )
        return json.loads(path.read_text())

    logger.info(
        "sweep.capture model=%s item=%s rep=%d (miss - calling provider)", model, item.id, rep
    )
    with tracer.capture(model, item.id, rep) as cap:
        # The generation's with-block duration IS the real measured latency - the
        # exact number backfill loses (a replayed span has no duration). Every
        # number the provider reported is sent in its canonical Langfuse field so
        # the dashboard's own cost/latency/token math has the full inputs:
        #   - usage_details -> the token counts (Langfuse adds the total)
        #   - cost_details  -> input/output/total split (not just the aggregate)
        #   - completion_start_time -> lets the UI derive time-to-first-token
        # Exact latency_ms/ttft_ms also go in metadata as the source of record.
        with cap.generation("answer-generation", item.question, model=model) as gen:
            started = dt.datetime.now(dt.UTC)
            m = await provider.measure(item.question)
            cost = cost_usd(m)
            in_cost, out_cost = cost_components_usd(m)
            gen_fields: dict = {
                "output": m.text,
                "usage_details": {"input": m.tokens_in, "output": m.tokens_out},
                "cost_details": {"input": in_cost, "output": out_cost, "total": cost},
                "metadata": {"latency_ms": m.latency_ms, "ttft_ms": m.ttft_ms},
            }
            if m.ttft_ms is not None:
                gen_fields["completion_start_time"] = started + dt.timedelta(milliseconds=m.ttft_ms)
            gen.update(**gen_fields)
        # The judge is itself an LLM call, so it is a generation too (the type that
        # carries model/usage/cost - an "evaluator" span would drop them). It nests
        # under the same root, so the trace shows the tree (answer -> judge); its
        # input is the answer under test, its output the verdict, and its with-block
        # times the judge call. The score is attached to the trace below.
        with cap.generation("judge-evaluation", m.text) as jgen:
            verdict = await scorer.score(item.question, m.text, item.expected)
            jgen_fields: dict = {
                "output": {
                    "score": verdict.score,
                    "passed": verdict.passed,
                    "reason": verdict.reason,
                }
            }
            # If the judge reported its own usage, put its model + tokens + cost on
            # the judge generation, so eval spend is visible like the answer's (free
            # local judge -> $0; a paid judge would show real cost, same price table).
            if verdict.judge_model is not None:
                jgen_fields["model"] = verdict.judge_model
                jgen_fields["usage_details"] = {
                    "input": verdict.judge_tokens_in,
                    "output": verdict.judge_tokens_out,
                }
                judge_m = Measurement(
                    text="",
                    tokens_in=verdict.judge_tokens_in,
                    tokens_out=verdict.judge_tokens_out,
                    latency_ms=0.0,
                    ttft_ms=None,
                    model=verdict.judge_model,
                )
                try:
                    j_in, j_out = cost_components_usd(judge_m)
                    jgen_fields["cost_details"] = {
                        "input": j_in,
                        "output": j_out,
                        "total": j_in + j_out,
                    }
                except KeyError:
                    pass  # judge model not in the price table - keep model+usage
            jgen.update(**jgen_fields)
        cap.score(verdict.score, verdict.reason)
        # The root span carries the capture's headline in/out (question -> answer)
        # and the eval reference + item facets in metadata (which propagates to the
        # trace, where difficulty/category are filter dimensions). Tags would be
        # the tidier home for the facets, but langfuse 4.14.4 has no clean public
        # trace-tag setter, so metadata (also filterable) is the right call here.
        cap.update(
            input=item.question,
            output=m.text,
            metadata={
                "expected": item.expected,
                "difficulty": item.difficulty,
                "category": item.category,
            },
        )

    # Persist the judge's own metadata alongside the verdict - the model that
    # graded, its tokens, and its cost (free local judge -> $0; a paid judge shows
    # real spend off the same price table). So the report can name WHO graded every
    # row, not just the score. Older captures predate these fields; the reader
    # falls back to the run's judge model for them.
    judge_cost = 0.0
    if verdict.judge_model is not None and verdict.judge_tokens_in is not None:
        jm_meas = Measurement(
            text="",
            tokens_in=verdict.judge_tokens_in,
            tokens_out=verdict.judge_tokens_out or 0,
            latency_ms=0.0,
            ttft_ms=None,
            model=verdict.judge_model,
        )
        try:
            j_in, j_out = cost_components_usd(jm_meas)
            judge_cost = j_in + j_out
        except KeyError:
            judge_cost = 0.0  # judge model not in the price table (free local)

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
            "judge_model": verdict.judge_model,
            "judge_tokens_in": verdict.judge_tokens_in,
            "judge_tokens_out": verdict.judge_tokens_out,
            "judge_cost_usd": judge_cost,
        },
    }
    cache_dir.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    logger.info(
        "sweep.wrote model=%s item=%s rep=%d cost_usd=%.6g score=%.3f -> %s",
        model,
        item.id,
        rep,
        cost,
        verdict.score,
        path,
    )
    return record


async def run_sweep(
    items: list[GoldenItem],
    providers: dict[str, Provider],
    scorer: Scorer,
    reps: int = 2,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    tracer: Tracer = _NULL_TRACER,
) -> list[dict]:
    """Run the whole grid: every item x every model x every rep, sequentially.

    Sequential on purpose - a first sweep is easier to monitor and read as a
    clean top-to-bottom log than a `gather` interleave, and the cost is time
    (mostly the free local judge), not money. Returns every record; math runs
    off these, off the cache, never off a re-call.

    `tracer` is passed straight through to each capture and defaults to a no-op;
    the job injects a real one for a live run."""
    records: list[dict] = []
    for item in items:
        for model, provider in providers.items():
            for rep in range(1, reps + 1):
                records.append(
                    await run_capture(item, model, rep, provider, scorer, cache_dir, force, tracer)
                )
    return records


def load_cached_records(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[dict]:
    """Read every sweep capture back off disk, newest analysis over old spend.

    The reporting and stats steps run off this, not off a fresh sweep - so a
    report can be regenerated for $0 after the models were paid for once. Skips
    the provider-smoke captures (their names carry `smoke`); those are adapter
    proofs, not part of the graded grid."""
    records: list[dict] = []
    for path in sorted(cache_dir.glob("*.json")):
        if "smoke" in path.name:
            continue
        records.append(json.loads(path.read_text()))
    return records


@dataclass(frozen=True)
class ModelSummary:
    """Per-model roll-up across the sweep - the three axes plus spend."""

    model: str
    n: int
    mean_cost_usd: float
    total_cost_usd: float
    mean_latency_ms: float
    mean_ttft_ms: float | None
    mean_tokens_out: float
    mean_score: float
    pass_rate: float


def summarize(records: list[dict]) -> list[ModelSummary]:
    """Aggregate raw captures into one row per model. Ordered by mean quality
    score descending so the "which model" ranking reads top-down."""
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    summaries: list[ModelSummary] = []
    for model, rs in by_model.items():
        n = len(rs)
        costs = [r["cost_usd"] for r in rs]
        lats = [r["measurement"]["latency_ms"] for r in rs]
        ttfts = [r["measurement"]["ttft_ms"] for r in rs if r["measurement"]["ttft_ms"] is not None]
        outs = [r["measurement"]["tokens_out"] for r in rs]
        scores = [r["judge"]["score"] for r in rs]
        passes = [1 for r in rs if r["judge"]["passed"]]
        summaries.append(
            ModelSummary(
                model=model,
                n=n,
                mean_cost_usd=sum(costs) / n,
                total_cost_usd=sum(costs),
                mean_latency_ms=sum(lats) / n,
                mean_ttft_ms=(sum(ttfts) / len(ttfts)) if ttfts else None,
                mean_tokens_out=sum(outs) / n,
                mean_score=sum(scores) / n,
                pass_rate=len(passes) / n,
            )
        )
    summaries.sort(key=lambda s: s.mean_score, reverse=True)
    return summaries


def format_table(summaries: list[ModelSummary]) -> str:
    """Fixed-width cost/latency/quality table for the log/report. The three
    axes side by side is the "which model for which use case" read."""
    header = (
        f"{'model':<28} {'n':>3} {'mean$/q':>10} {'total$':>9} "
        f"{'lat_ms':>9} {'ttft_ms':>9} {'out_tok':>8} {'score':>6} {'pass':>6}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        ttft = f"{s.mean_ttft_ms:.0f}" if s.mean_ttft_ms is not None else "n/a"
        lines.append(
            f"{s.model:<28} {s.n:>3} {s.mean_cost_usd:>10.6f} {s.total_cost_usd:>9.5f} "
            f"{s.mean_latency_ms:>9.0f} {ttft:>9} {s.mean_tokens_out:>8.1f} "
            f"{s.mean_score:>6.3f} {s.pass_rate:>6.0%}"
        )
    return "\n".join(lines)


def build_default_providers(
    include_billed: bool = True,
    paid_timeout: float = 120.0,
    ollama_timeout: float = 180.0,
) -> dict[str, Provider]:
    """Construct the real provider set from `.env` keys. Kept OUT of the sweep
    core so the core never reads env or touches HTTP (the mocked test injects
    fakes). A paid provider whose key is absent is simply omitted - the same
    clean-skip behavior the billed tests have.

    DeepSeek rides the OpenAI-compatible class with only a base_url + key swap
    (the seam proven mocked and billed in the smoke). Timeouts are
    generous: reasoning/thinking tokens (DeepSeek, Sonnet) and local cold-start
    (Ollama) make the short-QA default too tight for a hard item."""
    providers: dict[str, Provider] = {
        "llama3.2": OllamaProvider(model="llama3.2", timeout=ollama_timeout)
    }
    if not include_billed:
        return providers

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if openai_key:
        providers["gpt-5.6-luna"] = OpenAICompatProvider(
            model="gpt-5.6-luna", api_key=openai_key, timeout=paid_timeout
        )
    if deepseek_key:
        providers["deepseek-v4-pro"] = OpenAICompatProvider(
            model="deepseek-v4-pro",
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1",
            timeout=paid_timeout,
        )
    if anthropic_key:
        providers["claude-haiku-4-5-20251001"] = AnthropicProvider(
            model="claude-haiku-4-5-20251001", api_key=anthropic_key, timeout=paid_timeout
        )
        providers["claude-sonnet-5"] = AnthropicProvider(
            model="claude-sonnet-5", api_key=anthropic_key, timeout=paid_timeout
        )
    return providers


def default_scorer() -> Scorer:
    """The vendored LLM-judge on free local Ollama - so scoring the paid
    answers adds $0. Note the self-grading caveat: the judge and the Ollama
    answering model are the same weights, a known blind spot on that one row
    (see the judge module docstring)."""
    return LLMJudgeScorer()
