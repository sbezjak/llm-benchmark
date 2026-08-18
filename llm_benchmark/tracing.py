"""Optional Langfuse tracing for the sweep - an ADDITIVE dashboard layer.

OFF by default and a hard no-op unless BOTH Langfuse keys are in the env AND the
`langfuse` package is installed (the `tracing` optional extra). The disk artifact
under `cache/` stays the ground truth (architecture intent); this only MIRRORS
finished records to a Langfuse dashboard for cross-run comparison. It reads the
cached records, never re-calls a provider - so it spends nothing and adds no
latency to a measurement, and a report-only run can populate the dashboard from
captures the models were already paid for.

This is the one deliberate exception to "HTTP calls only inside providers/": a
trace export is job observability, not a model call. It is called only from the
sweep JOB (`llm_benchmark.sweep`), never from the pytest capture path, so a
mocked/free/CI run never exports (no keys, or the extra not installed => no-op).

The `langfuse` import is lazy (inside the function) so the base install - which
does not carry the extra - imports this module and runs the disabled path fine.
The v4 SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
from the env; `get_client()` picks them up.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Live instrumentation seam.
#
# Backfill (`export_records` below) replays finished records, so every Langfuse
# span is created and ended in the same instant and its latency reads 0. The
# idiomatic fix is to time the REAL call: wrap `provider.measure` in a span whose
# with-block duration IS the latency, and nest the judge under the same root so
# the trace shows the tree (answer generation -> judge evaluation).
#
# To keep P5's seams intact this is injected at the RUNNER, never inside
# `providers/`. The runner takes a `Tracer` that DEFAULTS to `NullTracer` - a
# hard no-op that imports no langfuse and touches no network - so mocked/free/CI
# runs stay offline BY CONSTRUCTION. The sweep JOB builds a real `LangfuseTracer`
# only for a live run with keys present (mirrors `build_default_providers` living
# outside the core). Live and backfill are complementary: a real run traces live;
# `--report-only` cache replay keeps using `export_records`.
# --------------------------------------------------------------------------- #


class Tracer(Protocol):
    """The seam the runner depends on. `NullTracer` and `LangfuseTracer` both
    satisfy it; the runner never learns which one it got (DI, not env flags).

    A `capture(...)` context manager yields an object exposing `generation(name,
    input, model=None)` (a context manager yielding a span with `.update(output=,
    model=, usage_details=, cost_details=, completion_start_time=, metadata=)` -
    used for both the answer call and the judge call), a `score(value, comment)`,
    and an `update(**kwargs)` that sets the capture root's headline in/out +
    metadata. The concrete classes below define that shape; the runner drives it
    structurally."""

    def capture(self, model: str, item_id: str, rep: int) -> Any: ...
    def flush(self) -> None: ...


# --- no-op implementation (the default; the whole offline-by-construction story) ---


class _NullSpan:
    def update(self, **kwargs: Any) -> None:
        pass


class _NullCapture:
    @contextmanager
    def generation(self, name: str, input: str, model: str | None = None) -> Iterator[_NullSpan]:
        yield _NullSpan()

    def score(self, value: float, comment: str) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class NullTracer:
    """Does nothing, imports nothing, and never opens a socket. The runner's
    default so a test or a keyless run is offline without any branching."""

    @contextmanager
    def capture(self, model: str, item_id: str, rep: int) -> Iterator[_NullCapture]:
        yield _NullCapture()

    def flush(self) -> None:
        pass


# --- live implementation (built by the job only, never by pytest) ---


class _LangfuseCapture:
    """Wraps a Langfuse root span for one capture. Children opened here nest
    under it automatically (start_as_current_observation sets the current
    context); each child's with-block duration is its real, measured latency."""

    def __init__(self, client: Any, root: Any) -> None:
        self._client = client
        self._root = root

    @contextmanager
    def generation(self, name: str, input: str, model: str | None = None) -> Iterator[Any]:
        # Both the answer call and the judge call are generations - that is the
        # observation type Langfuse persists model/usage/cost on (span-like types,
        # incl. "evaluator", have no such columns and silently drop them). The
        # with-block times the call, so each generation's latency is real.
        with self._root.start_as_current_observation(
            name=name, as_type="generation", input=input, model=model
        ) as gen:
            yield gen  # native .update(output, usage_details, cost_details) after the call

    def update(self, **kwargs: Any) -> None:
        # Set fields on the root span - notably input (the question) and output
        # (the final answer), so the trace's headline in/out is not blank. In v4
        # the trace-level input/output is derived from this root observation.
        self._root.update(**kwargs)

    def score(self, value: float, comment: str) -> None:
        # The judge verdict is the quality axis - attach it to the trace so the
        # dashboard ranks models the same way the report does (same as backfill).
        self._client.create_score(
            trace_id=self._root.trace_id,
            name="judge_score",
            value=value,
            data_type="NUMERIC",
            comment=comment,
        )


class LangfuseTracer:
    """Live tracer built by the sweep job only. Importing langfuse and calling
    `get_client()` happen here, off the pytest path, so the base install and the
    mocked suite never touch either."""

    def __init__(self) -> None:
        from langfuse import get_client

        self._client = get_client()

    @contextmanager
    def capture(self, model: str, item_id: str, rep: int) -> Iterator[_LangfuseCapture]:
        with self._client.start_as_current_observation(
            name=f"sweep/{model}/{item_id}/rep{rep}",
            as_type="span",
            metadata={"model": model, "item_id": item_id, "rep": rep},
        ) as root:
            yield _LangfuseCapture(self._client, root)

    def flush(self) -> None:
        """A job exits right after the sweep; flush or the buffered traces are
        lost (same reason the backfill path flushes)."""
        self._client.flush()


def build_tracer() -> Tracer:
    """The one place the job decides live-vs-noop. Returns a real tracer only
    when the keys are present AND the extra is installed; otherwise the no-op, so
    a missing extra degrades to a warning instead of crashing the run."""
    if not enabled():
        logger.info("langfuse: disabled (no LANGFUSE_* keys) - live tracing off (NullTracer)")
        return NullTracer()
    try:
        import langfuse  # noqa: F401
    except ImportError:
        logger.warning(
            "langfuse: keys set but the package is not installed - run "
            "`uv sync --extra tracing`; live tracing off (NullTracer)"
        )
        return NullTracer()
    logger.info("langfuse: live tracing ON (LangfuseTracer)")
    return LangfuseTracer()


def enabled() -> bool:
    """True only when both Langfuse keys are present. The single gate every
    caller checks; keeps the "off unless configured" rule in one place."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def export_records(records: list[dict]) -> int:
    """Mirror finished sweep records to Langfuse, one trace per capture. Returns
    the number exported - 0 when disabled (no keys) or when the `langfuse` extra
    is not installed, so the caller can log the outcome either way.

    Each record already holds everything a trace needs (prompt, answer, tokens,
    latency, cost, judge verdict); nothing is re-called. The paid measurement
    becomes a generation observation; the free judge verdict becomes a score."""
    if not enabled():
        logger.info("langfuse: disabled (no LANGFUSE_* keys) - skipping trace export")
        return 0
    try:
        from langfuse import get_client
    except ImportError:
        logger.warning(
            "langfuse: keys set but the package is not installed - run "
            "`uv sync --extra tracing`; skipping trace export"
        )
        return 0

    client = get_client()
    exported = 0
    for r in records:
        m = r["measurement"]
        judge = r["judge"]
        gen = client.start_observation(
            name=f"sweep/{r['model']}/{r['item_id']}/rep{r['rep']}",
            as_type="generation",
            input=r["prompt"],
            output=m["text"],
            model=r["model"],
            usage_details={"input": m["tokens_in"], "output": m["tokens_out"]},
            cost_details={"total": r["cost_usd"]},
            metadata={
                "item_id": r["item_id"],
                "rep": r["rep"],
                "expected": r["expected"],
                "latency_ms": m["latency_ms"],
                "ttft_ms": m["ttft_ms"],
                "judge_passed": judge["passed"],
                "captured_at": r.get("captured_at"),
            },
        )
        gen.end()
        # The judge verdict is the quality axis - attach it as a trace score so
        # the dashboard can rank models the same way the report does.
        client.create_score(
            trace_id=gen.trace_id,
            name="judge_score",
            value=judge["score"],
            data_type="NUMERIC",
            comment=judge["reason"],
        )
        exported += 1

    client.flush()  # a job exits right after; flush or the buffered traces are lost
    logger.info("langfuse: exported %d traces", exported)
    return exported
