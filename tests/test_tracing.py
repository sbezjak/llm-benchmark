"""The optional Langfuse layer must be a hard no-op unless configured.

These are the guards that keep a mocked/free/CI run offline: no keys => export
does nothing and returns 0; keys but the extra not installed => it degrades to a
warning, never an error. Both paths run with no network and no real client, so
they belong in the default (mocked) suite. The enabled export path is proven by
the live smoke against a real Langfuse project, not mocked here - mocking the
whole OTEL client would test the mock, not the integration.
"""

import sys

import pytest

from llm_benchmark import tracing

pytestmark = pytest.mark.mocked

# .env is loaded by conftest, so the Langfuse keys may be in the env - drop them
# explicitly per test to control which branch runs.
_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def _one_record() -> dict:
    return {
        "model": "m",
        "item_id": "i1",
        "rep": 1,
        "prompt": "2+2?",
        "expected": "4",
        "cost_usd": 0.0,
        "captured_at": "2026-08-12T00:00:00+00:00",
        "measurement": {
            "text": "4",
            "tokens_in": 3,
            "tokens_out": 1,
            "latency_ms": 5,
            "ttft_ms": None,
        },
        "judge": {"score": 1.0, "passed": True, "reason": "ok"},
    }


def test_disabled_without_keys_is_a_noop(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    assert tracing.enabled() is False
    # returns 0 and never reaches the langfuse import, so a base install is safe
    assert tracing.export_records([_one_record()]) == 0


def test_keys_but_package_missing_warns_and_noops(monkeypatch):
    for k in _KEYS:
        monkeypatch.setenv(k, "test-key")
    # simulate the base install (extra not installed): make the import fail
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert tracing.enabled() is True
    assert tracing.export_records([_one_record()]) == 0


# --------------------------------------------------------------------------- #
# Live instrumentation seam. The runner depends on a `Tracer`; its default
# is `NullTracer`, which must be a hard no-op that opens no client and no socket -
# the whole "mocked/free/CI stays offline by construction" story. The live
# `LangfuseTracer` path is proven by the live smoke against the real project, not
# mocked here (mocking the OTEL client would test the mock, not the integration).
# --------------------------------------------------------------------------- #


def test_null_tracer_is_a_hard_noop():
    """Every method of the no-op seam runs with no client, no network, no error -
    including `.update(...)` on the yielded spans and `.score(...)`/`.flush()`."""
    tracer = tracing.NullTracer()
    with tracer.capture("m", "i1", 1) as cap:
        with cap.generation("answer-generation", "2+2?", model="m") as gen:
            gen.update(
                output="4", usage_details={"input": 3, "output": 1}, cost_details={"total": 0}
            )
        with cap.generation("judge-evaluation", "4") as jgen:
            jgen.update(output="ok", model="llama3.2", usage_details={"input": 5, "output": 2})
        cap.score(1.0, "ok")
        cap.update(input="2+2?", output="4")  # root headline in/out
    tracer.flush()  # no buffer, no client - must not raise


def test_build_tracer_returns_null_without_keys(monkeypatch):
    """The job's one live-vs-noop decision: no keys => NullTracer, so a real run
    on a machine without Langfuse keys never imports the client (DI, not a flag)."""
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    assert isinstance(tracing.build_tracer(), tracing.NullTracer)


def test_build_tracer_keys_but_package_missing_returns_null(monkeypatch):
    """Keys set but the extra absent degrades to the no-op with a warning, not a
    crash - the base install can still run a live sweep, just untraced."""
    for k in _KEYS:
        monkeypatch.setenv(k, "test-key")
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert isinstance(tracing.build_tracer(), tracing.NullTracer)
