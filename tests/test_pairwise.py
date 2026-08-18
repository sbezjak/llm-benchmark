"""Pairwise / Elo lens: mocked ($0). The load-bearing assertion is that the
both-orders swap CATCHES position bias - a judge that always picks whichever
answer is shown first produces a `flip`, never a spurious win. A content-based
judge that prefers the same answer regardless of slot produces an order-
independent `win`. No HTTP, no model: the judge is a stub over cached answer text.
"""

from __future__ import annotations

import json

import pytest

import llm_benchmark.pairwise as pairwise_cli
from llm_benchmark.providers.base import Measurement, Provider
from llm_benchmark.runners.pairwise import (
    TIE,
    classify,
    elo_from_win_rate,
    flip_rate,
    kendall_tau,
    parse_winner,
    run_pairwise,
    standings,
)


def _first_pick_judge():
    """A maximally position-biased judge: always picks ANSWER A (whatever is
    shown first). Every pair must therefore FLIP under the order swap."""

    async def _judge(prompt: str) -> str:
        return json.dumps({"reasoning": "first looks better", "winner": "A"})

    return _judge


def _content_judge(win_token: str):
    """A content-based judge: picks whichever slot holds `win_token`, regardless
    of position - so its verdict is order-independent (a real `win`)."""

    async def _judge(prompt: str) -> str:
        a_part, _, _b_part = prompt.partition("ANSWER B:")
        winner = "A" if win_token in a_part else "B"
        return json.dumps({"reasoning": "content", "winner": winner})

    return _judge


def _counting_judge():
    calls = {"n": 0}

    async def _judge(prompt: str) -> str:
        calls["n"] += 1
        return json.dumps({"reasoning": "x", "winner": "A"})

    return _judge, calls


_ITEMS = [{"item_id": "q1", "prompt": "Explain X.", "expected": "the X answer"}]
_ANSWERS = {
    ("m1", "q1"): {"text": "ALPHA answer from m1"},
    ("m2", "q1"): {"text": "BETA answer from m2"},
}
_MODELS = ["m1", "m2"]


@pytest.mark.mocked
async def test_position_bias_is_caught_as_a_flip(tmp_path):
    judge = _first_pick_judge()
    comps = await run_pairwise(_ANSWERS, _ITEMS, _MODELS, "easy", judge, pairwise_dir=tmp_path)

    assert len(comps) == 1
    assert comps[0]["outcome"] == "flip"  # both-orders swap unmasked the position vote
    assert comps[0]["winner"] is None
    assert flip_rate(comps) == 1.0  # the direct position-bias measurement
    # A flip is a non-decision: half credit each, nobody separated.
    rows = {s.model: s for s in standings(_MODELS, comps)}
    assert rows["m1"].flips == 1 and rows["m2"].flips == 1
    assert rows["m1"].win_rate == 0.5 and rows["m2"].win_rate == 0.5


@pytest.mark.mocked
async def test_verdict_records_the_judges_reasoning_both_orders(tmp_path):
    """The judge's own reasoning is persisted per order in the verdict (not dropped
    to a transient log). A flip's two reasonings are the self-contradiction we cite."""

    async def reason_judge(prompt: str) -> str:
        # names the slot it favours so the two orders carry different reasoning text
        return json.dumps({"reasoning": "Answer A is better, it is in slot A", "winner": "A"})

    comps = await run_pairwise(
        _ANSWERS, _ITEMS, _MODELS, "easy", reason_judge, pairwise_dir=tmp_path
    )
    c = comps[0]
    assert c["order1_reasoning"] == "Answer A is better, it is in slot A"
    assert c["order2_reasoning"] == "Answer A is better, it is in slot A"
    # and it is on disk, not just in memory
    written = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "order1_reasoning" in written and "order2_reasoning" in written


@pytest.mark.mocked
async def test_content_preference_is_an_order_independent_win(tmp_path):
    judge = _content_judge("ALPHA")  # always prefers m1's answer text
    comps = await run_pairwise(_ANSWERS, _ITEMS, _MODELS, "easy", judge, pairwise_dir=tmp_path)

    assert comps[0]["outcome"] == "win"
    assert comps[0]["winner"] == "m1"
    assert flip_rate(comps) == 0.0
    rows = {s.model: s for s in standings(_MODELS, comps)}
    assert rows["m1"].wins == 1 and rows["m1"].win_rate == 1.0
    assert rows["m2"].losses == 1 and rows["m2"].win_rate == 0.0


@pytest.mark.mocked
async def test_pairwise_is_idempotent(tmp_path):
    judge, calls = _counting_judge()
    await run_pairwise(_ANSWERS, _ITEMS, _MODELS, "easy", judge, pairwise_dir=tmp_path)
    assert calls["n"] == 2  # one pair, judged in both orders

    # Second pass: the cached comparison is loaded, the judge is NOT called again.
    judge2, calls2 = _counting_judge()
    comps = await run_pairwise(_ANSWERS, _ITEMS, _MODELS, "easy", judge2, pairwise_dir=tmp_path)
    assert calls2["n"] == 0  # spend/time control: cache hit, no re-judge
    assert comps[0]["outcome"] == "flip"


@pytest.mark.mocked
def test_classify_outcomes():
    assert classify("m1", "m1") == ("win", "m1")
    assert classify(TIE, TIE) == ("genuine_tie", None)
    assert classify("m1", "m2") == ("flip", None)
    assert classify("m1", TIE) == ("flip", None)  # one order decided, one tied -> non-decision


@pytest.mark.mocked
def test_parse_winner_tolerates_shapes_and_defaults_to_tie():
    assert parse_winner('{"winner": "A"}') == "A"
    assert parse_winner('noise {"reasoning":"x","winner":"B"} trailing') == "B"
    assert parse_winner("the winner = TIE here") == "TIE"
    assert parse_winner("unparseable garbage") == TIE  # never a silent win for one side


@pytest.mark.mocked
def test_kendall_tau_agreement():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    # One adjacent swap out of three pairs -> (2-1)/3.
    assert kendall_tau(["a", "b", "c"], ["b", "a", "c"]) == pytest.approx(1 / 3)


@pytest.mark.mocked
def test_elo_transform_preserves_order():
    assert elo_from_win_rate(0.5) == pytest.approx(1000.0)
    assert elo_from_win_rate(0.75) > elo_from_win_rate(0.5) > elo_from_win_rate(0.25)


class _FakeProvider(Provider):
    """Returns a fixed pairwise verdict, priced as gpt so cost_usd resolves."""

    def __init__(self) -> None:
        self.calls = 0

    async def measure(self, prompt: str) -> Measurement:
        self.calls += 1
        return Measurement(
            text='{"reasoning": "a", "winner": "A"}',
            tokens_in=200,
            tokens_out=15,
            latency_ms=150.0,
            ttft_ms=None,
            model="gpt-5.6-luna",
        )


@pytest.mark.mocked
def test_pairwise_cli_end_to_end(tmp_path, monkeypatch):
    """The job glue: fake judge + fake cache + stubbed suite map, no HTTP, under a
    tmp cwd. One item per suite, two models -> one pair x two orders x two suites."""
    monkeypatch.chdir(tmp_path)  # run_pairwise writes cache/pairwise/<judge> here
    provider = _FakeProvider()
    monkeypatch.setattr(
        pairwise_cli, "build_default_providers", lambda **k: {"gpt-5.6-luna": provider}
    )
    monkeypatch.setattr(
        pairwise_cli,
        "load_cached_records",
        lambda **k: [
            {
                "model": "m1",
                "item_id": "q_easy",
                "rep": 1,
                "prompt": "p",
                "expected": "e",
                "measurement": {"text": "a1"},
            },
            {
                "model": "m2",
                "item_id": "q_easy",
                "rep": 1,
                "prompt": "p",
                "expected": "e",
                "measurement": {"text": "a2"},
            },
            {
                "model": "m1",
                "item_id": "q_adv",
                "rep": 1,
                "prompt": "p",
                "expected": "e",
                "measurement": {"text": "a1"},
            },
            {
                "model": "m2",
                "item_id": "q_adv",
                "rep": 1,
                "prompt": "p",
                "expected": "e",
                "measurement": {"text": "a2"},
            },
        ],
    )
    monkeypatch.setattr(
        pairwise_cli, "suite_map", lambda **k: {"q_easy": "easy", "q_adv": "adversarial"}
    )

    rc = pairwise_cli.main(["--judge", "gpt-5.6-luna"])

    assert rc == 0
    assert provider.calls == 4  # (1 pair x 2 orders) x 2 suites
    assert (tmp_path / "cache" / "pairwise" / "gpt-5.6-luna").exists()
