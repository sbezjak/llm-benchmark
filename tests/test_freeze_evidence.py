"""Freeze-evidence generator: pure ($0, no model, no HTTP). The load-bearing
assertions are (1) the freeze mirror-copies every cached verdict so the committed
evidence can never silently drift from the cache again, and (2) the position-bias
stats it emits are computed correctly off the verdicts - a flip where slot A wins
both times is first-answer bias, and a reasoning that opens "Answer A" is counted.
This is the regression guard for the manual-copy drift that the 2026-08-17 re-roll
exposed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_benchmark.freeze_evidence import (
    FREE_JUDGE_ARM,
    PAID_JUDGE_ARM,
    build_stats,
    freeze_verdicts,
    position_bias_receipts,
)


def _verdict(suite, item, a, b, o1_winner, o2_winner, o1_reason, o2_reason, outcome):
    return {
        "model_a": a,
        "model_b": b,
        "order1_winner": o1_winner,
        "order2_winner": o2_winner,
        "order1_reasoning": o1_reason,
        "order2_reasoning": o2_reason,
        "outcome": outcome,
        "winner": None if outcome == "flip" else o1_winner,
        "suite": suite,
        "item_id": item,
    }


@pytest.fixture
def fake_pairwise(tmp_path: Path) -> Path:
    """A minimal two-arm pairwise cache: one first-answer-bias flip and one clean
    win in the free arm, one flip in the paid arm."""
    pw = tmp_path / "pairwise"
    free = pw / FREE_JUDGE_ARM
    paid = pw / PAID_JUDGE_ARM
    free.mkdir(parents=True)
    paid.mkdir(parents=True)

    # free arm: a first-answer-bias flip (slot A wins both orders) + a real win.
    flip = _verdict(
        "easy",
        "q1",
        "mA",
        "mB",
        "mA",
        "mB",
        "Answer A is more concise.",
        "Answer A is more concise.",
        "flip",
    )
    win = _verdict(
        "easy",
        "q2",
        "mA",
        "mB",
        "mA",
        "mA",
        "Answer A is correct.",
        "Answer B is correct.",
        "win",
    )
    (free / "easy__q1__mA__mB.json").write_text(json.dumps(flip))
    (free / "easy__q2__mA__mB.json").write_text(json.dumps(win))

    # paid arm: one flip that is NOT first-answer bias (order2 keeps mA).
    paid_flip = _verdict(
        "easy",
        "q1",
        "mA",
        "mB",
        "mA",
        "mA",
        "Answer A wins.",
        "Answer B wins.",
        "flip",
    )
    (paid / "easy__q1__mA__mB.json").write_text(json.dumps(paid_flip))
    return pw


def test_freeze_copies_every_verdict(fake_pairwise, tmp_path):
    evidence = tmp_path / "evidence"
    counts = freeze_verdicts(fake_pairwise, evidence)
    assert counts == {FREE_JUDGE_ARM: 2, PAID_JUDGE_ARM: 1}
    # every cached verdict now has a frozen twin, byte-for-byte in content
    for arm in (FREE_JUDGE_ARM, PAID_JUDGE_ARM):
        for src in (fake_pairwise / arm).glob("*.json"):
            dst = evidence / "pairwise-verdicts" / arm / src.name
            assert dst.exists()
            assert json.loads(dst.read_text()) == json.loads(src.read_text())


def test_receipts_are_flips_only_and_point_at_frozen_file(fake_pairwise, tmp_path):
    evidence = tmp_path / "evidence"
    receipts = position_bias_receipts(fake_pairwise, evidence, FREE_JUDGE_ARM)
    assert len(receipts) == 1  # only the flip, not the win
    r = receipts[0]
    assert r["item"] == "q1"
    assert r["order1"]["shown_first"] == "mA" and r["order1"]["picked"] == "mA"
    assert r["order2"]["shown_first"] == "mB" and r["order2"]["picked"] == "mB"
    assert r["file"].endswith(f"pairwise-verdicts/{FREE_JUDGE_ARM}/easy__q1__mA__mB.json")


def test_stats_count_first_answer_bias_and_opener(fake_pairwise):
    stats = build_stats(fake_pairwise)
    easy = stats["arms"]["free-local"]["suites"]["easy"]
    assert easy["n_pairs"] == 2
    assert easy["n_flips"] == 1
    assert easy["flip_rate"] == 0.5
    # the single flip is slot-A-both -> first-answer bias 1/1
    assert easy["first_answer_bias_n"] == 1
    assert easy["first_answer_bias_rate"] == 1.0
    # both reasonings of the flip open "Answer A" -> 2/2
    assert easy["answer_a_opener_n"] == 2
    assert easy["answer_a_opener_total"] == 2
    # paid arm flip is NOT first-answer bias (order2 kept mA)
    paid = stats["arms"]["paid-gpt"]["suites"]["easy"]
    assert paid["n_flips"] == 1
    assert paid["first_answer_bias_n"] == 0
