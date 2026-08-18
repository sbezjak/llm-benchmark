"""Calibration generator: pure ($0, reads the judged cache). The load-bearing
assertion is BLINDNESS - the grading sheet must not leak the judge's score, band,
passed flag, or reasoning, or the human grade is contaminated and the calibration is
worthless. Also checks the sample is stratified (every non-perfect answer + perfect
controls per model) and that sheet ids align with the sealed key.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_benchmark.calibration import build_key, build_sheet, judge_band, main, sample


def _rec(model, item, rep, score, passed, reason="because"):
    return {
        "answer_model": model,
        "item_id": item,
        "rep": rep,
        "prompt": f"Q for {item}?",
        "expected": "the expected answer",
        "answer_text": f"answer from {model} for {item}",
        "judge": {"score": score, "passed": passed, "reason": reason},
    }


def _records():
    # two non-perfect (one fail, one borderline) + four perfect across two models
    return [
        _rec("llama3.2", "adv_count_001", 1, 0.5, False, "miscounts the letters"),
        _rec("llama3.2", "reasoning_003", 1, 0.9, True, "minor science slip"),
        _rec("gpt-5.6-luna", "factual_001", 1, 1.0, True, "perfect"),
        _rec("gpt-5.6-luna", "factual_002", 1, 1.0, True, "perfect"),
        _rec("gpt-5.6-luna", "factual_003", 1, 1.0, True, "perfect"),
        _rec("llama3.2", "definition_001", 1, 1.0, True, "perfect"),
    ]


def test_judge_band_is_binary_no_weak():
    assert judge_band(True) == "Good"
    assert judge_band(False) == "Wrong"


def test_sample_includes_every_nonperfect_plus_perfect_controls():
    chosen = sample(_records(), perfect_per_model=1)
    keys = {(r["answer_model"], r["item_id"], r["rep"]) for r in chosen}
    # both non-perfect answers must be in
    assert ("llama3.2", "adv_count_001", 1) in keys
    assert ("llama3.2", "reasoning_003", 1) in keys
    # exactly one perfect control per model (2 models -> 2 perfect picks)
    perfect = [r for r in chosen if r["judge"]["score"] >= 1.0]
    assert len(perfect) == 2
    assert {r["answer_model"] for r in perfect} == {"gpt-5.6-luna", "llama3.2"}


def test_sheet_is_blind_leaks_no_judge_signal():
    chosen = sample(_records(), perfect_per_model=1)
    sheet = build_sheet(chosen)
    # the judge's reasons must never appear in the sheet
    assert "miscounts the letters" not in sheet
    assert "minor science slip" not in sheet
    # no score/band/pass leakage
    for banned in ("judge_score", "judge_band", "0.5", "0.9", "passed", "Wrong -", "correctness"):
        assert banned not in sheet, f"sheet leaks judge signal: {banned!r}"
    # but the answer text and the empty band boxes must be there
    assert "answer from llama3.2 for adv_count_001" in sheet
    assert "[ ] Good" in sheet and "[ ] Weak" in sheet and "[ ] Wrong" in sheet


def test_key_ids_align_with_sheet_and_carry_the_signal():
    chosen = sample(_records(), perfect_per_model=1)
    key = build_key(chosen)
    sheet = build_sheet(chosen)
    for gid in key:
        assert f"### {gid} " in sheet
    # the sealed key DOES carry the score + reasoning (it is the answer key)
    assert any(e["judge_reason"] == "miscounts the letters" for e in key.values())
    assert {e["judge_band"] for e in key.values()} <= {"Good", "Wrong"}


def test_main_never_clobbers_a_graded_sheet(tmp_path: Path):
    """The regression guard for the near-miss: once a human has put an [x] on the
    sheet, regenerating must NOT overwrite it (the key still regenerates, aligned)."""
    judged = tmp_path / "judged"
    judged.mkdir()
    for i, r in enumerate(_records()):
        (judged / f"g{i}.json").write_text(json.dumps(r))
    sheet = tmp_path / "sheet.md"
    key = tmp_path / "key.json"

    # first generation writes a blank sheet
    main(["--judged-dir", str(judged), "--sheet", str(sheet), "--key", str(key)])
    assert "[x]" not in sheet.read_text()

    # a human grades it
    graded = sheet.read_text().replace("[ ] Good", "[x] Good", 1)
    sheet.write_text(graded)

    # regenerating must leave the graded sheet untouched (but still rewrite the key)
    main(["--judged-dir", str(judged), "--sheet", str(sheet), "--key", str(key)])
    assert sheet.read_text() == graded
    assert key.exists()
