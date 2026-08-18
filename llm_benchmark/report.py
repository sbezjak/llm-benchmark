"""Render the benchmark report from cached captures - the reporting step.

A production benchmark separates three things a single pytest run conflates: the
*job* that calls the models, the *artifact store* it writes to, and the *report*
someone reads. This module is the third. It takes the captured records (read
back from `cache/`, never a fresh call) and emits one self-contained HTML file:
the cost/latency/quality table, the validity table (score with a 95% interval,
tail latency, cost per passed answer), the paired significance read, and - the
part that makes the raw output the ground truth - every prompt and response
verbatim, one collapsible block per (model, item, rep).

Self-contained and dependency-free on purpose: string-built HTML with inline CSS,
no template engine and no pytest-html. The report is an artifact of the data, so
it can be regenerated off the cache for $0 whenever the analysis changes.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

from llm_benchmark.runners.stats import (
    ModelStats,
    PairedDiff,
    model_stats,
    paired_score_diffs,
)
from llm_benchmark.runners.sweep import ModelSummary, summarize

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
.meta { color: #666; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;
        font-variant-numeric: tabular-nums; }
th, td { padding: 0.35rem 0.6rem; border-bottom: 1px solid #ccc; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid #888; }
.tie { color: #888; } .gap { font-weight: 600; }
details { border: 1px solid #ccc; border-radius: 6px; margin: 0.4rem 0; padding: 0.3rem 0.6rem; }
summary { cursor: pointer; font-variant-numeric: tabular-nums; }
pre { white-space: pre-wrap; word-break: break-word; background: rgba(127,127,127,0.08);
      padding: 0.5rem; border-radius: 4px; margin: 0.4rem 0; }
.label { color: #666; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; }
.ans { color: #888; font-style: italic; font-size: 0.85rem; margin: 0.25rem 0 0;
       white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
"""


def _esc(x: object) -> str:
    return html.escape(str(x))


def _summary_table(summaries: list[ModelSummary]) -> str:
    head = (
        "<tr><th>model</th><th>n</th><th>mean $/q</th><th>total $</th>"
        "<th>mean lat ms</th><th>mean ttft ms</th><th>out tok</th>"
        "<th>score</th><th>pass</th></tr>"
    )
    rows = []
    for s in summaries:
        ttft = f"{s.mean_ttft_ms:.0f}" if s.mean_ttft_ms is not None else "n/a"
        rows.append(
            f"<tr><td>{_esc(s.model)}</td><td>{s.n}</td>"
            f"<td>{s.mean_cost_usd:.6f}</td><td>{s.total_cost_usd:.5f}</td>"
            f"<td>{s.mean_latency_ms:.0f}</td><td>{ttft}</td>"
            f"<td>{s.mean_tokens_out:.1f}</td><td>{s.mean_score:.3f}</td>"
            f"<td>{s.pass_rate:.0%}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _stats_table(stats: list[ModelStats]) -> str:
    head = (
        "<tr><th>model</th><th>score</th><th>95% CI</th>"
        "<th>lat p50 ms</th><th>lat p95 ms</th><th>passes</th><th>$/passed</th></tr>"
    )
    rows = []
    for s in stats:
        per_pass = f"{s.cost_per_pass_usd:.6f}" if s.cost_per_pass_usd is not None else "n/a"
        rows.append(
            f"<tr><td>{_esc(s.model)}</td><td>{s.mean_score:.3f}</td>"
            f"<td>[{s.score_ci_lo:.3f}, {s.score_ci_hi:.3f}]</td>"
            f"<td>{s.lat_p50_ms:.0f}</td><td>{s.lat_p95_ms:.0f}</td>"
            f"<td>{s.passes}/{s.n}</td><td>{per_pass}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _paired_table(baseline: str, diffs: list[PairedDiff]) -> str:
    head = (
        f"<tr><th>{_esc(baseline)} minus</th><th>pairs</th><th>mean diff</th>"
        "<th>95% CI</th><th>read</th></tr>"
    )
    rows = []
    for d in diffs:
        cls = "gap" if d.ci_excludes_zero else "tie"
        read = "gap (CI excludes 0)" if d.ci_excludes_zero else "tie (CI contains 0)"
        rows.append(
            f"<tr><td>{_esc(d.model)}</td><td>{d.n_pairs}</td>"
            f"<td>{d.mean_diff:+.3f}</td>"
            f"<td>[{d.ci_lo:+.3f}, {d.ci_hi:+.3f}]</td>"
            f"<td class='{cls}'>{read}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _trace_block(rec: dict, fallback_judge: str | None = None) -> str:
    m = rec["measurement"]
    j = rec["judge"]
    ttft = f"{m['ttft_ms']:.0f}ms" if m.get("ttft_ms") is not None else "n/a"
    verdict = "PASS" if j["passed"] else "FAIL"
    # WHO graded this row: the judge model persisted on the capture, else the
    # run's judge model (older captures predate the field).
    judge_model = j.get("judge_model") or fallback_judge or "unknown"
    head = (
        f"{_esc(rec['model'])} &middot; {_esc(rec['item_id'])} &middot; rep{rec['rep']} "
        f"&middot; score {j['score']:.2f} {verdict} &middot; judge {_esc(judge_model)} "
        f"&middot; {m['latency_ms']:.0f}ms &middot; ttft {ttft} &middot; {m['tokens_out']} tok "
        f"&middot; ${rec['cost_usd']:.6f}"
    )
    # The judge's own receipts when the capture kept them: its tokens and cost.
    bits = [f"model {_esc(judge_model)}"]
    jt_in, jt_out = j.get("judge_tokens_in"), j.get("judge_tokens_out")
    if jt_in is not None or jt_out is not None:
        bits.append(f"{jt_in or 0}&rarr;{jt_out or 0} tok")
    jcost = j.get("judge_cost_usd")
    if jcost is not None:
        bits.append("free" if not jcost else f"${jcost:.6f}")
    judge_meta = " &middot; ".join(bits)
    # A preview of the actual answer, in the always-visible summary, so a wrong
    # answer is scannable and findable without expanding every block (the raw
    # output is the ground truth, so it should not hide behind a collapsed toggle).
    preview = " ".join(m["text"].split())
    if len(preview) > 140:
        preview = preview[:140] + "…"
    return (
        "<details><summary>" + head + f'<div class="ans">&ldquo;{_esc(preview)}&rdquo;</div>'
        "</summary>"
        f"<div class='label'>prompt</div><pre>{_esc(rec['prompt'])}</pre>"
        f"<div class='label'>expected</div><pre>{_esc(rec['expected'])}</pre>"
        f"<div class='label'>response</div><pre>{_esc(m['text'])}</pre>"
        f"<div class='label'>judge &middot; {judge_meta}</div><pre>{_esc(j['reason'])}</pre>"
        "</details>"
    )


def render_report(
    records: list[dict], out_path: Path | str, *, title: str, judge_model: str | None = None
) -> Path:
    """Build the self-contained HTML report from cached records and write it.

    All three tables plus the full verbatim trace come off the records alone -
    no provider call - so this runs for $0 as many times as the analysis is
    revised. `judge_model` is the run's grader, shown up front and used as the
    per-trace fallback for captures that predate the stored judge field. Returns
    the path written."""
    summaries = summarize(records)
    stats = model_stats(records)
    baseline = stats[0].model if stats else ""
    diffs = paired_score_diffs(records, baseline_model=baseline) if stats else []

    total_cost = sum(r["cost_usd"] for r in records)
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    traces = "".join(_trace_block(r, judge_model) for r in records)
    # Name the grader up front: the models seen on the captures, else the run's.
    seen = sorted({jm for r in records if (jm := r["judge"].get("judge_model"))})
    graded_by = ", ".join(seen) if seen else (judge_model or "free local judge")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style></head><body>
<h1>{_esc(title)}</h1>
<p class="meta">{len(records)} captures &middot; total spend ${total_cost:.5f}
 &middot; graded by {_esc(graded_by)} &middot; generated {generated}
 &middot; computed off cached captures (no re-call)</p>

<h2>Cost / latency / quality</h2>
<p class="meta">Quality is the score from the grader named above ({_esc(graded_by)});
every trace below also shows which model graded it and that judge's own tokens/cost.</p>
{_summary_table(summaries)}

<h2>Benchmark validity - score with 95% interval, tail latency, cost per passed answer</h2>
{_stats_table(stats)}

<h2>Paired significance - is a score difference real, or noise?</h2>
<p class="meta">Baseline is the top-ranked model; each row bootstraps the per-item
score difference. A CI that contains 0 means the two are a statistical tie on this
suite. The judge emits discrete scores, so weigh the mean diff, not just the sign.</p>
{_paired_table(baseline, diffs)}

<h2>Full trace - every prompt and response verbatim</h2>
<p class="meta">One block per (model, item, rep). The raw output is the ground truth;
the tables above are all computed from these.</p>
{traces}
</body></html>"""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path
