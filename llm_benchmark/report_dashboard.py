"""Render the benchmark scoreboard from `findings.json`.

Plain tables with the real model ids in every row - no charts, minimal prose. The
tables carry the data; a one-line read sits under each. Sections, in reading order:

  1. Cost, latency, quality - the per-model table; the paid four tie on quality.
  2. Judge reliability - but the scores above come from a grader that flips its
     verdict when the two answers are swapped.
  3. Self-preference   - and the cheap grader ranks its own answers up.
  4. Limits / What production would add - the honest scope, stated plainly.

Spend is answers PLUS the paid judging that produced the scores (read from
`reports/spend.json` when present; the receipts are cited there). Everything else
is computed off `findings.json` (aggregated off the on-disk cache), so the page
regenerates for $0 with no model call. Self-contained and dependency-free (inline
CSS, no JS, no external asset) so a strict CSP is happy.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

FREE_JUDGE = "free-local"
PAID_JUDGE = "paid-gpt"

LOCAL_MODEL = "llama3.2"  # the free local baseline, flagged in the chrome

# The public repo - "check it" links point at the real files on GitHub so they
# resolve from anywhere the report is opened (Pages, a local file, a blob view).
REPO_URL = "https://github.com/sbezjak/llm-benchmark"

SUITE_LABEL = {"easy": "everyday", "adversarial": "adversarial"}
SUITE_SRC = {"easy": "data/golden_set.yaml", "adversarial": "data/adversarial_set.yaml"}
SUITE_DESC = {
    "easy": "10 items × 2 reps · one undisputed fact each",
    "adversarial": "10 items × 2 reps · one knowable answer each, with a failure trap",
}
_RANK_WORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


def _esc(x: object) -> str:
    return html.escape(str(x))


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _money(x: float) -> str:
    # The local model is free; say so plainly rather than printing $0.000000.
    return "free" if not x else f"${x:.6f}"


# ---------------------------------------------------------------------------
# CSS - token-level so all three theme states (system-light, system-dark, and
# the explicit data-theme toggle) resolve as a set. Colours are only defined
# through tokens; components never hardcode a literal that works in one theme.
# The look is deliberately plain: bordered tables, one blue accent, no charts.
# ---------------------------------------------------------------------------
_CSS = """
:root {
  --bg: #eef1f5; --surface: #ffffff; --surface-2: #f6f8fb;
  --ink: #12161c; --ink-2: #4b5563; --ink-3: #79828f;
  --border: #dde3ea; --border-2: #c8d0da;
  --accent: #2a78d6; --accent-soft: #e9f1fb;
  --warn: #c2410c;
  --row-hi: #f6f8fb;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1216; --surface: #171b21; --surface-2: #1c2129;
    --ink: #eef1f5; --ink-2: #aab3c0; --ink-3: #78828f;
    --border: #262c35; --border-2: #333b46;
    --accent: #4c93e8; --accent-soft: #16273c;
    --warn: #f0824f;
    --row-hi: #1c2129;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1216; --surface: #171b21; --surface-2: #1c2129;
  --ink: #eef1f5; --ink-2: #aab3c0; --ink-3: #78828f;
  --border: #262c35; --border-2: #333b46;
  --accent: #4c93e8; --accent-soft: #16273c;
  --warn: #f0824f;
  --row-hi: #1c2129;
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 780px; margin: 0 auto; padding: 2.6rem 1.25rem 3.5rem; }
.mono { font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace; }

header { margin-bottom: 1.4rem; }
.eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 0.6rem;
}
h1 { font-size: 1.75rem; line-height: 1.15; margin: 0 0 0.5rem; letter-spacing: -0.02em;
     text-wrap: balance; font-weight: 680; }
.lede { color: var(--ink-2); margin: 0.5rem 0 0; font-size: 1.02rem; max-width: 62ch; }
.dateline { font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem; color: var(--ink-3);
            margin: 0.45rem 0 0; letter-spacing: 0.02em; }
dl.legend { display: grid; grid-template-columns: max-content 1fr; gap: 0.28rem 1rem;
            margin: 0.7rem 0 0; max-width: 70ch; font-size: 0.8rem; color: var(--ink-3);
            align-items: baseline; }
dl.legend dt { font-family: ui-monospace, Menlo, monospace; font-size: 0.74rem; font-weight: 600;
               color: var(--ink-2); white-space: nowrap; }
dl.legend dd { margin: 0; line-height: 1.5; }
a.src { color: var(--accent); text-decoration: none; font-family: ui-monospace, Menlo, monospace;
        border-bottom: 1px dotted var(--accent); }
a.src:hover { border-bottom-style: solid; }

/* KPI strip - the run at a glance, modern stat tiles */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 0.6rem;
        margin: 1.4rem 0 0.5rem; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
       padding: 0.7rem 0.85rem; }
.kpi .n { font-size: 1.5rem; font-weight: 680; letter-spacing: -0.02em; color: var(--ink);
          font-variant-numeric: tabular-nums; line-height: 1.1; }
.kpi .l { margin-top: 0.15rem; font-family: ui-monospace, Menlo, monospace; font-size: 0.64rem;
          letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }
.kpi.accent { border-color: var(--accent); background: var(--accent-soft); }
.caveat { color: var(--ink-3); font-size: 0.86rem; margin: 0.5rem 0 0; max-width: 62ch; }
.caveat b { color: var(--ink-2); font-weight: 600; }

h2 { font-size: 1.05rem; letter-spacing: -0.01em; margin: 2.4rem 0 0.2rem; font-weight: 640; }
.bridge { color: var(--ink-2); font-size: 0.95rem; margin: 2.4rem 0 -0.6rem; }
.bridge b { color: var(--ink); font-weight: 620; }
.suite-lab { font-family: ui-monospace, Menlo, monospace; font-size: 0.68rem; letter-spacing: 0.08em;
             text-transform: uppercase; color: var(--ink-3); margin: 0.9rem 0 0.15rem; }

/* tables - the whole point: real model ids in rows, plain and scannable */
.tbl-wrap { overflow-x: auto; margin: 0.35rem 0 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
        font-size: 0.86rem; }
th, td { padding: 0.4rem 0.55rem; border-bottom: 1px solid var(--border); text-align: right;
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid var(--border-2); color: var(--ink-3); font-weight: 600;
           font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em; }
tbody tr:last-child td { border-bottom: none; }
td.model { font-family: ui-monospace, Menlo, monospace; color: var(--ink); font-size: 0.8rem; }
td.q { color: var(--ink); font-weight: 620; }
tr.hi td { background: var(--row-hi); }
.tag { font-family: ui-monospace, Menlo, monospace; font-size: 0.6rem; letter-spacing: 0.03em;
       text-transform: uppercase; color: var(--ink-3); border: 1px solid var(--border);
       border-radius: 4px; padding: 0.02rem 0.3rem; margin-left: 0.35rem; }
.warn-txt { color: var(--warn); font-weight: 620; }

.read { color: var(--ink-2); font-size: 0.9rem; margin: 0.55rem 0 0; }
.read b { color: var(--ink); font-weight: 620; }
.proof { font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem; color: var(--ink-3);
         margin: 0.45rem 0 0; line-height: 1.5; }
.proof b { color: var(--ink-2); font-weight: 600; }

/* inline flip receipt - one real verdict shown verbatim, P4-style */
.receipt { border: 1px solid var(--border); border-left: 3px solid var(--warn); border-radius: 8px;
           background: var(--surface); padding: 0.7rem 0.95rem; margin: 0.8rem 0 0; }
.receipt-h { font-family: ui-monospace, Menlo, monospace; font-size: 0.66rem; letter-spacing: 0.04em;
             text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.5rem; }
.receipt-h b { color: var(--ink-2); font-weight: 600; }
.ord { font-size: 0.84rem; color: var(--ink-2); margin: 0.35rem 0; }
.ord .lab { font-family: ui-monospace, Menlo, monospace; font-size: 0.6rem; text-transform: uppercase;
            letter-spacing: 0.06em; color: var(--accent); border: 1px solid var(--border);
            border-radius: 4px; padding: 0.03rem 0.32rem; margin-right: 0.45rem; }
.ord b { color: var(--ink); font-weight: 620; }
.why { color: var(--ink-3); font-style: italic; font-size: 0.82rem; margin: 0.18rem 0 0 0.2rem; }
.receipt-cap { font-size: 0.84rem; color: var(--ink-2); margin-top: 0.55rem;
               border-top: 1px solid var(--border); padding-top: 0.5rem; }
.receipt-cap b { color: var(--ink); font-weight: 620; }

/* limits + production: terse bullet lists */
.notebox { border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
           padding: 0.6rem 1.15rem; margin: 0.5rem 0 0; }
.notebox.warn { border-left: 3px solid var(--warn); }
.notebox.acc { border-left: 3px solid var(--accent); }
.notebox ul { margin: 0.4rem 0; padding-left: 1.1rem; color: var(--ink-2); font-size: 0.9rem; }
.notebox li { margin: 0.25rem 0; }
.notebox li b { color: var(--ink); font-weight: 620; }

footer { margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid var(--border);
         color: var(--ink-3); font-size: 0.76rem; }
"""


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------
def _quality_table(rows: list[dict]) -> str:
    """Cost / latency / quality for one suite, real model id per row. Rows arrive
    already sorted by score (findings.json); the local row is highlighted."""
    head = (
        "<tr><th>model</th><th>$ / query</th><th>lat p50</th><th>lat p95</th>"
        "<th>out tok</th><th>quality</th><th>95% CI</th><th>pass</th></tr>"
    )
    body = []
    for r in rows:
        m = r["model"]
        local = m == LOCAL_MODEL
        tag = '<span class="tag">local</span>' if local else ""
        q = f"{r['mean_score']:.3f}"
        qcell = f'<span class="warn-txt">{q}</span>' if local else q
        body.append(
            f'<tr class="{"hi" if local else ""}">'
            f'<td class="model">{_esc(m)}{tag}</td>'
            f"<td>{_money(r['mean_cost_usd'])}</td>"
            f"<td>{r['lat_p50_ms']:.0f}</td><td>{r['lat_p95_ms']:.0f}</td>"
            f"<td>{r['mean_tokens_out']:.0f}</td>"
            f'<td class="q">{qcell}</td>'
            f"<td>[{r['score_ci_lo']:.3f}, {r['score_ci_hi']:.3f}]</td>"
            f"<td>{r['passes']}/{r['n']}</td></tr>"
        )
    return (
        f'<div class="tbl-wrap"><table><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _judge_table(f: dict) -> str:
    """One row per (grader, suite): flip rate under the swap, flip count, and how
    many of those flips just kept the answer shown first. The cheap grader is a
    contestant (llama3.2); the paid one (gpt-5.6-luna) is shown for contrast."""
    head = (
        "<tr><th>grader</th><th>suite</th><th>flip rate</th><th>flips</th>"
        "<th>kept first-shown</th></tr>"
    )
    body = []
    for judge in (FREE_JUDGE, PAID_JUDGE):
        grader = f["pairwise"]["judge_models"][judge]
        cheap = judge == FREE_JUDGE
        for suite in f["suites"]:
            v = f["pairwise"]["views"][suite][judge]
            fr, flips, wins, bias = (
                v["flip_rate"],
                v["n_flips"],
                v["first_answer_wins"],
                v["first_answer_bias"],
            )
            fr_cell = f'<span class="warn-txt">{_pct(fr)}</span>' if fr >= 0.4 else _pct(fr)
            kept = f"{wins} ({_pct(bias)})"
            kept_cell = f'<span class="warn-txt">{kept}</span>' if bias >= 0.5 else kept
            tag = '<span class="tag">contestant</span>' if cheap else ""
            body.append(
                f'<tr class="{"hi" if cheap else ""}">'
                f'<td class="model">{_esc(grader)}{tag}</td>'
                f"<td>{_esc(SUITE_LABEL.get(suite, suite))}</td>"
                f"<td>{fr_cell}</td><td>{flips}</td><td>{kept_cell}</td></tr>"
            )
    return (
        f'<div class="tbl-wrap"><table><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _selfpref_table(f: dict, suite: str) -> tuple[str, int, int]:
    """The cheap grader's head-to-head win rate (its own arm) beside each model's
    solo score (paid grader), sorted by win rate. Returns the table plus llama's
    head-to-head rank and the field size, for the read line."""
    standings = f["pairwise"]["views"][suite][FREE_JUDGE]["standings"]
    solo = {m["model"]: m["mean_score"] for m in f["views"][suite][PAID_JUDGE]["models"]}
    rank = next((i + 1 for i, s in enumerate(standings) if s["model"] == LOCAL_MODEL), 0)
    head = "<tr><th>model</th><th>head-to-head win</th><th>solo score</th></tr>"
    body = []
    for s in standings:
        m = s["model"]
        local = m == LOCAL_MODEL
        tag = '<span class="tag">grader</span>' if local else ""
        sc = solo.get(m)
        sc_txt = f"{sc:.3f}" if sc is not None else "-"
        body.append(
            f'<tr class="{"hi" if local else ""}">'
            f'<td class="model">{_esc(m)}{tag}</td>'
            f"<td>{s['win_rate']:.3f}</td><td>{sc_txt}</td></tr>"
        )
    tbl = (
        f'<div class="tbl-wrap"><table><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
    return tbl, rank, len(standings)


# ---------------------------------------------------------------------------
# page sections
# ---------------------------------------------------------------------------
def _src(path: str, label: str | None = None) -> str:
    """Link a repo file/dir on GitHub so it resolves from anywhere the report is
    viewed. Directories (trailing /) use /tree, files use /blob, on the default
    branch. The visible label defaults to the repo-relative path."""
    p = path.rstrip("/")
    kind = "tree" if path.endswith("/") else "blob"
    href = f"{REPO_URL}/{kind}/main/{p}"
    return f'<a class="src" href="{href}">{_esc(label or path)}</a>'


def _proof(text: str) -> str:
    return f'<p class="proof"><b>Check it:</b> {text}</p>'


def _first_answer_bias(f: dict, judge: str) -> tuple[int, int]:
    """Overall (both suites) flips where the first-shown answer won both orders,
    and total flips - straight off the computed pairwise views."""
    wins = sum(f["pairwise"]["views"][s][judge]["first_answer_wins"] for s in f["suites"])
    flips = sum(f["pairwise"]["views"][s][judge]["n_flips"] for s in f["suites"])
    return wins, flips


def _suite_caption(suite: str) -> str:
    label = _esc(SUITE_LABEL.get(suite, suite))
    desc = _esc(SUITE_DESC.get(suite, ""))
    src = SUITE_SRC.get(suite)
    link = f" · {_src(src)}" if src else ""
    return f'<div class="suite-lab">{label} suite - {desc}{link}</div>'


_LEGEND = (
    '<dl class="legend">'
    "<dt>$ / query</dt><dd>average cost of one answer</dd>"
    "<dt>lat p50</dt><dd>typical response time (half the calls were faster)</dd>"
    "<dt>lat p95</dt><dd>slow-case time - only 1 call in 20 was slower; the lag you actually "
    "feel</dd>"
    "<dt>out tok</dt><dd>answer length in output tokens (drives cost and speed)</dd>"
    "<dt>quality</dt><dd>average judge score, 0 to 1 (1 = fully correct)</dd>"
    "<dt>95% CI</dt><dd>the range the real score could land in on a rerun of this small sample; "
    "if two ranges overlap, the difference could be luck</dd>"
    "<dt>pass</dt><dd>answers scoring at least 0.7, out of the total</dd>"
    "</dl>"
)


def _finding_tie(f: dict) -> str:
    tables = []
    for suite in f["suites"]:
        rows = f["views"][suite][PAID_JUDGE]["models"]
        tables.append(_suite_caption(suite) + _quality_table(rows))
    return (
        "<h2>Cost, latency, quality</h2>"
        + "".join(tables)
        + _LEGEND
        + '<p class="read">Quality is the paid grader\'s solo score; cost and latency come from the '
        "saved calls. The four paid models overlap on quality - no winner - so the tie-break is cost "
        "and speed: <b>gpt-5.6-luna</b> is cheapest and fastest. Local is free, slowest, and the "
        "only quality dip.</p>"
        + _proof(
            "every cell rebuilds from "
            + _src("reports/findings.json")
            + " (aggregated off the on-disk answer cache) - no model is re-called."
        )
    )


def _flip_example(receipts: list | None) -> str:
    """Inline one real flip, verbatim, so the reader sees the bias, not just a rate.
    Picks the first receipt where the grader picked the FIRST-shown answer in both
    orders (pure position bias) - drawn live from the receipts file, so it can't
    drift from the data. Returns "" when no receipts are available."""
    if not receipts:
        return ""
    rec = next(
        (
            r
            for r in receipts
            if r["order1"]["picked"] == r["order1"]["shown_first"]
            and r["order2"]["picked"] == r["order2"]["shown_first"]
        ),
        None,
    )
    if rec is None:
        return ""
    o1, o2 = rec["order1"], rec["order2"]

    def _ord(n: int, o: dict) -> str:
        return (
            f'<div class="ord"><span class="lab">order {n}</span> shown first '
            f"<b>{_esc(o['shown_first'])}</b>, picked <b>slot A</b>"
            f'<div class="why">“{_esc(o["reason"])}”</div></div>'
        )

    link = _src(rec["file"], "full verdict")
    return (
        '<div class="receipt">'
        f'<div class="receipt-h">One flip, verbatim - item <b>{_esc(rec["item"])}</b> · grader '
        f"<b>{LOCAL_MODEL}</b> · {_esc(rec['model_a'])} vs {_esc(rec['model_b'])}</div>"
        + _ord(1, o1)
        + _ord(2, o2)
        + '<div class="receipt-cap">Same two answers, order swapped. The grader picked '
        "<b>“slot A” both times</b> - it followed the position, not the answer. "
        + link
        + "</div></div>"
    )


def _finding_judge(f: dict, receipts: list | None = None) -> str:
    cheap_wins, cheap_flips = _first_answer_bias(f, FREE_JUDGE)
    paid_wins, paid_flips = _first_answer_bias(f, PAID_JUDGE)
    cheap_pct = round(100 * cheap_wins / cheap_flips) if cheap_flips else 0
    paid_pct = round(100 * paid_wins / paid_flips) if paid_flips else 0
    max_flip = max(f["pairwise"]["views"][s][FREE_JUDGE]["flip_rate"] for s in f["suites"])
    return (
        "<h2>Judge reliability - position bias</h2>"
        + f'<p class="read"><b>Position bias</b> is a known LLM-judge failure: the verdict follows '
        "which answer comes first, not which is better. <b>How it is tested:</b> show the grader the "
        "same two answers in both orders and check the verdict holds. <b>llama3.2</b> (itself a "
        f"contestant) flips on up to <b>{round(100 * max_flip)}%</b> of pairs, and "
        f"<b>{cheap_pct}%</b> of those flips just follow whichever came first (paid grader: "
        f"<b>{paid_pct}%</b>) - order, not quality, decided it.</p>"
        + _judge_table(f)
        + _flip_example(receipts)
        + _proof(
            "the flip above is one of 200 both-orders verdicts under "
            + _src("evidence/pairwise-verdicts/llama3.2/")
            + "; the aggregate is "
            + _src("evidence/judge-position-bias-stats.json")
            + ", written up in "
            + _src("evidence/judge-position-bias.md")
            + "."
        )
    )


def _selfpref_example(verdict: dict | None) -> str:
    """Inline one real self-win, verbatim: llama3.2 as grader picking its OWN answer
    over a stronger model's, in both orders. Drawn from the winning verdict file so
    it can't drift. Returns "" when none is supplied."""
    if not verdict:
        return ""
    a, b = verdict["model_a"], verdict["model_b"]
    opp = a if b == LOCAL_MODEL else b
    # order 2 swaps the pair, so there "Answer A" is model_b and "Answer B" is model_a
    own_slot = "A" if b == LOCAL_MODEL else "B"
    reason = verdict["order2_reasoning"]
    link = _src(verdict["_file"], "full verdict")
    return (
        '<div class="receipt">'
        f'<div class="receipt-h">One self-win, verbatim - item <b>{_esc(verdict["item_id"])}</b> · '
        f"grader <b>{LOCAL_MODEL}</b> · {LOCAL_MODEL} vs {_esc(opp)}</div>"
        '<div class="ord"><span class="lab">both orders</span> picked <b>its own answer</b> '
        "(a consistent win, not a flip)"
        f'<div class="why">its reason (its own answer is “Answer {own_slot}” here): '
        f"“{_esc(reason)}”</div></div>"
        '<div class="receipt-cap">A small local model, grading its own answer against '
        f"<b>{_esc(opp)}</b>, rates itself the winner. Across the pool it ranks itself third of "
        "five, while the independent paid grader ranks it last. " + link + "</div></div>"
    )


def _finding_selfpref(f: dict, selfpref: dict | None = None) -> str:
    suite = f["suites"][0]
    tbl, rank, size = _selfpref_table(f, suite)
    rank_txt = _RANK_WORD.get(rank, str(rank))
    return (
        "<h2>Self-preference</h2>"
        + f'<p class="read"><b>Self-preference bias</b> (self-enhancement): a model-judge rating '
        "its own answers higher than a neutral judge does. <b>How it is tested:</b> let "
        "<b>llama3.2</b> grade a pool that includes its own answers. The left column is how often "
        "each model wins under llama's judging; the right is its score from the independent paid "
        f"grader. llama ranks itself <b>{rank_txt} of {size}</b> by its own judging, but the neutral "
        "grader puts it <b>last</b> - it favours itself. A grader should sit outside the pool it "
        "scores.</p>"
        + tbl
        + _selfpref_example(selfpref)
        + _proof(
            "the head-to-head standings are the llama3.2 arm of the pairwise view in "
            + _src("reports/findings.json")
            + ", built from the verdicts in "
            + _src("evidence/pairwise-verdicts/llama3.2/")
            + "."
        )
    )


def _limits() -> str:
    return (
        '<h2>Limits</h2><div class="notebox warn"><ul>'
        "<li><b>No best model.</b> Too close, and the grader is biased.</li>"
        "<li><b>Small n.</b> 2 reps can't resolve a 0.003 gap.</li>"
        "<li><b>One suite.</b> 20 items - other questions could shift it.</li>"
        "</ul></div>"
    )


def _production() -> str:
    return (
        '<h2>What production would add</h2><div class="notebox acc"><ul>'
        "<li><b>Grader outside the pool</b>, or a panel of judges by majority.</li>"
        "<li><b>More items, standard sets</b> - hundreds to thousands, shared and public.</li>"
        "<li><b>Extra reps only where scores are close.</b></li>"
        "<li><b>Human spot-checks</b> and drift monitoring in production.</li>"
        "</ul></div>"
    )


def _kpis(findings: dict, n_items: int, n_reps: int, spend: dict | None) -> tuple[str, str]:
    """The stat-tile strip + a spend caveat line. Total spend is answers plus the
    paid judging that produced the scores (from spend.json); if absent, fall back
    to the answer-only figure in findings.json and label it as such."""
    n_ans = findings["n_captures"]
    if spend and "total_usd" in spend:
        total = spend["total_usd"]
        judging = spend.get("solo_judge_gpt_usd", 0.0) + spend.get("pairwise_judge_gpt_usd", 0.0)
        answers = spend.get("answer_generation_usd", total - judging)
        caveat = (
            f'<p class="caveat">Spend = answers <b>${answers:.2f}</b> + paid judging '
            f"<b>${judging:.2f}</b>. The grader is one of the five tested models - read every "
            "quality number as a demo, not a verdict.</p>"
        )
    else:
        total = findings["spend"]["total_answer_cost_usd"]
        caveat = (
            '<p class="caveat">Spend is answer generation only. The grader is also one of the '
            "tested models, so treat every number as a demo, not a verdict.</p>"
        )
    tiles = [
        (str(n_items), "items", False),
        (str(n_reps), "reps", False),
        (str(len(findings["models"])), "models", False),
        (str(n_ans), "graded answers", False),
        (f"${total:.2f}", "total spend", True),
    ]
    cells = "".join(
        f'<div class="kpi{" accent" if acc else ""}"><div class="n">{_esc(n)}</div>'
        f'<div class="l">{_esc(lab)}</div></div>'
        for n, lab, acc in tiles
    )
    return f'<div class="kpis">{cells}</div>', caveat


def render_dashboard(
    findings: dict,
    out_path: Path | str,
    *,
    title: str,
    n_items: int = 20,
    n_reps: int = 2,
    spend: dict | None = None,
    pairwise_receipts: list | None = None,
    selfpref_example: dict | None = None,
) -> Path:
    """Build the self-contained scoreboard HTML from a `findings.json` dict and
    write it. Pure render - no provider call, $0 - so it regenerates as often as
    the analysis is revised. `n_items`/`n_reps` describe the run in the stat tiles;
    `spend` (from spend.json) carries the answers-plus-judging total;
    `pairwise_receipts` (from judge-position-bias-receipts.json) supplies the one
    verbatim flip inlined under the judge section; `selfpref_example` (a winning
    verdict dict) supplies the verbatim self-win under self-preference. Returns the
    path written."""
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    kpis, caveat = _kpis(findings, n_items, n_reps, spend)

    body = [
        '<div class="wrap">',
        "<header>",
        '<p class="eyebrow">Cost · latency · quality</p>',
        f"<h1>{_esc(title)}</h1>",
        f'<p class="dateline">Generated {generated} · rebuilt from the cached answers, $0</p>',
        (
            '<p class="lede">Five models, one 20-item suite, 2 reps. The paid models tie on quality '
            "- so the measurable result is grader instability, not a ranking.</p>"
        ),
        kpis,
        caveat,
        "</header>",
        _finding_tie(findings),
        _finding_judge(findings, pairwise_receipts),
        _finding_selfpref(findings, selfpref_example),
        _limits(),
        _production(),
        (
            "<footer>Regenerate ($0, no model calls): "
            '<span class="mono">python -m llm_benchmark.report_dashboard</span>. '
            "Sources: quality, cost and latency from "
            + _src("reports/findings.json")
            + "; spend from "
            + _src("reports/spend.json")
            + "; raw pairwise verdicts under "
            + _src("evidence/pairwise-verdicts/")
            + ".</footer>"
        ),
        "</div>",
    ]

    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path


DEFAULT_FINDINGS_PATH = Path("reports") / "findings.json"
DEFAULT_SPEND_PATH = Path("reports") / "spend.json"
DEFAULT_RECEIPTS_PATH = Path("evidence") / "judge-position-bias-receipts.json"
DEFAULT_VERDICTS_DIR = Path("evidence") / "pairwise-verdicts" / LOCAL_MODEL

# Opponents strong enough that a self-win is a clean self-preference receipt; the
# rank also makes the pick deterministic (prefer the most impressive opponent).
_STRONG_RANK = {
    "claude-sonnet-5": 0,
    "claude-haiku-4-5-20251001": 1,
    "gpt-5.6-luna": 2,
    "deepseek-v4-pro": 3,
}


def _find_selfpref_example(verdict_dir: Path) -> dict | None:
    """Scan llama3.2's verdict files for one consistent self-win over a strong
    model (both orders picked llama). Deterministic: the strongest opponent, then
    suite/item order. Returns the verdict dict (with a `_file` repo path) or None."""
    if not verdict_dir.is_dir():
        return None
    best: tuple[tuple, dict] | None = None
    for f in sorted(verdict_dir.glob("*.json")):
        d = json.loads(f.read_text())
        a, b = d.get("model_a"), d.get("model_b")
        if LOCAL_MODEL not in (a, b) or d.get("winner") != LOCAL_MODEL:
            continue
        opp = a if b == LOCAL_MODEL else b
        if opp not in _STRONG_RANK:
            continue
        d["_file"] = f"evidence/pairwise-verdicts/{LOCAL_MODEL}/{f.name}"
        key = (_STRONG_RANK[opp], d.get("suite", ""), d.get("item_id", ""))
        if best is None or key < best[0]:
            best = (key, d)
    return best[1] if best else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    today = dt.datetime.now(dt.UTC).date()
    p = argparse.ArgumentParser(
        prog="python -m llm_benchmark.report_dashboard",
        description="Render the benchmark scoreboard off findings.json ($0).",
    )
    p.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS_PATH)
    p.add_argument("--spend", type=Path, default=DEFAULT_SPEND_PATH)
    p.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS_PATH)
    p.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS_DIR)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports") / f"report-benchmark-scoreboard-{today}.html",
    )
    p.add_argument("--title", default="Five models: cost, latency, quality")
    args = p.parse_args(argv)

    findings = json.loads(args.findings.read_text())
    spend = json.loads(args.spend.read_text()) if args.spend and args.spend.exists() else None
    receipts = (
        json.loads(args.receipts.read_text()) if args.receipts and args.receipts.exists() else None
    )
    selfpref = _find_selfpref_example(args.verdicts)
    out = render_dashboard(
        findings,
        args.out,
        title=args.title,
        spend=spend,
        pairwise_receipts=receipts,
        selfpref_example=selfpref,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
