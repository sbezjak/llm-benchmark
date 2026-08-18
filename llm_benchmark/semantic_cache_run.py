"""Run the semantic-cache probe experiment and write its evidence.

  python -m llm_benchmark.semantic_cache_run

Embeds the probe set through the FREE local Ollama lane (needs the embed model
pulled: `ollama pull nomic-embed-text`), sweeps the similarity threshold, and
writes the false-hit list to `evidence/`. The false hit - a stale seed answer
served to a near-neighbour whose correct answer differs - is the finding; the
sweep shows it is a knob (higher threshold = fewer paraphrases caught AND fewer
stale answers served), not a fixed number.

$0: local embeddings only. Not near-instant on a cold model - arm a log monitor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from llm_benchmark.providers.ollama_embed import OllamaEmbedder
from llm_benchmark.semantic_cache import (
    SemanticCache,
    cosine,
    evaluate,
    format_sweep_table,
    load_probe_set,
)

logger = logging.getLogger("llm_benchmark.semantic_cache_run")

DEFAULT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


def _evidence_markdown(probe_set, results, focus_threshold: float) -> str:
    focus = min(results, key=lambda r: abs(r.threshold - focus_threshold))
    lines = [
        "# Semantic cache - false-hit analysis",
        "",
        (
            f"Seeds: {len(probe_set.seeds)}  Probes: {len(probe_set.probes)} "
            f"({sum(p.kind == 'paraphrase' for p in probe_set.probes)} paraphrase, "
            f"{sum(p.kind == 'trap' for p in probe_set.probes)} trap)"
        ),
        "",
        "## Threshold sweep (the hit-rate vs false-hit trade)",
        "",
        "```",
        format_sweep_table(results),
        "```",
        "",
        f"## False hits at threshold {focus.threshold:.2f} (the finding)",
        "",
        "A false hit = the cache served a stored answer to a near-neighbour whose",
        "correct answer differs. Each row is a stale answer a user would have gotten:",
        "",
    ]
    if not focus.false_hits:
        lines.append("_None at this threshold._")
    else:
        lines.append("| probe query | served (stale) | correct | similarity |")
        lines.append("|---|---|---|---|")
        for o in focus.false_hits:
            assert o.hit is not None
            lines.append(
                f"| {o.probe.query} | {o.hit.entry.answer} | {o.probe.expected} "
                f"| {o.hit.similarity:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m llm_benchmark.semantic_cache_run")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--probes", default=None, help="Probe set YAML. Default: the bundled set.")
    p.add_argument(
        "--focus-threshold",
        type=float,
        default=0.85,
        help="Threshold whose false-hit list is written out in full. Default 0.85.",
    )
    p.add_argument(
        "--out",
        default="evidence/semantic-cache-false-hits.md",
        help="Evidence markdown output path.",
    )
    return p.parse_args(argv)


def _receipts(probe_set, seed_vectors, probe_vectors, results, embed_model: str) -> dict:
    """The machine-checkable ground truth behind the .md: every probe's similarity
    to EVERY seed (not just the nearest), plus the per-threshold outcome tallies.
    Anyone can recompute the finding from this - the .md is the read, this is the
    receipt."""
    seed_qs = [q for q, _ in probe_set.seeds]
    probes = []
    for p, pv in zip(probe_set.probes, probe_vectors, strict=True):
        sims = {sq: round(cosine(pv, sv), 6) for sq, sv in zip(seed_qs, seed_vectors, strict=True)}
        nearest = max(sims, key=sims.get)
        probes.append(
            {
                "query": p.query,
                "kind": p.kind,
                "expected": p.expected,
                "nearest_seed": nearest,
                "nearest_similarity": sims[nearest],
                "similarity_to_every_seed": sims,
            }
        )
    return {
        "embed_model": embed_model,
        "seeds": [{"question": q, "answer": a} for q, a in probe_set.seeds],
        "probes": probes,
        "threshold_sweep": [
            {
                "threshold": r.threshold,
                "true_hits": len(r.true_hits),
                "false_hits": len(r.false_hits),
                "misses": len(r.misses),
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    probe_set = load_probe_set(args.probes) if args.probes else load_probe_set()
    embedder = OllamaEmbedder(model=args.embed_model)

    # Embed ONCE (the raw vectors are the ground truth), then evaluate at each
    # threshold off the same vectors - so the receipts and the sweep agree.
    async def _embed():
        seeds = await embedder.embed_many([q for q, _ in probe_set.seeds])
        probes = await embedder.embed_many([p.query for p in probe_set.probes])
        return seeds, probes

    seed_vectors, probe_vectors = asyncio.run(_embed())
    results = []
    for threshold in DEFAULT_THRESHOLDS:
        cache = SemanticCache(threshold=threshold)
        for (q, a), v in zip(probe_set.seeds, seed_vectors, strict=True):
            cache.add(q, a, v)
        results.append(evaluate(cache, probe_set.probes, probe_vectors))
    logger.info("threshold sweep:\n%s", format_sweep_table(results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_evidence_markdown(probe_set, results, args.focus_threshold))
    logger.info("wrote evidence -> %s", out)

    # The receipt: full similarity matrix + sweep tallies, so every number in the
    # .md is recomputable from raw data, not taken on faith.
    receipts_path = out.with_name("semantic-cache-receipts.json")
    receipts_path.write_text(
        json.dumps(
            _receipts(probe_set, seed_vectors, probe_vectors, results, args.embed_model), indent=2
        )
    )
    logger.info("wrote receipts -> %s", receipts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
