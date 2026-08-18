"""Semantic (embedding-nearest) response cache - and the failure it exists to find.

A semantic cache serves a stored answer when a NEW query is close enough (cosine
similarity over embeddings) to a query it already answered. The production pitch
is cost/latency: a reworded question hits the cache instead of re-calling the
model. On this suite's 20 standalone questions the naive hit rate is ~0 by
construction (no two questions collide), so a naive demo would prove nothing -
that is exactly the trap this module is built to avoid.

To make the result real, the experiment adds a small probe set (`data/
semantic_cache_probes.yaml`): each seed question is asked again as PARAPHRASES
(should hit, same answer) and sat next to near-neighbor TRAPS (a different
question whose vector is close but whose correct answer differs). Then the
interesting result appears, and it is a FAILURE:

  the FALSE HIT - the cache serves a stale seed answer to a near-neighbor whose
  correct answer is different, because similarity crossed the threshold.

The false hit is the whole thesis of the piece - where does the system fail -
so the deliverable is the list of false hits at a given threshold, not the
headline hit rate. Raising the threshold trades hit rate (fewer paraphrases
caught) against false hits (fewer stale answers served); the experiment reports
both so the trade is visible, not asserted.

The cache core here is pure vector math - no I/O - so it is unit-testable with
fabricated vectors. The embedding call lives behind `providers/ollama_embed.py`
(the free local lane); `run_probe_experiment` is the thin seam that does the
embedding I/O and then drives this pure core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_THRESHOLD = 0.85
DEFAULT_PROBE_SET_PATH = Path(__file__).parent.parent / "data" / "semantic_cache_probes.yaml"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. 0.0 when either is the zero vector -
    an undefined angle is treated as 'not similar', never a divide-by-zero."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class CacheEntry:
    """One cached (question -> answer) with the embedding that keys it."""

    question: str
    answer: str
    embedding: list[float]


@dataclass(frozen=True)
class CacheHit:
    """The nearest entry above threshold, and how near it was."""

    entry: CacheEntry
    similarity: float


class SemanticCache:
    """Embedding-nearest cache: serve a stored answer when a new query's vector
    is within `threshold` cosine similarity of a stored query's vector.

    Pure and I/O-free - it operates on vectors the caller already computed, so
    the cache logic (nearest-neighbor + threshold) is testable without a model.
    Linear scan: the suite is tiny, so a vector index (FAISS/hnswlib) would be
    machinery the task didn't ask for - the named production pattern, and the
    named reason it's not built here."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self.entries: list[CacheEntry] = []

    def add(self, question: str, answer: str, embedding: list[float]) -> None:
        self.entries.append(CacheEntry(question=question, answer=answer, embedding=embedding))

    def lookup(self, embedding: list[float]) -> CacheHit | None:
        """The single nearest entry with similarity >= threshold, or None (a
        miss - which in production falls through to a real model call)."""
        best: CacheHit | None = None
        for entry in self.entries:
            sim = cosine(embedding, entry.embedding)
            if sim >= self.threshold and (best is None or sim > best.similarity):
                best = CacheHit(entry=entry, similarity=sim)
        return best


def _normalize(answer: str) -> str:
    """Loose answer match for short factual answers: case-insensitive, trimmed.
    A served answer 'counts as correct' for a probe if each contains the other
    (so 'Ljubljana' matches 'Ljubljana.' and 'The capital is Ljubljana')."""
    return answer.strip().lower().rstrip(".")


def _answers_match(served: str, expected: str) -> bool:
    s, e = _normalize(served), _normalize(expected)
    if not s or not e:
        return s == e
    return s in e or e in s


@dataclass(frozen=True)
class Probe:
    """A query put to the cache, with the answer it SHOULD get and its kind.
    `kind` is 'paraphrase' (should hit its seed, same answer) or 'trap' (a
    near-neighbor whose correct answer differs - a hit here is a false hit)."""

    query: str
    expected: str
    kind: str


@dataclass(frozen=True)
class ProbeOutcome:
    """What the cache did with one probe, and whether that was right."""

    probe: Probe
    hit: CacheHit | None
    # Classification: 'true_hit' (served, correct), 'false_hit' (served, WRONG),
    # 'miss' (fell through - correct behavior when no seed truly matches).
    verdict: str


@dataclass(frozen=True)
class ExperimentResult:
    """The probe experiment's payload. `false_hits` is the finding."""

    threshold: float
    outcomes: list[ProbeOutcome]

    @property
    def true_hits(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.verdict == "true_hit"]

    @property
    def false_hits(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.verdict == "false_hit"]

    @property
    def misses(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.verdict == "miss"]


def classify(probe: Probe, hit: CacheHit | None) -> ProbeOutcome:
    """A hit is TRUE when the served answer matches the probe's correct answer,
    FALSE when it doesn't (a stale near-neighbor answer). No hit is a miss."""
    if hit is None:
        return ProbeOutcome(probe=probe, hit=None, verdict="miss")
    verdict = "true_hit" if _answers_match(hit.entry.answer, probe.expected) else "false_hit"
    return ProbeOutcome(probe=probe, hit=hit, verdict=verdict)


def evaluate(
    cache: SemanticCache, probes: list[Probe], probe_vectors: list[list[float]]
) -> ExperimentResult:
    """Drive an already-seeded cache with pre-embedded probes (pure, no I/O).
    `probe_vectors[i]` is the embedding of `probes[i]`."""
    outcomes = [
        classify(probe, cache.lookup(vec)) for probe, vec in zip(probes, probe_vectors, strict=True)
    ]
    return ExperimentResult(threshold=cache.threshold, outcomes=outcomes)


@dataclass(frozen=True)
class ProbeSet:
    """Loaded probe experiment: canonical seeds + the probes that test them."""

    seeds: list[tuple[str, str]] = field(default_factory=list)  # (question, answer)
    probes: list[Probe] = field(default_factory=list)


def load_probe_set(path: Path | str = DEFAULT_PROBE_SET_PATH) -> ProbeSet:
    """Load and validate the probe experiment YAML. Strict, like the golden set:
    a malformed probe file should fail loud at load time."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "seeds" not in raw or "probes" not in raw:
        raise ValueError(f"probe set at {path} must be a mapping with 'seeds' and 'probes'")

    seeds: list[tuple[str, str]] = []
    for i, row in enumerate(raw["seeds"]):
        if not isinstance(row, dict) or "question" not in row or "answer" not in row:
            raise ValueError(f"seed {i} must have 'question' and 'answer': {row!r}")
        seeds.append((row["question"], row["answer"]))

    probes: list[Probe] = []
    for i, row in enumerate(raw["probes"]):
        missing = {"query", "expected", "kind"} - row.keys()
        if missing:
            raise ValueError(f"probe {i} missing {sorted(missing)}: {row!r}")
        if row["kind"] not in {"paraphrase", "trap"}:
            raise ValueError(f"probe {i} kind must be 'paraphrase' or 'trap': {row['kind']!r}")
        probes.append(Probe(query=row["query"], expected=row["expected"], kind=row["kind"]))

    return ProbeSet(seeds=seeds, probes=probes)


def _seeded_cache(
    seeds: list[tuple[str, str]], seed_vectors: list[list[float]], threshold: float
) -> SemanticCache:
    cache = SemanticCache(threshold=threshold)
    for (question, answer), vec in zip(seeds, seed_vectors, strict=True):
        cache.add(question, answer, vec)
    return cache


async def run_probe_experiment(
    probe_set: ProbeSet,
    embedder,
    threshold: float = DEFAULT_THRESHOLD,
) -> ExperimentResult:
    """The I/O seam: embed the seeds + probes through the free local lane, seed
    the cache, then drive it. `embedder` is any object with an async
    `embed_many(list[str]) -> list[list[float]]` (OllamaEmbedder in production,
    a fake in tests). One batched embed call for seeds, one for probes."""
    seed_vectors = await embedder.embed_many([q for q, _ in probe_set.seeds])
    probe_vectors = await embedder.embed_many([p.query for p in probe_set.probes])
    cache = _seeded_cache(probe_set.seeds, seed_vectors, threshold)
    return evaluate(cache, probe_set.probes, probe_vectors)


async def run_threshold_sweep(
    probe_set: ProbeSet, embedder, thresholds: list[float]
) -> list[ExperimentResult]:
    """The finding, as a trade curve: embed ONCE, then re-run the same probes at
    each threshold. A high threshold catches fewer paraphrases (lower hit rate)
    but serves fewer stale answers (fewer false hits); a low one, the reverse.
    Showing both across thresholds is the honest deliverable - the false hit is
    not a fixed number, it is a knob the operator sets."""
    seed_vectors = await embedder.embed_many([q for q, _ in probe_set.seeds])
    probe_vectors = await embedder.embed_many([p.query for p in probe_set.probes])
    results: list[ExperimentResult] = []
    for threshold in thresholds:
        cache = _seeded_cache(probe_set.seeds, seed_vectors, threshold)
        results.append(evaluate(cache, probe_set.probes, probe_vectors))
    return results


def format_sweep_table(results: list[ExperimentResult]) -> str:
    """Fixed-width threshold / hit-rate / false-hit table - the trade curve as
    the writeup would show it: at each threshold, how many paraphrases the cache
    caught vs how many stale answers it served."""
    header = f"{'threshold':>9} {'true_hits':>9} {'false_hits':>10} {'misses':>7} {'n':>4}"
    lines = [header, "-" * len(header)]
    for r in results:
        n = len(r.outcomes)
        lines.append(
            f"{r.threshold:>9.2f} {len(r.true_hits):>9} {len(r.false_hits):>10} "
            f"{len(r.misses):>7} {n:>4}"
        )
    return "\n".join(lines)
