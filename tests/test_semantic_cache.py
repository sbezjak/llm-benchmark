"""Semantic cache: pure vector math + the false-hit finding, mocked embedder, live probe.

The cache core is I/O-free, so its logic (nearest-neighbour, threshold, and the
true-hit / false-hit / miss classification that IS the experiment's payload) is
tested with fabricated vectors - no model. The embedder is respx-mocked like
every other adapter. One `live` test embeds real paraphrases through local
Ollama to confirm paraphrases really do land nearer than traps (skipped when the
embed model isn't pulled).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from llm_benchmark.providers.ollama_embed import OllamaEmbedder, OllamaEmbedError
from llm_benchmark.semantic_cache import (
    Probe,
    ProbeSet,
    SemanticCache,
    classify,
    cosine,
    evaluate,
    format_sweep_table,
    load_probe_set,
    run_probe_experiment,
    run_threshold_sweep,
)

EMBED_URL = "http://localhost:11434/api/embed"


# --- pure vector math -----------------------------------------------------


@pytest.mark.mocked
def test_cosine_identical_orthogonal_and_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # A zero vector has no angle - treated as 'not similar', never a zero-div.
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="length mismatch"):
        cosine([1.0], [1.0, 0.0])


@pytest.mark.mocked
def test_cache_lookup_picks_nearest_above_threshold():
    cache = SemanticCache(threshold=0.9)
    cache.add("A", "answer-A", [1.0, 0.0])
    cache.add("B", "answer-B", [0.0, 1.0])

    # Close to A, far from B -> hits A.
    hit = cache.lookup([0.99, 0.14])
    assert hit is not None and hit.entry.question == "A"
    assert hit.similarity >= 0.9

    # Between the two, below threshold to either -> miss (falls through).
    assert cache.lookup([0.71, 0.71]) is None


# --- the finding: true hit vs false hit vs miss ---------------------------


@pytest.mark.mocked
def test_false_hit_is_a_near_neighbour_with_a_wrong_answer():
    """The whole thesis: a trap whose vector crosses the threshold to a seed
    gets served that seed's stale answer, which is WRONG for the trap."""
    cache = SemanticCache(threshold=0.9)
    cache.add("What is the capital of Slovenia?", "Ljubljana", [1.0, 0.0])

    # Paraphrase: near AND its correct answer is the cached one -> true hit.
    para = Probe("Slovenia's capital?", "Ljubljana", "paraphrase")
    true_hit = classify(para, cache.lookup([0.995, 0.1]))
    assert true_hit.verdict == "true_hit"

    # Trap: near the Slovenia seed but its correct answer is Bratislava.
    trap = Probe("What is the capital of Slovakia?", "Bratislava", "trap")
    false_hit = classify(trap, cache.lookup([0.99, 0.14]))
    assert false_hit.verdict == "false_hit"
    assert false_hit.hit is not None
    assert false_hit.hit.entry.answer == "Ljubljana"  # stale answer served

    # A genuinely distant query is a miss - correct behaviour, no stale serve.
    miss = classify(Probe("What is 2+2?", "4", "trap"), cache.lookup([0.0, 1.0]))
    assert miss.verdict == "miss"


@pytest.mark.mocked
def test_answer_match_is_loose_for_short_factual_answers():
    cache = SemanticCache(threshold=0.5)
    cache.add("Q", "Ljubljana", [1.0, 0.0])
    # 'Ljubljana.' and 'The capital is Ljubljana' both count as the same answer.
    out = classify(Probe("q", "The capital is Ljubljana", "paraphrase"), cache.lookup([1.0, 0.0]))
    assert out.verdict == "true_hit"


@pytest.mark.mocked
def test_evaluate_aggregates_outcomes():
    cache = SemanticCache(threshold=0.9)
    cache.add("seed", "Ljubljana", [1.0, 0.0])
    probes = [
        Probe("para", "Ljubljana", "paraphrase"),
        Probe("trap", "Bratislava", "trap"),
        Probe("far", "4", "trap"),
    ]
    vectors = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]]
    result = evaluate(cache, probes, vectors)
    assert len(result.true_hits) == 1
    assert len(result.false_hits) == 1
    assert len(result.misses) == 1


# --- probe set loading ----------------------------------------------------


@pytest.mark.mocked
def test_load_probe_set_reads_and_validates_bundled_file():
    probe_set = load_probe_set()
    assert len(probe_set.seeds) >= 3
    assert any(p.kind == "trap" for p in probe_set.probes)
    assert any(p.kind == "paraphrase" for p in probe_set.probes)


@pytest.mark.mocked
def test_load_probe_set_rejects_bad_kind(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "seeds:\n  - {question: Q, answer: A}\nprobes:\n  - {query: q, expected: A, kind: bogus}\n"
    )
    with pytest.raises(ValueError, match="kind must be"):
        load_probe_set(bad)


# --- embedder adapter + the I/O seam --------------------------------------


class _FakeEmbedder:
    """Deterministic embed_many keyed by exact string - no HTTP, for the seam."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.table[t] for t in texts]


@pytest.mark.mocked
async def test_run_probe_experiment_seeds_then_drives_cache():
    probe_set = ProbeSet(
        seeds=[("What is the capital of Slovenia?", "Ljubljana")],
        probes=[
            Probe("Slovenia's capital?", "Ljubljana", "paraphrase"),
            Probe("What is the capital of Slovakia?", "Bratislava", "trap"),
        ],
    )
    embedder = _FakeEmbedder(
        {
            "What is the capital of Slovenia?": [1.0, 0.0],
            "Slovenia's capital?": [0.995, 0.1],
            "What is the capital of Slovakia?": [0.99, 0.14],
        }
    )
    result = await run_probe_experiment(probe_set, embedder, threshold=0.9)
    assert len(result.true_hits) == 1
    assert len(result.false_hits) == 1  # the trap crossed the threshold


@pytest.mark.mocked
@respx.mock
async def test_ollama_embedder_parses_and_orders():
    respx.post(EMBED_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    embedder = OllamaEmbedder(model="nomic-embed-text")
    vecs = await embedder.embed_many(["a", "b"])
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]

    respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json={"embeddings": [[0.5, 0.6]]}))
    assert await embedder.embed("solo") == [0.5, 0.6]


@pytest.mark.mocked
async def test_ollama_embedder_empty_is_noop():
    # No HTTP call for an empty batch.
    assert await OllamaEmbedder().embed_many([]) == []


@pytest.mark.mocked
@respx.mock
async def test_ollama_embedder_count_mismatch_raises():
    respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]}))
    with pytest.raises(OllamaEmbedError, match="expected 2 embeddings"):
        await OllamaEmbedder().embed_many(["a", "b"])


@pytest.mark.mocked
@respx.mock
async def test_ollama_embedder_http_error_surfaces_detail():
    respx.post(EMBED_URL).mock(
        return_value=httpx.Response(404, json={"error": "model 'nope' not found"})
    )
    with pytest.raises(OllamaEmbedError, match="not found"):
        await OllamaEmbedder(model="nope").embed_many(["a"])


@pytest.mark.mocked
async def test_threshold_sweep_shows_the_trade():
    """Raising the threshold trades true hits down and false hits down together -
    the whole point of reporting the curve, not one number."""
    probe_set = ProbeSet(
        seeds=[("Slovenia?", "Ljubljana")],
        probes=[
            Probe("Slovenia para", "Ljubljana", "paraphrase"),
            Probe("Slovakia trap", "Bratislava", "trap"),
        ],
    )
    embedder = _FakeEmbedder(
        {
            "Slovenia?": [1.0, 0.0],
            "Slovenia para": [0.98, 0.199],  # cosine ~0.98 to seed
            "Slovakia trap": [0.96, 0.28],  # cosine ~0.96 to seed
        }
    )
    # 0.90 catches both (true hit + FALSE hit); 0.99 catches neither.
    results = await run_threshold_sweep(
        embedder=embedder, probe_set=probe_set, thresholds=[0.90, 0.99]
    )
    low, high = results
    assert len(low.true_hits) == 1 and len(low.false_hits) == 1
    assert len(high.true_hits) == 0 and len(high.false_hits) == 0 and len(high.misses) == 2
    # The table renders one row per threshold plus a header + rule.
    table = format_sweep_table(results)
    assert "threshold" in table and table.count("\n") == 3


# --- live: real local embeddings (free, slow; needs the embed model pulled) ---


@pytest.mark.live
@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING (evidence/semantic-cache-false-hits.md): nomic-embed-text cannot "
        "separate the Slovakia trap from a real Slovenia paraphrase - it scores the "
        "trap at cosine 1.000, at or above the genuine paraphrase (~0.974). So a "
        "similarity threshold set to catch paraphrases also catches the trap, which "
        "IS the false-hit mechanism. If this XPASSes the embed model was upgraded and "
        "now distinguishes them - re-run the threshold sweep and update the finding."
    ),
)
async def test_paraphrases_land_nearer_than_traps_live():
    """The false-hit premise, encoded as a strict-xfail contract: with REAL
    embeddings the Slovakia trap is NOT farther from the Slovenia seed than a
    genuine paraphrase, so no single threshold separates them. Skips when the
    embed model isn't pulled (the xfail only applies once it actually runs)."""
    embedder = OllamaEmbedder(model="nomic-embed-text")
    seed = "What is the capital of Slovenia?"
    para = "Which city is the capital of Slovenia?"
    trap = "What is the capital of Slovakia?"
    try:
        vecs = await embedder.embed_many([seed, para, trap])
    except OllamaEmbedError as e:
        pytest.skip(
            f"local Ollama embed model unavailable ({e}); run: ollama pull nomic-embed-text"
        )

    seed_v, para_v, trap_v = vecs
    sim_para = cosine(seed_v, para_v)
    sim_trap = cosine(seed_v, trap_v)
    assert sim_para > sim_trap, f"paraphrase {sim_para:.3f} not nearer than trap {sim_trap:.3f}"
