"""
test_fed_narrative_rag.py

Contract tests for the Fed-narrative RAG corpus
(src/sector_rotation_agent/fed_narrative_rag.py).

Status:
  * find_fed_narrative is IMPLEMENTED, so the retrieval tests below run for real.
  * check_corpus_freshness is still a STUB, so only those four tests are marked
    xfail(strict=False) -- they flip to XPASS the moment you implement the body, your
    signal to drop the `freshness_stub` marker. Same pattern as test_audit.py.

Design notes that keep these offline:
  * `embed` is an injected seam, so querying uses a fake 2-D embedder -- no model
    download, no network.
  * The retrieval tests seed the Chroma collection DIRECTLY (see _seed_chunks_directly)
    rather than through seed_corpus, because seed_corpus -> _chunk_document loads real
    PDFs via Docling. Seeding canned chunks isolates the query path; the seed/chunk
    (Docling) path is better exercised separately with a real fixture PDF under
    TEST_MODE=Integration.
  * `temp_store` repoints the collection at a throwaway Chroma dir, so a round-trip
    never touches the real data/ store.
  * Freshness is pure date arithmetic; an explicit `as_of` keeps it deterministic.
"""
from __future__ import annotations

import pytest

import sector_rotation_agent.constants as const
import sector_rotation_agent.fed_narrative_rag as fnr

AS_OF = "2026-06-01"


# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #
def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 2-D cosine embedder: 'inflation'-flavored text points one way,
    everything else the other, so nearest-neighbor results are predictable without a
    real model."""
    return [[1.0, 0.0] if "inflation" in t.lower() else [0.0, 1.0] for t in texts]


# Pre-chunked records in the exact shape seed_corpus upserts (one per chunk), so we can
# write them straight to the collection and skip the Docling/PDF chunking path.
CHUNKS = [
    {
        "id": "fomc_minutes:2026-04-30#0",
        "text": "Participants discussed inflation risks and the path of policy.",
        "ordinal": 0,
        "source": fnr.FedSource.FOMC_MINUTES.value,
        "date": "2026-04-30",
        "title": "Minutes of the April 2026 FOMC meeting",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260430.htm",
    },
    {
        "id": "beige_book:2026-05-21#0",
        "text": "Labor markets cooled modestly across most Districts.",
        "ordinal": 0,
        "source": fnr.FedSource.BEIGE_BOOK.value,
        "date": "2026-05-21",
        "title": "Beige Book — May 2026",
        "url": "https://www.federalreserve.gov/monetarypolicy/beigebook202605.htm",
    },
]


def _seed_chunks_directly(chunks: list[dict], embed) -> None:
    """Write canned chunk dicts straight to the collection, bypassing seed_corpus /
    _chunk_document (which load real PDFs via Docling). Isolates the QUERY path so these
    tests exercise find_fed_narrative with no document-parsing dependency. The upsert
    mirrors seed_corpus exactly, so the stored shape is identical to production."""
    collection = fnr._get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {k: c[k] for k in ("source", "date", "title", "url", "ordinal")}
            for c in chunks
        ],
    )


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """Point the Fed-narrative collection at a throwaway Chroma dir so seeding/querying
    never touches the real data/ store. Relies on _get_collection rebinding when the
    path changes (resetting the module's cached globals forces that)."""
    monkeypatch.setattr(const, "STORE_PATH", tmp_path / "chroma")
    fnr._collection = None
    fnr._collection_path = None
    yield
    fnr._collection = None
    fnr._collection_path = None


# --------------------------------------------------------------------------- #
# find_fed_narrative  (retrieval contract -- implemented, runs for real)
# --------------------------------------------------------------------------- #
def test_find_returns_nearest_chunk_with_provenance(temp_store):
    """A query retrieves the most similar chunk, shaped with full provenance and a
    clamped [0, 1] similarity -- the 'inflation' query lands on the minutes chunk."""
    _seed_chunks_directly(CHUNKS, fake_embed)
    out = fnr.find_fed_narrative("what is the inflation outlook?", n=1, embed=fake_embed)

    assert len(out) == 1
    hit = out[0]
    assert {"id", "source", "date", "title", "url", "text", "similarity"} <= set(hit)
    assert 0.0 <= hit["similarity"] <= 1.0
    assert hit["source"] == fnr.FedSource.FOMC_MINUTES.value


def test_source_filter_restricts_results(temp_store):
    """source_filter keeps only chunks from the requested source."""
    _seed_chunks_directly(CHUNKS, fake_embed)
    out = fnr.find_fed_narrative(
        "labor markets", n=5, source_filter=fnr.FedSource.BEIGE_BOOK, embed=fake_embed
    )
    assert out and all(h["source"] == fnr.FedSource.BEIGE_BOOK.value for h in out)


def test_as_of_excludes_future_documents(temp_store):
    """Point-in-time guard: a chunk whose document was published after `as_of` is never
    returned, so a replay can't read text that didn't exist yet."""
    _seed_chunks_directly(CHUNKS, fake_embed)
    # 2026-05-01 is after the minutes (04-30) but before the Beige Book (05-21)
    out = fnr.find_fed_narrative("inflation or labor", n=5, as_of="2026-05-01", embed=fake_embed)
    assert out and all(h["date"] <= "2026-05-01" for h in out)


def test_empty_collection_returns_empty(temp_store):
    """No matching chunks (here: filtering to a source that was never seeded) yields an
    empty list, not an error."""
    _seed_chunks_directly(CHUNKS, fake_embed)
    out = fnr.find_fed_narrative(
        "anything", n=5, source_filter=fnr.FedSource.MONETARY_POLICY, embed=fake_embed
    )
    assert out == []


# --------------------------------------------------------------------------- #
# check_corpus_freshness  (guardrail #2, corpus edition -- still a stub)
# --------------------------------------------------------------------------- #
def _docs(*dates: str, source: str = "fomc_minutes") -> list[dict]:
    return [{"source": source, "date": d} for d in dates]


def test_fresh_corpus_not_flagged():
    """A recent newest-document is within the ceiling and trips nothing."""
    out = fnr.check_corpus_freshness(_docs("2026-05-20"), as_of=AS_OF)  # ~12 days old
    assert out["flagged"] is False
    assert out["checks"]["freshness"]["flagged"] is False


def test_stale_corpus_flagged():
    """A newest-document older than the ceiling (default 60d) is flagged."""
    out = fnr.check_corpus_freshness(_docs("2026-02-01"), as_of=AS_OF)  # ~120 days old
    assert out["flagged"] is True
    assert out["checks"]["freshness"]["flagged"] is True


def test_empty_corpus_is_flagged():
    """Nothing ingested is itself a freshness failure, not a silent pass."""
    out = fnr.check_corpus_freshness([], as_of=AS_OF)
    assert out["flagged"] is True
    assert out["checks"]["freshness"]["newest_date"] is None


def test_freshness_uses_the_newest_document():
    """Freshness keys off the NEWEST document; old docs alongside a recent one are fine."""
    out = fnr.check_corpus_freshness(_docs("2026-01-01", "2026-05-25"), as_of=AS_OF)
    assert out["flagged"] is False
