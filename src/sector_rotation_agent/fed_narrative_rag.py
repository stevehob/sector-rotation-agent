"""
fed_narrative_rag.py

Second RAG corpus (spec §10.1 / §11 deferred item): qualitative Fed narrative text
-- Beige Book reports and FOMC meeting minutes -- to complement the NUMERIC analog
store in historical_analogs.py.

WHY A SEPARATE MODULE
---------------------
historical_analogs.py stores macro *vectors* (8 normalized indicators) and hands them
to ChromaDB directly -- no text embedding. Its own header calls out the boundary:
"sentence-transformers would only matter if you later store TEXT, like analyst
reports." This is that case. Fed narrative is unstructured prose, so it needs a real
text embedding model and its own collection. Keeping it here -- separate store
collection, separate retrieval function -- means the numeric path is untouched.

WHAT IT ADDS
------------
A retrieval tool the synthesis/macro reasoning can use for QUALITATIVE evidence:
"what was the Fed actually saying about labor markets the last time the curve looked
like this?" The numeric analogs say *what happened to sectors*; this says *what
policymakers were thinking*. Plus a freshness check (spec §9 guardrail #2, the same
family as the per-series ceilings in audit.py) so a stale corpus -- a Beige Book or
minutes release that never got ingested -- is caught rather than silently trusted.

  seed_corpus()           fetch -> chunk -> embed -> store (run once / on a schedule)
  find_fed_narrative()    embed the query, retrieve the most similar chunks
  check_corpus_freshness() is the newest ingested document recent enough to trust?

SYMMETRY (the lesson from historical_analogs)
---------------------------------------------
Seed and query MUST embed with the SAME model. A text embedding from model A is not
comparable to one from model B, so cosine distance becomes noise -- exactly the
failure the numeric store guards against with shared normalization. Pin EMBED_MODEL_NAME
and use it on both sides; the `embed` injection seam below exists so tests can pass a
fake instead of loading the model.

INTEGRATION POINTS (not wired here -- your call when you implement)
-------------------------------------------------------------------
  * Retrieval: have the macro or synthesis agent call find_fed_narrative() and cite the
    returned chunks via synthesize.build_sources (tool tag e.g. "FedNarrative").
  * Freshness: ResultAuditor (audit.py) could call check_corpus_freshness() and translate
    a flagged result into an AuditFlag, the same way it wraps check_statistical_anomaly.
  * Caps: a retrieval still counts against the coordinator's 20-tool-call budget
    (audit_log.record_tool_call), so wire it through the loop, not around it.

Dependencies: `chromadb` and `sentence-transformers` are already project deps.
"""

from __future__ import annotations

import os
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum

import chromadb as cdb

import sector_rotation_agent.constants as const
from  sector_rotation_agent.config import AppConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Local configuration.
# These are module-local for now (the same way audit.py's freshness ceiling started
# life before moving to constants.py). Promote to constants.py once a second caller
# needs them.
# --------------------------------------------------------------------------- #

# A SEPARATE collection in the SAME persistent Chroma dir as the numeric store
# (const.STORE_PATH). Text embeddings and macro vectors must never share a collection
# -- different dimensionality, different meaning of "distance".
FED_NARRATIVE_COLLECTION = "fed_narrative"

# Freshness ceiling for the corpus (spec §9 guardrail #2), in days. Both sources publish
# ~8x/year: the Beige Book lands ~2 weeks before each FOMC meeting, the minutes ~3 weeks
# after -- so on any given day the newest document is normally well under ~7 weeks old.
# ~60 days clears the normal cadence while still catching a release that was missed or
# never ingested. Tunable, like the per-series ceilings in audit.py.
CORPUS_MAX_AGE_DAYS = 60

# Fed data downloaded locally for now.  It's here:
_local_file_path = const._PROJECT_ROOT / "data/fed_source_data" 
_local_list = "local_fed_files.json"
_file_list: dict[str, dict]

class FedSource(str, Enum):
    """The three narrative Federal Reserve sources this corpus ingests."""
    BEIGE_BOOK = "beige_book"           # Federal Reserve "Summary of Commentary on Current Economic Conditions"
    FOMC_MINUTES = "fomc_minutes"       # FOMC meeting minutes
    MONETARY_POLICY = "monetary_policy" # Federal Reserve Monetry Policy Press Releases

# Chunking. Fed documents are long; retrieval works on passages, not whole reports.
# Character-based is simplest to start; a token-aware splitter is a later refinement.
CHUNK_PROFILES = {
    FedSource.BEIGE_BOOK: {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    FedSource.MONETARY_POLICY:{
        "chunk_size": 250,
        "chunk_overlap": 35
    },
    FedSource.FOMC_MINUTES:{
        "chunk_size": 250,
        "chunk_overlap": 35
    },
}

@dataclass(frozen=True)
class FedDocument:
    """One fetched narrative document, before chunking.

    `date` is the PUBLICATION date (ISO YYYY-MM-DD) and is the freshness anchor -- the
    point-in-time guard and check_corpus_freshness both key off it, so it must be the
    real release date, not the meeting/reporting period it covers.
    """
    source: FedSource
    date: str
    title: str
    url: str
    name: str
    text: str


# Helper to get the local files list
def _get_file_list():
    global _file_list

    # read in the pre-determined list of fed files
    try:
        with open(_local_file_path / _local_list, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.critical(f"File not found try to read {_local_file_path / _local_list}")
        raise RuntimeError(f"_get_file_list failed: file not found: {_local_file_path / _local_list}")
    except json.JSONDecodeError:
        logger.exception(f"File {_local_list} is not valid json")
        raise RuntimeError(f"_get_file_list faild; {_local_list} is not valid json")
    if not isinstance(data, dict):
        raise RuntimeError(f"_get_file_list faild; {_local_list} is json, but not proper structure")
    
    _file_list = data

    return


# --------------------------------------------------------------------------- #
# Collection accessor (boilerplate -- mirrors historical_analogs._get_collection)
# --------------------------------------------------------------------------- #
# Own module-level globals so this collection is independent of the numeric store's.
# Thread-safe + path-aware for the same reasons documented there (parallel callers must
# not each spin up a PersistentClient; tests may rebind the store path).
_collection = None
_collection_path: str | None = None
_collection_lock = threading.Lock()


def _get_collection():
    """Open (or create) the persistent ChromaDB collection for Fed narrative chunks.

    Cosine space, so collection.query() returns cosine DISTANCE; convert to a 0..1
    similarity in find_fed_narrative (clamped, exactly as the numeric store does).
    """
    global _collection, _collection_path
    path = str(const.STORE_PATH)
    if _collection is None or _collection_path != path:
        with _collection_lock:
            if _collection is None or _collection_path != path:
                client = cdb.PersistentClient(path=path)
                _collection = client.get_or_create_collection(
                    name=FED_NARRATIVE_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
                _collection_path = path
    return _collection


# --------------------------------------------------------------------------- #
# Embedding + chunking helpers (the text-specific work the numeric store never needed)
# --------------------------------------------------------------------------- #

def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text chunks into vectors with the pinned sentence-transformers
    model (EMBED_MODEL_NAME).

    Contract
    --------
    - Input: N strings. Output: N vectors (list[list[float]]), order preserved.
    - The SAME model must be used at seed and query time (see SYMMETRY in the header).
    - Load the model lazily (import sentence_transformers inside the body, cache the
      loaded model at module scope) so importing this module stays cheap and the suite
      doesn't pull the model unless an embedding is actually requested -- the same lazy
      pattern synthesize.report_to_pdf uses for markdown-pdf.
    """
    from sentence_transformers import SentenceTransformer

    # Note: not passing value for key here, becuase config took care of setting env HF_TOKEN value (which the API uses by default)
    model = SentenceTransformer(AppConfig.embedding_model)

    return model.encode(texts, normalize_embeddings=True).tolist()


def _chunk_document(doc: FedDocument) -> list[dict]:
    """Split one document into passages via Docling's structure-aware HybridChunker.

    Returns ONE dict per chunk, each carrying the document's provenance so seed_corpus
    can build Chroma metadata without needing the FedDocument again.
    """
    from langchain_docling.loader import DoclingLoader
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    # Cap at the embedder's max sequence length, or chunk tails get silently truncated
    # at embed time (all-MiniLM-L6-v2 == 256 tokens, so your 1000 for Beige Book loses
    # most of every chunk).
    profile = CHUNK_PROFILES[doc.source]
    max_tokens = min(int(profile["chunk_size"]), AppConfig.embedding_model_max_tokens)

    key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_KEY")
    if not key:
        logger.warning("No Hugging Face token (HF_TOKEN) set; tokenizer load may be rate-limited")
    hf_tokenizer = HuggingFaceTokenizer.from_pretrained(
                                model_name="sentence-transformers/" + AppConfig.embedding_model,
                                max_tokens=max_tokens,
                                token=key)

    custom_chunker = HybridChunker(
        tokenizer = hf_tokenizer
    )

    loader = DoclingLoader(
        file_path=str(_local_file_path / doc.name),
        chunker = custom_chunker,
    )

    # loader.load() returns a list of LangChain Document objects (one per chunk)
    #  each with the real text on .page_content
    chunks: list[dict] = []
    for ord, lc_doc in enumerate(loader.load()):
        chunks.append(
            {
                "id":f"{doc.source.value}:{doc.name}:{doc.date}:{ord}",
                "text": lc_doc.page_content,  # the chunk body (string)
                "ordinal": ord,
                "source": doc.source.value,
                "date": doc.date,
                "title": doc.title,
                "url": doc.url,
            }
        )

    logger.debug("Chunked %s (%s) into %d passage(s)", doc.name, doc.source.value, len(chunks))
    return chunks


# --------------------------------------------------------------------------- #
# Fetchers (network) -- one per source
# --------------------------------------------------------------------------- #

def fetch_beige_book(start: str, end: str | None = None) -> list[FedDocument]:
    """Fetch Beige Book reports published in [start, end] as FedDocuments.

    Source: federalreserve.gov publishes the Beige Book ~8x/year. There's no clean JSON
    API, so this is a scrape/parse of the published HTML (or PDF) into plain text. Set
    each FedDocument.date to the RELEASE date and keep the source URL for citation.

    Keep network/parsing concerns isolated here so seed_corpus stays a pure
    chunk->embed->store pipeline over whatever these return.
    """
    #TODO: Future - scrape the web site.
    # For now, I've downloaded a fixed list
    # start and end are no-op at this point
    logger.info(f"fetching list of Beige Books within {start}:{end}.")
    logger.warning("Beige books aren't fetched - fixed list.")

    bb_files: list[FedDocument] = []
    # get local files from the json list
    if _file_list:
        for bb in _file_list[FedSource.BEIGE_BOOK]:
            doc = FedDocument(
                source=FedSource.BEIGE_BOOK,
                date=bb["date"],
                title= bb["title"],
                url="",
                name=bb["name"],
                text=""
            )
            bb_files.append(doc)
    else:
        logger.error("File list is not populated")
    
    return bb_files


def fetch_fomc_minutes(start: str, end: str | None = None) -> list[FedDocument]:
    """Fetch FOMC meeting minutes published in [start, end] as FedDocuments.

    Source: federalreserve.gov publishes minutes ~3 weeks after each of the 8 annual
    meetings. As with the Beige Book, parse to plain text; FedDocument.date is the
    minutes' RELEASE date (not the meeting date), since freshness is about what was
    available to read as of a given day.
    """
    logger.info(f"fetching list of FOMC Meeting Minutes within {start}:{end}.")
    logger.warning("FOMC Meeting Minutes aren't fetched - fixed list.")

    fomc_files: list[FedDocument] = []
    # get local files from the json list
    if _file_list:
        for bb in _file_list[FedSource.FOMC_MINUTES]:
            doc = FedDocument(
                source=FedSource.FOMC_MINUTES,
                date=bb["date"],
                title= bb["title"],
                url="",
                name=bb["name"],
                text=""
            )
            fomc_files.append(doc)
    else:
        logger.error("File list is not populated")
    
    return fomc_files


def fetch_monetary_policy(start: str, end: str | None = None) -> list[FedDocument]:
    """Fetch Fed Monetary Policy Statements published in [start, end] as FedDocuments.

    Source: federalreserve.gov
    """
    logger.info(f"fetching list of Monetary Policy Releases within {start}:{end}.")
    logger.warning("Monetary Policy Releases aren't fetched - fixed list.")

    mon_files: list[FedDocument] = []
    # get local files from the json list
    if _file_list:
        for bb in _file_list[FedSource.MONETARY_POLICY]:
            doc = FedDocument(
                source=FedSource.MONETARY_POLICY,
                date=bb["date"],
                title= bb["title"],
                url="",
                name=bb["name"],
                text=""
            )
            mon_files.append(doc)
    else:
        logger.error("File list is not populated")
    
    return mon_files


def build_corpus(start: str = const.HISTORY_SEED_START, end: str | None = None) -> list[FedDocument]:
    """Assemble the full document set to ingest -- the narrative analogue of
    historical_analogs.build_seed_history.

    Concatenates the two fetchers over the window. Default `start` matches the numeric
    store's seed start so the two corpora cover the same era, though Fed text only needs
    to go back as far as you intend to retrieve against.
    """
    _get_file_list()
    bb = fetch_beige_book(start, end)
    fomc = fetch_fomc_minutes(start, end)
    mon = fetch_monetary_policy(start, end)

    docs = bb + fomc + mon
    logger.info("Built Fed corpus document set: %d total (%d beige book, %d FOMC, %d monetary policy)",
                len(docs), len(bb), len(fomc), len(mon))
    return docs


# --------------------------------------------------------------------------- #
# Seed side (run once / on a schedule)
# --------------------------------------------------------------------------- #

def seed_corpus(
    documents: list[FedDocument],
    *,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> None:
    """Chunk, embed, and upsert documents into the Fed-narrative collection.

    Parameters
    ----------
    documents
        Output of build_corpus (or a hand-built list in tests).
    embed
        Injection seam for the embedder (defaults to _embed). Tests pass a fake that
        returns canned vectors so seeding needs no model download -- the same DI style
        as generate_hypotheses' call_model and classify_regime_tot's collaborators.
        
    Uses upsert (idempotent) so re-running to pick up new releases doesn't duplicate.
        Note: that embedding model must be the same for upsert to work (change the model requires re-seeding)
    """

    embed = embed or _embed
    collection = _get_collection()
    
    logger.info("Seeding Fed-narrative corpus: chunking %d document(s)", len(documents))
    chunks = [c for doc in documents for c in _chunk_document(doc)]   # carry doc meta onto each
    if not chunks:
        logger.warning("No chunks produced; nothing to upsert")
        return

    # if we do have chunk data, upsert to the Chroma DB
    vectors = embed([c["text"] for c in chunks])
    
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,                         #type: ignore
        documents=[c["text"] for c in chunks],      # store the text so retrieval can return it
        metadatas=[
            { 
                "source": c["source"],
                "date":c["date"],
                "title":c["title"],
                "url":c["url"],
                "ordinal":c["ordinal"],
            }
            for c in chunks
        ],
    )
    logger.info(f"upserted {len(chunks)} chunks from {len(documents)} documents")

# --------------------------------------------------------------------------- #
# Query side (the retrieval tool)
# --------------------------------------------------------------------------- #

def find_fed_narrative(
    query_text: str,
    n: int,
    *,
    source_filter: FedSource | None = None,
    as_of: str | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict]:
    """Retrieve the `n` Fed-narrative chunks most similar to `query_text`.

    Parameters
    ----------
    query_text
        The natural-language query to match against (e.g. a regime statement, or a
        rendered macro snapshot).
    n
        Max chunks to return (may come back shorter under a narrow filter).
    source_filter
        Restrict to one source (Beige Book or minutes), via Chroma's `where`.
    as_of
        Point-in-time guard: exclude any chunk whose document was published AFTER
        `as_of`, so a backtest/replay never reads text that wasn't available yet -- the
        same no-lookahead discipline the macro agent applies to FRED data. (Store the
        date as an int ordinal like 20260430 in metadata to filter with a Chroma
        `$lte`, or post-filter in Python after the query.)
    embed
        Embedder injection seam (defaults to _embed); tests pass a fake.

    Returns
    -------
    list[dict]
        Up to `n` chunks, nearest first, each shaped:

            {
              "id":         "fomc_minutes:2026-04-30#3",
              "source":     "fomc_minutes",
              "date":       "2026-04-30",
              "title":      "Minutes of the ... meeting",
              "url":        "https://www.federalreserve.gov/...",
              "text":       "<the matched passage>",
              "similarity": 0.83,        # clamp(1 - cosine_distance) to [0, 1]
            }
    """
    if not query_text:
        logger.debug("No query text provided")
        return []
    
    collection = _get_collection()

    embed = embed or _embed
    query_vector = embed([query_text])

    where_filter = {
        "source": source_filter.value } if source_filter else None

    # When as_of is set, over-fetch so the point-in-time filter below can drop
    # too-recent chunks without starving the result down past n. (If the corpus grows,
    # store an int date in metadata and push this into the Chroma `where` as a $lte.)
    fetch_n = n
    if as_of is not None:
        available = collection.count()
        fetch_n = min(max(n * 5, n), available) if available else n

    result = collection.query(
        query_embeddings=query_vector,      # type: ignore  -- list[list[float]], one query
        n_results=fetch_n,
        where=where_filter,                 # type: ignore  -- typed Chroma param
    )

    metadatas = result["metadatas"]
    distances = result["distances"]
    documents = result["documents"]
    if metadatas is None or distances is None or documents is None:
        logger.debug("find_fed_narrative: query returned no documents/metadatas/distances")
        return []

    # query() is batched; we sent ONE query, so each field is a single-element list and
    # everything we want lives at index 0, already ordered nearest-first.
    ids = result["ids"][0]
    outputs: list[dict] = []
    for cid, text, meta, dist in zip(ids, documents[0], metadatas[0], distances[0]):
        # no-lookahead guard: skip any chunk whose document postdates as_of (ISO date
        # strings sort chronologically, so a plain string compare is enough)
        if as_of is not None and str(meta["date"]) > as_of:
            continue
        outputs.append({
            "id": cid,
            "source": meta["source"],
            "date": meta["date"],
            "title": meta["title"],
            "url": meta["url"],
            "text": text,
            "similarity": max(0.0, min(1.0, 1.0 - dist)),   # cosine distance -> [0,1] similarity
        })
        if len(outputs) >= n:
            break   # nearest-first, so the first n survivors are the n best

    logger.info("Fed-narrative retrieval: %d/%d chunk(s) (source=%s, as_of=%s)",
                len(outputs), n, source_filter.value if source_filter else "any", as_of or "now")
    return outputs


# --------------------------------------------------------------------------- #
# Freshness check (spec §9 guardrail #2 -- corpus edition)
# --------------------------------------------------------------------------- #

def check_corpus_freshness(
    documents: list[dict] | None = None,
    *,
    as_of: str | None = None,
    max_age_days: int = CORPUS_MAX_AGE_DAYS,
    source_filter: FedSource | None = None,
) -> dict:
    """Flag the corpus as stale if its most recent document predates `as_of` by more
    than `max_age_days` -- the guardrail-#2 freshness idea applied to a document set
    rather than a numeric series.

    Parameters
    ----------
    documents
        Optional: the retrieved chunks to judge (each with a "date"). If None, inspect
        the whole stored collection's newest metadata date instead -- "is the corpus as
        a whole up to date?" vs. "are these particular hits recent?".
    as_of
        Reference "today" (ISO date); None -> system clock. Keyword-only and explicit so
        tests are deterministic, matching check_statistical_anomaly.
    max_age_days
        Freshness ceiling; defaults to CORPUS_MAX_AGE_DAYS.
    source_filter
        Optionally judge freshness for one source only (Beige Book and minutes can fall
        out of date independently).

    Returns
    -------
    dict
        An audit-compatible flag dict, deliberately shaped like
        check_statistical_anomaly's output so ResultAuditor can translate it into an
        AuditFlag with no special-casing:

            {
              "source_id": "fed_narrative" | "<source>",
              "flagged":   bool,
              "reasons":   [ "newest fed_narrative doc is 84 days old (> 60)" ],
              "checks": {
                  "freshness": {"flagged": bool, "newest_date": str|None,
                                "age_days": int|None, "max_age_days": int},
              },
            }

        An empty corpus (no dates) reports flagged=True with newest_date=None: nothing
        ingested is itself a freshness failure, not a silent pass.

    Implementation sketch:
        # gather candidate dates from `documents` or the collection metadata,
        # filtered by source_filter; newest = max(dates) or None
        # age = (as_of_d - newest).days; flagged = newest is None or age > max_age_days
    """
    as_of_d = date.fromisoformat(as_of) if as_of else date.today()
    source_id = source_filter.value if source_filter else "fed_narrative"

    # Candidate publication dates: from the passed-in chunks, or -- when documents is
    # None -- from the whole stored collection's metadata ("is the corpus as a whole
    # current?"). Filtered to one source when asked.
    if documents is None:
        try:
            stored = _get_collection().get(include=["metadatas"])
            records: list[dict] = list(stored.get("metadatas") or [])  #type: ignore
        except Exception:
            logger.exception("check_corpus_freshness: failed to read the collection")
            records = []
    else:
        records = documents

    dates = [
        r["date"]
        for r in records
        if r.get("date") and (source_filter is None or r.get("source") == source_filter.value)
    ]
    newest = max(dates) if dates else None      # ISO date strings sort chronologically

    if newest is None:
        # Nothing ingested (or nothing for this source) is a freshness failure, not a
        # silent pass.
        age_days = None
        flagged = True
        reasons = [f"no {source_id} documents found in the corpus"]
    else:
        age_days = (as_of_d - date.fromisoformat(newest)).days
        flagged = age_days > max_age_days
        reasons = (
            [f"newest {source_id} doc is {age_days} days old (> {max_age_days})"]
            if flagged
            else []
        )

    logger.debug("Corpus freshness[%s]: newest=%s, age_days=%s, flagged=%s",
                 source_id, newest, age_days, flagged)
    return {
        "source_id": source_id,
        "flagged": flagged,
        "reasons": reasons,
        "checks": {
            "freshness": {
                "flagged": flagged,
                "newest_date": newest,
                "age_days": age_days,
                "max_age_days": max_age_days,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Run-once seeder (mirrors historical_analogs.__main__)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    """
    print("Build/refresh the Fed-narrative RAG corpus (Beige Book + FOMC minutes)?")
    print("This fetches from federalreserve.gov and embeds the text -- it may take a while.")
    """
    print("This function will process local federalreserve.gov files as specified in local_fed_files.json")
    choice = input("This will take a while. Type 'yes' to execute: ")
    if choice.lower() == "yes":
        seed_corpus(build_corpus(start="2025-01-01", end="2026-06-15"))
    