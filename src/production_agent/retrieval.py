"""
Retrieval: reconnect to an already-built Chroma store and fetch relevant
chunks for a query.

WHY THIS IS SEPARATE FROM ingestion.py:
This module never re-embeds documents. It loads the embedding function
(needed so Chroma can embed the *query* the same way documents were
embedded) and reconnects to the persisted vector store on disk. This is
what makes querying cheap: no PDF parsing, no re-chunking, no rewriting
the vector store - just reading from it.
"""
import threading

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config
from logging_setup import get_logger

logger = get_logger(__name__)

# WHY A MODULE-LEVEL CACHE FOR THE VECTOR STORE (THIS FIXES A REAL BUG):
# get_vector_store() previously constructed a NEW HuggingFaceEmbeddings
# instance - reloading the embedding model's weights from disk - on
# EVERY call, and it's called once per retrieve_with_scores() call, i.e.
# once per QUESTION. This is exactly the cost reranking.py's _reranker
# cache was built to avoid for the cross-encoder ("loading a model means
# loading weights from disk/HF cache - real, measurable latency...
# reused across calls") - that reasoning was just never applied here.
# It was masked in CLI testing by a warm OS file cache (fast on a local
# machine that had just run ingestion moments before), but with api.py
# now handling live requests, this meant every single incoming HTTP
# request would pay a full model-reload cost that should only happen
# once, at process startup.
#
# WHY A threading.Lock, NOT JUST "if _vector_store is None":
# A bare None-check has a race condition under concurrent requests - two
# simultaneous first-requests (real once this runs as a multi-worker
# FastAPI service) could both see None and both trigger a load. The lock
# makes the check-and-set atomic, so only one thread ever actually builds
# the client; every other concurrent caller just waits briefly and then
# gets the cached instance.
_vector_store: Chroma | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> Chroma:
    """
    Reconnect to the persisted Chroma collection - once per process, not
    once per call.

    WHY THIS FUNCTION EXISTS:
    Your original code only ever built the vector store via
    Chroma.from_documents(...) inside the same script that also queried it.
    There was no way to query without re-ingesting. This function is the
    "read" counterpart: it opens the existing collection on disk without
    touching or re-embedding any documents.

    WHY collection_metadata IS PASSED HERE TOO:
    ingestion.py sets this when the collection is FIRST created (cosine
    distance instead of Chroma's default L2 - see config.VECTOR_DISTANCE_
    METRIC for why). Passing the same value here is a defensive no-op for
    the normal case (the setting is already baked into the collection on
    disk and this is ignored), but it also means if this function ever
    connects to a collection that doesn't exist yet, it's created with the
    correct metric rather than defaulting to L2.
    """
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:  # re-check: another thread may have built it while this one waited for the lock
                logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
                embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
                _vector_store = Chroma(
                    persist_directory=config.CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                    collection_name=config.COLLECTION_NAME,
                    collection_metadata={"hnsw:space": config.VECTOR_DISTANCE_METRIC},
                )
    # WHY THIS CACHE DOESN'T GO STALE WHEN NEW DOCUMENTS ARE INGESTED
    # LATER IN THE SAME PROCESS (e.g. api.py's /ingest endpoint, called
    # after /query has already warmed this cache): the cached object is a
    # CLIENT connected to the on-disk persist_directory, not a snapshot of
    # its contents - Chroma reads from disk per query. ingestion.py writes
    # to that same directory/collection through its own separate Chroma
    # client. So a query after a later ingestion still sees the new data;
    # only the embedding MODEL and the client WRAPPER are being reused
    # here, not a frozen copy of the collection's contents.
    return _vector_store


def retrieve(query: str, k: int = config.RETRIEVAL_K):
    """Return the top-k most relevant chunks for a query (no scores)."""
    vector_db = get_vector_store()
    logger.info(f"Retrieving top {k} chunk(s) for query: {query!r}")
    results = vector_db.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunk(s)")
    return results


def retrieve_with_scores(query: str, k: int = config.RETRIEVAL_K):
    """
    Return the top-k chunks for a query along with a per-chunk relevance
    score: a list of (document, relevance_score) tuples where
    relevance_score is in [0, 1] - higher means more relevant.

    WHY THIS FUNCTION EXISTS (separate from retrieve() above):
    You asked for a "result score" - a way to see how confident the
    system is that it retrieved the right material. Chroma's
    similarity_search() only returns documents, not how well they matched.

    IMPORTANT HONESTY NOTE - WHAT THIS SCORE IS AND ISN'T:
    This measures how closely the retrieved chunks' embeddings match the
    QUESTION's embedding - i.e. "did we find the right source material".
    It does NOT measure whether the LLM's final answer is factually
    correct - no model can reliably self-grade its own correctness, so
    this pipeline doesn't attempt to fake that. A high retrieval score
    with a low-quality LLM answer is possible (rare, but retrieval being
    right doesn't guarantee generation used it well) - this score is a
    diagnostic signal, not a correctness guarantee.

    WHY similarity_search_with_relevance_scores() NOW, NOT A HAND-ROLLED
    TRANSFORM:
    The previous version computed 1/(1+distance) on raw Chroma distances.
    Chroma's default metric, L2, isn't bounded to a fixed range, so that
    transform made even strong matches read as ~50% - technically
    monotonic (higher always meant better) but not intuitively readable as
    a percentage. Now that the collection is built with cosine distance
    (config.VECTOR_DISTANCE_METRIC, set at collection creation in
    ingestion.py), LangChain's built-in similarity_search_with_relevance_
    scores() uses Chroma's own cosine relevance-score function
    (score = 1 - cosine_distance, clipped to [0, 1]) instead of a
    hand-rolled approximation - a properly bounded, standard score.

    NOTE: this only produces meaningfully calibrated scores if the
    underlying collection was actually created with cosine distance. If
    you're seeing this after upgrading from an older chroma_db/, delete it
    and re-run ingestion - see ingestion.py's embed_and_store() docstring.
    """
    vector_db = get_vector_store()
    logger.info(f"Retrieving top {k} chunk(s) with scores for query: {query!r}")
    scored = vector_db.similarity_search_with_relevance_scores(query, k=k)
    logger.info(f"Retrieved {len(scored)} chunk(s) with scores")
    return scored


def interpret_confidence(score: float) -> str:
    """
    Translate a raw cosine relevance score into a calibrated label, using
    config.CONFIDENCE_BANDS.

    WHY THIS FUNCTION EXISTS:
    A raw score like 0.452 reads as "worse than a coin flip" to anyone
    going by generic percentage intuition - but for EMBEDDING_MODEL_NAME,
    0.42-0.48 has been this project's consistent score for CONFIRMED GOOD
    matches across every test question so far. The number itself wasn't
    wrong; showing it as a bare percentage with no context was misleading.
    This maps the same raw score to a label calibrated against what this
    specific model's "good" actually looks like.

    WHY THE BANDS THEMSELVES LIVE IN config.py, NOT HERE:
    Same reasoning as CHUNK_SIZE, MODEL_PRICING, etc: this is a tunable
    threshold, not fixed logic. As you gather more retrieval results
    (especially confirmed-BAD matches, which no test has produced yet -
    see the caveat on CONFIDENCE_BANDS in config.py) you'll want to
    adjust the thresholds without touching this function's code.
    """
    for threshold, label in config.CONFIDENCE_BANDS:
        if score >= threshold:
            return label
    return config.CONFIDENCE_BANDS[-1][1]  # fallback: lowest band's label