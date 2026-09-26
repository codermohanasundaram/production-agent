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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config
from logging_setup import get_logger

logger = get_logger(__name__)


def get_vector_store() -> Chroma:
    """
    Reconnect to the persisted Chroma collection.

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
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": config.VECTOR_DISTANCE_METRIC},
    )


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