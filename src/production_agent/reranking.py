"""
Cross-encoder reranking: a second, more accurate relevance pass over the
candidate chunks cosine similarity already narrowed down.

WHY A SECOND STAGE ON TOP OF COSINE SIMILARITY (not a replacement for it):
Cosine similarity (retrieval.py) compares PRE-COMPUTED embeddings - the
question's embedding was computed independently of any chunk, and each
chunk's embedding was computed independently of the question, at
ingestion time, long before this question existed. That's what makes it
cheap (nothing to recompute per query) and also what caps its accuracy:
it never actually looks at the question and a candidate chunk together.

A cross-encoder looks at them together - it takes (question, chunk) as
ONE joint input and scores that pair directly. This is consistently more
accurate at judging true relevance, at the cost of one model inference
PER CANDIDATE - too expensive to run against your entire collection, but
cheap enough against a small shortlist.

Stage 1 (retrieval.py's retrieve_with_scores, called with a wider k) does
the expensive part cheaply: narrows the whole collection down to a small
candidate pool with fast cosine comparison. Stage 2 (this module) does
the accurate part on a small input: re-scores and re-orders just those
candidates with the slower but more accurate cross-encoder. Running the
cross-encoder against the whole collection would be far too slow; running
cosine alone gives a weaker final ranking. Two stages gets both
properties.

WHY THE CALIBRATED CONFIDENCE BANDS (config.CONFIDENCE_BANDS) STILL USE
THE ORIGINAL COSINE SCORE, NOT A CROSS-ENCODER SCORE:
This cross-encoder's raw output is an UNCALIBRATED logit - not bounded to
[0, 1] the way the cosine relevance score is, and not on the same scale
at all. Introducing a second, differently-scaled number into the
"Strong match / Weak match" display would undo the calibration work
already validated against real good- and bad-match data. So the
cross-encoder is used ONLY to decide ranking and selection (which chunks
make the final top-k, and in what order) - the score shown to the user
is still each chunk's original, calibrated cosine relevance score,
carried through unchanged.
"""
from sentence_transformers import CrossEncoder

import config
from logging_setup import get_logger

logger = get_logger(__name__)

# WHY A MODULE-LEVEL CACHE (not built fresh per call, unlike generation.py's
# per-call ChatAnthropic client): loading a CrossEncoder means loading model
# weights from disk/HF cache - real, measurable latency. Unlike the LLM
# client (cheap to construct, since the actual model runs on Anthropic's
# servers), this model runs locally, so reconstructing it on every query
# would pay that load cost repeatedly for no reason. Reused across calls.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {config.RERANK_MODEL_NAME}")
        _reranker = CrossEncoder(config.RERANK_MODEL_NAME)
    return _reranker


def rerank(query: str, scored_results: list, top_k: int) -> list:
    """
    Re-score and re-order a candidate list using a cross-encoder, while
    keeping each chunk's ORIGINAL cosine relevance score for display.

    scored_results: list of (Document, cosine_relevance_score) tuples -
    typically from retrieval.retrieve_with_scores() called with a WIDER
    k (config.RERANK_CANDIDATE_K) than the final desired count, so there's
    an actual pool for the cross-encoder to select from.

    Returns the top_k (Document, cosine_relevance_score) tuples, re-ordered
    by cross-encoder relevance rather than cosine relevance. Note: the
    cosine score in each returned tuple is unchanged - it's the same value
    computed in stage 1, just possibly in a different position now.
    """
    if not scored_results:
        return []

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc, _ in scored_results]

    logger.info(f"Reranking {len(pairs)} candidate(s) with cross-encoder")
    cross_scores = reranker.predict(pairs)

    # WHY zip THEN sort BY cross_scores, NOT the original cosine scores:
    # This is the entire point of reranking - the FINAL order should
    # reflect the cross-encoder's joint judgment of (question, chunk),
    # not the cosine pre-filter's independent-embedding judgment. The
    # cosine score travels along unchanged, purely for display.
    combined = list(zip(scored_results, cross_scores))
    combined.sort(key=lambda item: item[1], reverse=True)

    reranked = [scored_result for scored_result, _ in combined[:top_k]]
    logger.info(f"Reranked {len(scored_results)} candidate(s) down to top {len(reranked)}")
    return reranked