"""
Orchestrates retrieval + generation into a single callable function.

WHY THIS FILE EXISTS:
Your original script ended with a hardcoded query string
(`query = "Why did the database connection fail?"`) and ran straight
through to printing an answer - it was a smoke-test script, not something
that could be called or reused as a service.

answer_question() below is the seam between "pipeline logic" and "how it
gets invoked" (CLI, API endpoint, another script, tests, etc). main.py
is one such caller; you could just as easily wrap this same function in
a FastAPI route later without touching this file.
"""
import config
from logging_setup import get_logger
from retrieval import retrieve_with_scores
from generation import build_context, generate_answer

logger = get_logger(__name__)


def answer_question(question: str, k: int = config.RETRIEVAL_K) -> dict:
    """
    Run retrieval + generation for a single question.

    Returns a dict (not just the raw answer string) so callers can also
    inspect what was retrieved - useful for debugging bad answers and for
    showing sources to an end user, neither of which was possible when
    everything was printed inline in one script.
    """
    scored_results = retrieve_with_scores(question, k=k)

    if not scored_results:
        # WHY handle this explicitly:
        # Your original code assumed similarity_search always returns
        # something. An empty collection (e.g. ingestion never ran, or ran
        # against the wrong persist_directory) would otherwise send an
        # empty context straight to the LLM with no signal about why.
        logger.warning("No documents retrieved - vector store may be empty")
        return {
            "answer": "I don't have enough information.",
            "sources": [],
            "model": None,
            "cost_usd": 0.0,
            "retrieval_confidence": 0.0,
        }

    results = [doc for doc, score in scored_results]
    # WHY AVERAGE THE PER-CHUNK SCORES INTO ONE NUMBER:
    # Individual chunk scores are in each source's entry below (for
    # debugging a specific weak match). The average gives one glance-able
    # number for "how well did retrieval do on this question overall" -
    # e.g. printed alongside model/cost in main.py.
    retrieval_confidence = sum(score for _, score in scored_results) / len(scored_results)

    context = build_context(results)
    generation_result = generate_answer(context, question)
    # WHY UNPACKED HERE INSTEAD OF PASSED THROUGH AS-IS:
    # generate_answer() now returns a dict (answer + model + token counts +
    # cost), not just a string - see generation.py. Merging its fields into
    # this function's own return dict, rather than nesting it under a
    # "generation" key, keeps the shape callers (main.py) work with flat
    # and simple: result["answer"], result["cost_usd"], etc.

    return {
        "answer": generation_result["answer"],
        "model": generation_result["model"],
        "input_tokens": generation_result["input_tokens"],
        "output_tokens": generation_result["output_tokens"],
        "cost_usd": generation_result["cost_usd"],
        "retrieval_confidence": retrieval_confidence,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content,
                "relevance_score": score,
            }
            for doc, score in scored_results
        ],
    }