"""
Orchestrates retrieval + generation as an LCEL chain.

WHY THIS FILE NOW BUILDS A RunnableLambda | RunnableLambda CHAIN:
Previously answer_question() was a plain function calling retrieve_with_
scores() then generate_answer() in sequence - correct, but not composed
using LangChain's own chaining layer (LCEL). This version wraps each
stage as a RunnableLambda and composes them with the `|` operator, same
as generation.py's `prompt | llm`. Two stages (not one long chain) because
the retrieval stage needs to short-circuit on empty results before
generation ever runs - see _generate_step()'s docstring for why that
early-exit doesn't fit cleanly into a single linear `|` chain.

WHY THIS STILL EXISTS AS A SEPARATE FILE FROM generation.py:
Same reasoning as before this refactor: this is the seam between
"pipeline logic" and "how it gets invoked" (CLI, API endpoint, tests).
main.py and api.py both call answer_question() without knowing or caring
that it's now backed by an LCEL chain internally - the external contract
(a question in, a dict out) hasn't changed.
"""
from langchain_core.runnables import RunnableLambda

import config
from logging_setup import get_logger
from retrieval import retrieve_with_scores
from generation import build_context, generate_answer

logger = get_logger(__name__)


def _retrieve_step(inputs: dict) -> dict:
    """
    LCEL stage 1: retrieve scored chunks and assemble context.

    WHY A dict IN, dict OUT (not question: str -> list[Document]):
    LCEL steps composed with `|` pass ONE value from each step to the
    next. Using a dict as that value (rather than positional args) is the
    standard LCEL convention for multi-field state, since it lets later
    steps in the chain read whichever earlier fields they need (question,
    k, scored_results, context) without every step's signature having to
    change when a new field is added.
    """
    question = inputs["question"]
    k = inputs.get("k", config.RETRIEVAL_K)

    scored_results = retrieve_with_scores(question, k=k)
    context = build_context([doc for doc, _ in scored_results]) if scored_results else ""

    return {"question": question, "scored_results": scored_results, "context": context}


def _generate_step(inputs: dict) -> dict:
    """
    LCEL stage 2: route to a model and generate, or short-circuit if
    retrieval found nothing.

    WHY THE EMPTY-RESULTS CHECK LIVES HERE, NOT AS A RunnableBranch:
    LCEL has RunnableBranch for conditional routing, but it's built for
    choosing between multiple FULL sub-chains based on a condition - here
    the "empty" case just needs to skip generation entirely and return a
    fixed dict, which is simplest to express as a plain early-return
    inside the lambda rather than wiring up a branch object for one
    trivial case.
    """
    scored_results = inputs["scored_results"]

    if not scored_results:
        # WHY handle this explicitly:
        # An empty collection (e.g. ingestion never ran, or ran against
        # the wrong persist_directory) would otherwise send empty context
        # straight to the LLM with no signal about why.
        logger.warning("No documents retrieved - vector store may be empty")
        return {
            "answer": "I don't have enough information.",
            "sources": [],
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "retrieval_confidence": 0.0,
        }

    # WHY AVERAGE THE PER-CHUNK SCORES INTO ONE NUMBER:
    # Individual chunk scores are kept in each source's entry below (for
    # debugging a specific weak match). The average gives one glance-able
    # number for "how well did retrieval do on this question overall" -
    # e.g. printed alongside model/cost in main.py.
    retrieval_confidence = sum(score for _, score in scored_results) / len(scored_results)

    generation_result = generate_answer(inputs["context"], inputs["question"])

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


# WHY THIS CHAIN IS BUILT ONCE AT MODULE LOAD, UNLIKE generation.py's
# `prompt | llm` (built fresh per call):
# Neither step here depends on which MODEL gets used - that decision
# happens INSIDE _generate_step (via generate_answer -> model_router),
# not in how these two stages connect to each other. So the outer
# retrieve-then-generate structure is genuinely static and can be
# composed once, the textbook LCEL pattern - only the inner
# `prompt | llm` sub-chain in generation.py needs per-call construction.
rag_chain = RunnableLambda(_retrieve_step) | RunnableLambda(_generate_step)


def answer_question(question: str, k: int = config.RETRIEVAL_K) -> dict:
    """
    Run the LCEL retrieve-then-generate chain for a single question.

    WHY THIS FUNCTION STILL EXISTS (rather than callers invoking rag_chain
    directly):
    Keeps the external interface unchanged for main.py/api.py - they call
    answer_question(question) exactly as before. This is also the natural
    place to later add rag_chain.ainvoke(...) for an async variant, or
    rag_chain.batch(...) for evaluating multiple questions at once,
    without touching any caller.
    """
    return rag_chain.invoke({"question": question, "k": k})