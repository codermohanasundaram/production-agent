"""
CLI entrypoint - the ONLY file you need to run.

WHY THIS CHANGED FROM THE PREVIOUS VERSION:
Originally ingestion.py and main.py were fully decoupled - you had to
remember to run `python ingestion.py` once before `python main.py "..."`
would return real answers. If you skipped that step, main.py ran without
error but silently fell back to "I don't have enough information" every
time, because retrieval found nothing in an empty/missing chroma_db - a
confusing trap with no error message pointing at the actual cause.

This version removes that trap: main.py now checks whether chroma_db
already has data in it, and if not, runs ingestion automatically before
answering. On the FIRST run, this makes it a bit slower (it has to load
the PDF, chunk, and embed before it can answer). On every run AFTER that,
chroma_db already exists, so the check short-circuits and it goes
straight to retrieval - same speed as before.

You still get the benefit of NOT re-ingesting on every single question
(the whole reason ingestion/query were split in the first place) - this
just adds "ingest automatically if it's never been done" on top, rather
than requiring you to remember a separate command.

Usage:
    python main.py "Why did the database connection fail?"
"""
import sys
from pathlib import Path

import config
from rag_pipeline import answer_question
from logging_setup import get_logger

logger = get_logger(__name__)


def ensure_ingested():
    """
    Run ingestion only if chroma_db doesn't already exist / is empty.

    WHY CHECK persist_directory FOR CONTENTS, NOT JUST EXISTENCE:
    Chroma can create an empty directory structure on first connect even
    if nothing was ever written to it (e.g. from a prior failed run or the
    retrieval side connecting before ingestion ran). Checking that the
    directory both exists AND has files in it is a more reliable signal
    that real data was actually persisted.
    """
    persist_dir = Path(config.CHROMA_PERSIST_DIR)
    already_ingested = persist_dir.exists() and any(persist_dir.iterdir())

    if already_ingested:
        logger.info(f"Found existing vector store at {persist_dir} - skipping ingestion")
        return

    logger.info(f"No existing vector store found at {persist_dir} - running ingestion now")
    from ingestion import run_ingestion  # imported here, not at top-level:
    # WHY import inside the function instead of at the top of the file:
    # ingestion.py pulls in PyPDFLoader, the text splitter, and the
    # embedding model - all only needed on a cold start. Importing lazily,
    # only when ingestion is actually about to run, keeps main.py's normal
    # (already-ingested) path lighter and avoids loading those dependencies
    # on every single query when they're not needed.
    run_ingestion()
    logger.info("Ingestion complete")


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<your question>"')
        sys.exit(1)

    question = sys.argv[1]

    ensure_ingested()
    result = answer_question(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    # WHY THIS BLOCK: makes cost-based routing AND retrieval confidence
    # visible to you, not just to the logs - shows which model answered
    # this specific question, what it actually cost, and how well
    # retrieval matched the question (see the honesty note in
    # retrieval.py's retrieve_with_scores() docstring - this is a
    # RETRIEVAL match score, not an answer-correctness score).
    #
    # WHY THE LABEL LEADS AND THE RAW % IS NOW SECONDARY (reversed from
    # before): a bare "45.2%" reads as mediocre by generic percentage
    # intuition, when it's actually this embedding model's typical score
    # for a CONFIRMED GOOD match (see config.CONFIDENCE_BANDS). Leading
    # with the calibrated label ("Strong match") and keeping the raw
    # number as parenthetical detail fixes the misread without hiding the
    # underlying number for anyone who wants it.
    if result.get("model"):
        print(
            f"\n(model: {result['model']} | "
            f"tokens: {result['input_tokens']} in / {result['output_tokens']} out | "
            f"cost: ${result['cost_usd']:.6f} | "
            f"retrieval confidence: {result['retrieval_confidence_label']} "
            f"({result['retrieval_confidence'] * 100:.1f}%))"
        )

    print("\n" + "=" * 70)
    print(f"SOURCES ({len(result['sources'])})")
    print("=" * 70)
    for i, src in enumerate(result["sources"], 1):
        page = src["page"] if src["page"] is not None else "N/A"  # non-PDF sources (txt/docx) have no page number
        score_pct = src["relevance_score"] * 100
        print(f"\n[{i}] {src['source']} (page {page}) - {src['relevance_label']} ({score_pct:.1f}%)")
        print(src["content"])


if __name__ == "__main__":
    main()