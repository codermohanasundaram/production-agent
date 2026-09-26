"""
Ingestion pipeline: load documents (PDF/DOCX/TXT) -> chunk -> embed -> persist to Chroma.

WHY THIS IS A SEPARATE MODULE FROM RETRIEVAL/GENERATION:
Your original ingestion.py did load -> chunk -> embed -> store -> retrieve
-> prompt -> generate all in one linear script. That conflates two very
different lifecycles:

  - Ingestion happens once per document (or on a schedule, or when new
    documents arrive). It's expensive: embedding models load, files parse,
    vectors get computed.
  - Querying happens once PER USER QUESTION. It should be cheap and fast -
    reconnect to an existing vector store and retrieve, nothing more.

With everything in one script, every question you ask requires re-running
the ENTIRE embedding pipeline again, which is both wasteful and wrong for
a real system. This module only does ingestion. retrieval.py and
generation.py handle the query side and simply reconnect to what this
module already built.

WHY THIS VERSION SUPPORTS MULTIPLE FILES AND MULTIPLE FORMATS:
Originally this module only knew how to open a single, specific PDF path
(PyPDFLoader against config.DEFAULT_PDF_PATH). Real ingestion needs to
scan a whole documents folder and handle whatever's dropped into it - PDFs,
Word docs, and plain text logs - without you hand-editing this file every
time a new file type shows up. The design below is a "loader registry":
one dict mapping file extension -> the LangChain loader class that knows
how to read it. Adding a new format later (e.g. .md, .csv) means adding
one line to that dict, not rewriting the ingestion flow.
"""
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config
from logging_setup import get_logger

logger = get_logger(__name__)


# WHY A REGISTRY DICT INSTEAD OF if/elif CHAINS:
# A dict keyed by extension is the standard "strategy pattern" for this
# kind of dispatch - it's declarative (the supported formats are visible
# at a glance), and extending it later is a one-line change instead of
# editing branching logic. TextLoader needs an explicit encoding or it can
# fail on non-UTF-8 log files pulled from Windows tools, so it's wrapped in
# a lambda to pin encoding="utf-8" without changing the other loaders'
# call signature.
LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
}


def load_document(file_path: Path):
    """
    Load a single file into LangChain Document objects, dispatching on
    file extension via LOADER_REGISTRY.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at: {file_path}")

    suffix = file_path.suffix.lower()
    loader_factory = LOADER_REGISTRY.get(suffix)

    if loader_factory is None:
        # WHY warn-and-skip instead of raising here:
        # This function gets called once per file while scanning a whole
        # directory (see load_documents_from_directory below). If one
        # unsupported file (e.g. a stray .xlsx or .png dropped in the same
        # folder) raised an exception, it would kill ingestion for every
        # OTHER valid file in the batch too. Skipping just that one file
        # with a warning is more resilient for a directory-scan workflow.
        logger.warning(f"Skipping unsupported file type: {file_path.name}")
        return []

    logger.info(f"Loading {suffix} file: {file_path.name}")
    loader = loader_factory(str(file_path))
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document object(s) from {file_path.name}")
    return documents


def load_documents_from_directory(directory: Path = config.DOCUMENTS_DIR):
    """
    Scan a directory for every supported file (PDF/DOCX/TXT) and load them
    all into one combined list of Document objects.

    WHY THIS FUNCTION EXISTS:
    You asked to stop checking a single hardcoded file and instead support
    multiple files/formats. This is the "multi-file" half of that: it
    globs the documents directory for anything whose extension is in
    LOADER_REGISTRY, loads each one via load_document(), and concatenates
    the results - so run_ingestion() below can embed everything in one
    pass regardless of how many files or which formats are present.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found at: {directory}")

    all_documents = []
    matched_files = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in LOADER_REGISTRY
    ]

    if not matched_files:
        # WHY warn instead of silently returning []:
        # An empty documents folder (or one full of only unsupported
        # formats) is easy to miss otherwise - ingestion would "succeed"
        # having embedded nothing, and the caller would only notice much
        # later when queries come back with "I don't have enough
        # information" for reasons that look unrelated to this.
        logger.warning(
            f"No supported files found in {directory} "
            f"(supported: {', '.join(LOADER_REGISTRY.keys())})"
        )
        return all_documents

    logger.info(f"Found {len(matched_files)} file(s) to ingest: "
                f"{[f.name for f in matched_files]}")

    for file_path in matched_files:
        all_documents.extend(load_document(file_path))

    logger.info(f"Loaded {len(all_documents)} document object(s) total "
                f"from {len(matched_files)} file(s)")
    return all_documents


def chunk_documents(documents):
    """
    Split documents into retrieval-sized chunks.

    CHUNKING STRATEGY - NOW FIXED (previously flagged, not changed):
    Your source documents are line-oriented: each log line is one complete,
    self-contained event. The original chunk_size=500 with default
    separators was merging 3-4 unrelated log lines into a single chunk,
    which blurred that chunk's embedding across several unrelated topics -
    this was the direct cause of the low retrieval-precision results seen
    in testing (only 1 of 3 retrieved chunks actually relevant to the
    question asked).

    config.CHUNK_SIZE is now 300 (down from 500) and config.CHUNK_SEPARATORS
    puts "\\n" ahead of " " and "" - RecursiveCharacterTextSplitter tries
    each separator in order and only falls back to the next if a chunk is
    still oversized, so putting "\\n" first means it prefers breaking
    between lines over breaking mid-line. Combined, most chunks now hold
    one log line, occasionally two short ones, instead of three or four
    unrelated events mashed together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def _make_chunk_id(chunk, index: int) -> str:
    """
    Build a deterministic ID for a chunk.

    WHY THIS FUNCTION EXISTS - THIS FIXES A REAL BUG IN THE ORIGINAL CODE:
    Your original script called Chroma.from_documents(...) every time
    ingestion.py ran, with no IDs. Chroma doesn't deduplicate by content -
    it appends. Run the script three times and you have three copies of
    every chunk sitting in the same collection, silently degrading
    retrieval (duplicate chunks crowd out distinct ones in top-k results).

    Building a deterministic ID from stable fields (source path, page
    number, position within the page) means re-running ingestion on the
    same document UPSERTS instead of duplicating - same ID in, same vector
    overwritten, not appended.

    WHY page DEFAULTS TO "na" NOW, NOT JUST metadata.get("page"):
    PyPDFLoader sets metadata["page"] for every chunk, but Docx2txtLoader
    and TextLoader don't - a .docx or .txt file has no concept of "page".
    Without a stable fallback, every chunk from a non-PDF file would get
    the same "unknown" placeholder for page, and combined with the same
    source and (potentially, across two different runs) the same index,
    IDs could collide across unrelated chunks. Using "na" is just a
    clearer, explicit marker than "unknown" that this format has no page
    concept - the index still keeps each chunk's ID unique within a file.
    """
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", "na")
    return f"{source}::page={page}::chunk={index}"


def embed_and_store(chunks) -> Chroma:
    """
    Embed chunks and persist them to Chroma, upserting by deterministic ID.

    WHY collection_metadata={"hnsw:space": ...} WAS ADDED:
    This sets Chroma's internal distance metric to cosine (see
    config.VECTOR_DISTANCE_METRIC) instead of its default, L2 (Euclidean).
    L2 distance isn't bounded to a fixed range, which made retrieval
    confidence scores read as misleadingly low (~50%) even for genuinely
    strong matches. Cosine distance IS bounded, and LangChain's Chroma
    integration has a built-in relevance-score function for it that
    retrieval.py now uses directly - see retrieve_with_scores() there.

    This only takes effect when the collection is FIRST created - it's
    baked into the collection's index at creation time, not something that
    can be changed on an existing collection. That's why this feature
    requires deleting chroma_db/ and re-running ingestion from scratch,
    same as the chunking change above.
    """
    embeddings = get_embeddings()
    ids = [_make_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

    logger.info(
        f"Writing {len(chunks)} chunk(s) to Chroma "
        f"(collection='{config.COLLECTION_NAME}', dir='{config.CHROMA_PERSIST_DIR}', "
        f"distance_metric='{config.VECTOR_DISTANCE_METRIC}')"
    )
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": config.VECTOR_DISTANCE_METRIC},
        ids=ids,  # <-- the fix: same document re-ingested = same IDs = upsert, not duplicate
    )
    logger.info("Vector database write complete")
    return vector_db


def run_ingestion(directory: Path = config.DOCUMENTS_DIR) -> Chroma:
    """
    Full ingestion pipeline for every supported file in a directory.
    Entry point for this module.

    WHY THIS SIGNATURE CHANGED (file_path -> directory):
    Previously this took one PDF path and processed exactly that file.
    Now it takes a directory and processes every PDF/DOCX/TXT file inside
    it in one pass - this is what makes "drop more files into documents/,
    re-run ingestion, they all get embedded" possible, instead of only
    ever handling the single file that was hardcoded in config.py.
    """
    documents = load_documents_from_directory(directory)

    if not documents:
        # WHY bail out here with a clear message instead of proceeding:
        # chunk_documents([]) and embed_and_store([]) would technically
        # "succeed" on an empty list and write nothing to Chroma - the
        # same silent-failure trap described in load_documents_from_directory.
        # Stopping here with a clear log line makes the "nothing to ingest"
        # case visible immediately instead of surfacing later as a
        # confusing empty answer from main.py.
        logger.warning("No documents loaded - nothing to ingest")
        return None

    chunks = chunk_documents(documents)
    vector_db = embed_and_store(chunks)
    return vector_db


if __name__ == "__main__":
    # WHY the `if __name__ == "__main__":` guard:
    # Without it, importing this module anywhere else (e.g. from main.py)
    # would re-run the entire ingestion pipeline as a side effect of the
    # import statement. The guard makes "run ingestion" an explicit action
    # (`python ingestion.py`) rather than something that happens implicitly.
    run_ingestion()
    print("\nIngestion complete.")