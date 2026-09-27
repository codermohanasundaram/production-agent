"""
Central configuration for the RAG pipeline.

WHY THIS FILE EXISTS:
Your original script had chunk_size=500, chunk_overlap=50, k=3, the model
name "claude-sonnet-4-5", the embedding model name, the collection name,
and the chroma_db path all hardcoded inline across the file. That means
tuning retrieval quality (say, trying chunk_size=800) requires opening
ingestion.py and editing pipeline logic - the same file that also loads
PDFs and calls Chroma. Config and logic end up tangled.

Pulling these into one place means:
- You can tune retrieval/chunking without touching pipeline code.
- Different environments (dev/staging/prod) can override values via .env
  without a code change.
- It's obvious at a glance what the "knobs" of this system are.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly relative to project root, not relative to CWD.
# WHY: load_dotenv() with no args only searches the current working
# directory (or walks upward from it). If you run `python ingestion.py`
# from a different folder than the project root, it silently finds
# nothing and ANTHROPIC_API_KEY ends up empty - which is exactly the
# authentication error you hit earlier. Pointing at PROJECT_ROOT / ".env"
# explicitly removes that ambiguity.
PROJECT_ROOT = Path(__file__).resolve().parents[0]
load_dotenv()

# --- Paths -------------------------------------------------------------
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DEFAULT_PDF_PATH = DOCUMENTS_DIR / "error_log_sample.pdf"  # kept for reference/back-compat; ingestion now scans DOCUMENTS_DIR, not this single file
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")  # Chroma wants a str, not a Path

# WHY THIS CONSTANT WAS ADDED:
# ingestion.py's LOADER_REGISTRY is the source of truth for what's
# actually supported (it maps extension -> loader class). This constant
# just mirrors those keys for anywhere outside ingestion.py that wants to
# reference "what formats do we support" (e.g. an upload endpoint later
# validating a file before accepting it) without importing ingestion.py's
# internals directly.
SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".txt")

# --- Chunking ------------------------------------------------------------
# WHY THESE VALUES CHANGED FROM (500, 50) TO (300, 0):
# Your source documents (error_log_sample.pdf, cloudwatch_sample_log.txt)
# are line-oriented: each log line is one complete, self-contained event.
# At chunk_size=500, RecursiveCharacterTextSplitter was routinely merging
# 3-4 unrelated log lines into a single chunk (each line here runs roughly
# 100-260 characters), which blurred that chunk's embedding across several
# unrelated topics - directly causing the low retrieval-precision results
# seen in testing (e.g. only 1 of 3 retrieved chunks actually relevant).
#
# chunk_size=300 is sized so that, combined with separators=["\n", ...]
# below (which makes the splitter prefer breaking on line boundaries over
# mid-line), most chunks end up holding ONE log line, occasionally two
# short ones - not three or four unrelated events mashed together.
#
# chunk_overlap=0: overlap exists to avoid cutting a sentence/idea in half
# across a chunk boundary. Since each log line is already a complete,
# atomic unit, overlap here would only duplicate whole lines into
# neighboring chunks (adding noise, not preserving meaning) - so it's set
# to 0 rather than carried over from the original prose-oriented default.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

# WHY THIS SEPARATOR LIST:
# RecursiveCharacterTextSplitter tries each separator in order and only
# falls back to the next one if a chunk is still over chunk_size. Putting
# "\n" first (before " " and "") means it will always prefer to break
# between lines rather than mid-word or mid-line - this is what actually
# keeps individual log lines intact as chunk boundaries, not chunk_size
# alone. Kept here (not just in ingestion.py) so it's visible as a tuned
# setting, same as CHUNK_SIZE/CHUNK_OVERLAP above.
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# --- Embeddings ----------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "production_logs"

# WHY THIS CONSTANT WAS ADDED:
# Chroma's default distance metric is L2 (Euclidean), which is NOT bounded
# to a fixed range - even a strong semantic match can produce a distance
# of ~1.0, which earlier made the hand-rolled 1/(1+distance) "relevance
# score" read as ~50% for genuinely good matches (misleadingly low-looking
# despite being a good result). Cosine distance IS bounded, and LangChain's
# Chroma integration has a built-in cosine relevance-score function
# (similarity_search_with_relevance_scores) that produces a properly
# interpretable 0-1 score once the collection is configured for it - see
# ingestion.py's embed_and_store() and retrieval.py's get_vector_store(),
# both of which pass this into Chroma's collection_metadata.
#
# IMPORTANT: this only takes effect for a NEWLY CREATED collection - an
# existing chroma_db/ built under the old default metric keeps using it.
# Delete chroma_db/ and re-run ingestion for this to apply.
VECTOR_DISTANCE_METRIC = "cosine"

# --- Retrieval -------------------------------------------------------------
RETRIEVAL_K = 3

# WHY THIS CONSTANT WAS ADDED:
# Raw cosine relevance scores from EMBEDDING_MODEL_NAME have consistently
# landed in the 0.42-0.48 range for CONFIRMED GOOD matches throughout
# testing (multiple different questions, all verified against ground
# truth). Shown as a bare percentage, "45%" reads as mediocre/uncertain
# to anyone unfamiliar with this embedding model's actual score
# distribution - when it's actually representative of a strong match for
# this model. These bands translate the raw score into a label calibrated
# against what THIS embedding model's "good" actually looks like, instead
# of against generic percentage intuition (where 45% suggests "worse than
# a coin flip", which is the wrong read here).
#
# CAVEAT - THIS IS A PROVISIONAL CALIBRATION, NOT A MEASURED ONE:
# All observations so far are from CONFIRMED-RELEVANT retrievals - every
# test question so far has returned genuinely on-topic chunks. There is
# no data yet on what a confirmed-IRRELEVANT match scores with this
# model, so the lower bands (WEAK/VERY_WEAK below) are reasonable
# starting guesses, not measurements. Revisit these thresholds once
# you've observed some clearly-bad retrievals (e.g. asking a question
# with no relevant log data present) and noted where their scores land.
#
# Ordered highest-to-lowest; retrieval.interpret_confidence() walks this
# list and returns the first band whose threshold the score meets.
CONFIDENCE_BANDS = [
    (0.45, "Strong match"),
    (0.35, "Good match"),
    (0.20, "Weak match - verify sources"),
    (0.00, "Very weak - likely no relevant data"),
]

# --- Reranking -------------------------------------------------------------
# WHY RERANKING WAS ADDED:
# Cosine similarity compares PRE-COMPUTED embeddings - the question's
# embedding and each chunk's embedding were computed independently, at
# different times, without ever "looking at" each other together. That's
# what makes it cheap (no recomputation per query), and also what caps
# its precision. A cross-encoder scores (question, chunk) as a SINGLE
# joint input, which is consistently more accurate - at the cost of one
# model inference per candidate, too slow to run against a whole
# collection but cheap against a small shortlist. See reranking.py for
# the full mechanics.
RERANK_ENABLED = True
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# WHY THIS IS LARGER THAN RETRIEVAL_K:
# Reranking needs a wider candidate pool to actually improve on cosine's
# top-k - if you only ever pulled the same k candidates cosine would have
# returned anyway, reranking could only reorder them, never surface a
# genuinely better chunk that cosine ranked just outside the top k. Pull
# more candidates cheaply via cosine, then let the more accurate
# cross-encoder pick the real top RETRIEVAL_K from that wider pool.
RERANK_CANDIDATE_K = 15

# --- LLM ---------------------------------------------------------------
# WHY THREE MODELS INSTEAD OF ONE:
# Your original code called one fixed model ("claude-sonnet-4-5") for every
# question, regardless of whether the question was a simple one-line lookup
# or a multi-part analytical question needing more reasoning. That means
# you were paying Sonnet-level cost even for questions a cheaper model
# would answer just as correctly.
#
# NOTE: "claude-sonnet-4-5" (your original value) is the older Sonnet 4.5,
# priced at $3/$15 per MTok. The current-generation Sonnet 5 is $2/$10 per
# MTok - cheaper AND newer. Updated LLM_MODEL_NAME below accordingly; it's
# kept as the "standard" tier's model and as the fallback default.
#
# Below, MODEL_PRICING and COMPLEXITY_MODEL_MAP set up complexity-based
# routing: model_router.py classifies each question as simple/standard/
# complex and picks the cheapest model tier that's still appropriate,
# rather than always paying for the most capable (most expensive) model.
LLM_TEMPERATURE = 0
LLM_MAX_RETRIES = 3  # retry transient API errors/rate limits instead of failing hard

# Prices in USD per million tokens (MTok), from Anthropic's published
# pricing as of this writing. Pull these into config (not hardcoded in
# model_router.py) so a price change is a one-line edit, not a code change.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

# WHY A SEPARATE MAP FROM MODEL_PRICING:
# MODEL_PRICING answers "what does this model cost". COMPLEXITY_MODEL_MAP
# answers "which model do we use for this tier of question". Keeping them
# separate means you can retune routing (e.g. move "standard" questions to
# Opus for a higher-stakes deployment) without touching pricing data, and
# vice versa when Anthropic updates prices.
COMPLEXITY_MODEL_MAP = {
    "simple": "claude-haiku-4-5-20251001",   # short factual lookups - cheapest tier
    "standard": "claude-sonnet-5",            # most production questions - balanced cost/quality
    "complex": "claude-opus-5-5",             # multi-part / analytical questions - highest quality, highest cost
}

# Fallback/default model if routing is bypassed or a tier lookup fails.
LLM_MODEL_NAME = COMPLEXITY_MODEL_MAP["standard"]

# --- API key -------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    # WHY fail here, at import time, with a clear message:
    # Your original code let os.environ["ANTHROPIC_API_KEY"] pass an empty
    # string all the way into the Anthropic SDK, which only failed deep in
    # a stack of library internals with a confusing TypeError. Failing loud
    # and early, in plain language, at the config layer, saves you from
    # chasing that traceback again.
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set or is empty. "
        f"Check that {PROJECT_ROOT / '.env'} exists and contains "
        "ANTHROPIC_API_KEY=<your key> with no quotes and no trailing spaces."
    )