"""
Complexity-based model routing.

WHY THIS FILE EXISTS:
Every question previously went to the same fixed model. That's simple but
wasteful: a one-line factual lookup ("what error occurred at 02:18?") costs
the same as a multi-part analytical question ("compare failure patterns
across all services and identify the root cause"), even though the first
needs far less reasoning capability.

This module classifies each question into a complexity tier (simple /
standard / complex) using cheap, fast heuristics - no extra LLM call
needed just to decide which LLM to call, which would defeat the purpose -
then maps that tier to a specific model via config.COMPLEXITY_MODEL_MAP.

WHY HEURISTICS INSTEAD OF ASKING AN LLM TO CLASSIFY:
Using an LLM call to decide "which LLM should I use" adds a full extra
request (latency + cost) before your real request even starts - for a
routing decision that needs to be cheap and fast, not another judgment
call. Simple signals (question length, presence of analytical keywords,
how much context was retrieved) are fast, free, and, for a support-log
Q&A use case, correlate well enough with actual complexity to route
correctly most of the time. This is a heuristic, not a guarantee - see
the note in classify_complexity() about tuning it as you observe results.
"""
import config
from logging_setup import get_logger

logger = get_logger(__name__)


# Words/phrases that signal a question needs more reasoning than a single
# fact lookup - comparison, synthesis across multiple sources, causal
# analysis, or open-ended "tell me everything about X".
COMPLEX_KEYWORDS = [
    "compare", "comparison", "analyze", "analysis", "root cause",
    "across", "trend", "correlate", "correlation", "recommend",
    "architecture", "trade-off", "tradeoff", "explain in detail",
    "summarize all", "summarise all", "pattern", "patterns",
    "relationship between", "impact of", "why do", "why does",
]

# Rough chars-per-token estimate for English text - good enough for a
# routing threshold, not meant to be an exact token count (that would
# require actually tokenizing, which is unnecessary overhead here).
CHARS_PER_TOKEN_ESTIMATE = 4

SIMPLE_MAX_WORDS = 12
SIMPLE_MAX_CONTEXT_TOKENS = 1200
COMPLEX_MIN_CONTEXT_TOKENS = 3000


def classify_complexity(question: str, context: str) -> str:
    """
    Classify a question into 'simple', 'standard', or 'complex'.

    WHY THESE PARTICULAR SIGNALS:
    - word_count: a short question ("why did X fail?") is usually a direct
      lookup; a long question usually has multiple clauses/conditions to
      satisfy.
    - context size: more retrieved context generally means the question
      touches more source material, which needs more reasoning to
      synthesize correctly.
    - keyword match: words like "compare" or "root cause" are strong,
      cheap signals that the question wants synthesis/analysis, not just
      fact retrieval - these override the word-count/context signals.

    TUNING THIS FOR YOUR USE CASE:
    These thresholds are reasonable starting points, not universal truth.
    If you find simple questions getting routed to "standard"/"complex"
    too often (overpaying) or vice versa (underpowered answers), the
    numbers here are what to adjust - or log actual outcomes and revisit.
    """
    word_count = len(question.split())
    context_tokens_estimate = len(context) / CHARS_PER_TOKEN_ESTIMATE
    question_lower = question.lower()

    has_complex_keyword = any(kw in question_lower for kw in COMPLEX_KEYWORDS)

    if has_complex_keyword or context_tokens_estimate > COMPLEX_MIN_CONTEXT_TOKENS:
        tier = "complex"
    elif word_count <= SIMPLE_MAX_WORDS and context_tokens_estimate < SIMPLE_MAX_CONTEXT_TOKENS:
        tier = "simple"
    else:
        tier = "standard"

    logger.info(
        f"Complexity classification: {tier} "
        f"(words={word_count}, context_tokens~={int(context_tokens_estimate)}, "
        f"complex_keyword_match={has_complex_keyword})"
    )
    return tier


def select_model(question: str, context: str) -> str:
    """Classify the question and return the model name to use for it."""
    tier = classify_complexity(question, context)
    model_name = config.COMPLEXITY_MODEL_MAP[tier]
    logger.info(f"Routing to model: {model_name} (tier={tier})")
    return model_name


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute the USD cost of a call given actual token counts and a model's
    published per-MTok pricing in config.MODEL_PRICING.

    WHY THIS FUNCTION EXISTS:
    Knowing which model was used isn't the same as knowing what it cost -
    input and output tokens are priced differently (output is typically
    5x input), so an accurate cost figure needs both counts, not just a
    per-call flat estimate. This is called AFTER the real API response
    comes back (see generation.py), using the model's ACTUAL reported
    token usage - not a guess made before the call.
    """
    pricing = config.MODEL_PRICING.get(model_name)
    if pricing is None:
        logger.warning(f"No pricing entry for model '{model_name}' - cost not tracked")
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost