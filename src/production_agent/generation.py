"""
Generation: build a grounded prompt from retrieved chunks and call the LLM.

WHY MODEL SELECTION MOVED FROM A FIXED MODEL TO model_router:
generate_answer() now asks model_router.select_model() which model to use
PER QUESTION, based on the question's classified complexity, instead of
always using config.LLM_MODEL_NAME. After the response comes back, actual
token usage from the API response is used to compute and log real cost -
not an estimate made in advance, but what the call actually cost.
"""
from langchain_anthropic import ChatAnthropic

import config
import model_router
from logging_setup import get_logger

logger = get_logger(__name__)


def build_context(documents) -> str:
    """
    Format retrieved documents into a single context string for the prompt.

    THIS FIXES A REAL BUG IN THE ORIGINAL CODE:
    Your original function had `return "\\n".join(contexts)` indented
    INSIDE the for loop:

        for i, document in enumerate(documents):
            contexts.append(...)
            return "\\n".join(contexts)   # <- returns after first iteration!

    That means the function always exited after processing just the FIRST
    retrieved document, silently discarding the other k-1 results. You'd
    call similarity_search(query, k=3), get 3 chunks back, and only ever
    feed 1 of them to the LLM - no error, no warning, just quietly worse
    answers. Fixed below by dedenting the return to loop level.
    """
    contexts = []
    for i, document in enumerate(documents):
        contexts.append(
            f"""
            Context: {i + 1}
            Source: {document.metadata.get("source")}
            Page: {document.metadata.get("page")}
            {document.page_content}
            """
        )
    return "\n".join(contexts)  # now outside the loop - uses ALL retrieved documents


def build_prompt(context: str, question: str) -> str:
    return f"""You are a production support assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use information outside the context.
2. Do not make up facts.
3. If the answer cannot be found in the context,
   say "I don't have enough information."
4. Keep the answer concise and factual.

Context:
{context}

Question:
{question}
"""


def get_llm(model_name: str) -> ChatAnthropic:
    """
    Build the Anthropic chat client for a specific model.

    WHY model_name IS NOW A PARAMETER INSTEAD OF ALWAYS config.LLM_MODEL_NAME:
    Different questions now route to different models (see model_router.py),
    so this needs to build a client for WHICHEVER model was selected for
    this particular call, not a single hardcoded one.

    WHY max_retries IS SET EXPLICITLY:
    Your original code didn't set this, so a transient network blip or a
    momentary rate limit would fail the whole request immediately. Setting
    max_retries gives the client a chance to recover from transient errors
    on its own before surfacing a failure to the caller - standard
    resilience practice for any external API call in a production path.

    Note: the API key itself now comes from config.ANTHROPIC_API_KEY, which
    fails loudly and early (at import time, in config.py) with a clear
    message if the key is missing or empty - instead of failing deep inside
    the Anthropic SDK's internals the way your original TypeError did.
    """
    return ChatAnthropic(
        model=model_name,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_retries=config.LLM_MAX_RETRIES,
    )


def generate_answer(context: str, question: str) -> dict:
    """
    Generate an answer, routing to a complexity-appropriate model and
    logging the actual cost of the call.

    WHY THIS NOW RETURNS A dict INSTEAD OF JUST response.content:
    Previously this returned only the answer text. Now that different
    calls can use different models at different prices, callers (and you,
    reading logs) need visibility into WHICH model answered and WHAT it
    cost - not just the answer itself. rag_pipeline.py and main.py are
    updated to read result["answer"] instead of using the return value
    directly as a string.
    """
    model_name = model_router.select_model(question, context)
    prompt = build_prompt(context, question)
    logger.debug(f"Prompt sent to LLM:\n{prompt}")  # debug-level: hidden by default

    llm = get_llm(model_name)
    logger.info(f"Calling {model_name}")
    response = llm.invoke(prompt)

    # WHY ACTUAL usage_metadata INSTEAD OF ESTIMATING TOKENS OURSELVES:
    # langchain_anthropic populates response.usage_metadata with the real
    # input/output token counts Anthropic billed for this specific call.
    # Estimating token counts ourselves (e.g. len(text)/4) before the call
    # would only ever be an approximation; using the API's own reported
    # counts after the call gives an exact cost figure, not a guess.
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost_usd = model_router.estimate_cost(model_name, input_tokens, output_tokens)

    logger.info(
        f"Call cost: model={model_name} "
        f"input_tokens={input_tokens} output_tokens={output_tokens} "
        f"cost=${cost_usd:.6f}"
    )

    return {
        "answer": response.content,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }