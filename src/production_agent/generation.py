"""
Generation: build a grounded prompt from retrieved chunks and call the LLM.

WHY THIS FILE NOW USES LCEL (LangChain Expression Language):
Previously build_prompt() was a raw f-string and get_llm().invoke(prompt)
was called manually - functionally correct, but hand-rolled instead of
using LangChain's own composition layer. This version uses
ChatPromptTemplate (validates variables, supports future few-shot/system
message additions) piped into ChatAnthropic via the `|` operator - the
same `prompt | llm` pattern LangChain's own docs and examples use.

WHY MODEL ROUTING IS STILL DONE OUTSIDE A SINGLE STATIC CHAIN:
A textbook LCEL chain is built once and reused: `chain = prompt | llm`.
But here, WHICH model to use depends on the classified complexity of
THIS SPECIFIC question - it's a runtime decision, not something knowable
when the module loads. The standard LCEL pattern for this ("route to a
different chain/config per call based on a runtime classification") is a
RunnableLambda that builds and invokes the right sub-chain on each call -
see generate_answer() below. The `prompt | llm` composition itself is
still genuine LCEL; it's just constructed fresh per call with whichever
model was routed to, rather than being one fixed global object.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

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


# WHY A MODULE-LEVEL ChatPromptTemplate INSTEAD OF A build_prompt() FUNCTION:
# ChatPromptTemplate.from_template() parses {context}/{question} as named
# input variables ONCE, at import time, and validates them on every
# .invoke() call - a typo'd variable name now fails fast with a clear
# error instead of silently producing a prompt with a literal "{context}"
# in it. It's also directly composable with `| llm` below, which a plain
# f-string function isn't.
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a production support assistant.

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
)


def get_llm(model_name: str) -> ChatAnthropic:
    """
    Build the Anthropic chat client for a specific model.

    WHY model_name IS A PARAMETER INSTEAD OF ALWAYS config.LLM_MODEL_NAME:
    Different questions route to different models (see model_router.py),
    so this needs to build a client for WHICHEVER model was selected for
    this particular call, not a single hardcoded one.

    WHY max_retries IS SET EXPLICITLY:
    A transient network blip or momentary rate limit would otherwise fail
    the whole request immediately. max_retries gives the client a chance
    to recover on its own - standard resilience practice for any external
    API call in a production path.

    Note: the API key comes from config.ANTHROPIC_API_KEY, which fails
    loudly and early (at import time, in config.py) with a clear message
    if it's missing or empty - instead of failing deep inside the
    Anthropic SDK's internals the way the original bug did.
    """
    return ChatAnthropic(
        model=model_name,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_retries=config.LLM_MAX_RETRIES,
    )


def generate_answer(context: str, question: str) -> dict:
    """
    Generate an answer via an LCEL chain (prompt | llm), routing to a
    complexity-appropriate model and logging the actual cost of the call.

    WHY THIS BUILDS `prompt | llm` HERE RATHER THAN AT MODULE LOAD:
    The model (and therefore the llm object piped into the chain) isn't
    known until model_router classifies THIS question. So the LCEL chain
    itself is constructed per-call, using whichever model was routed to -
    still a real `prompt | llm` composition, just not a single global
    chain reused unchanged across every request the way a fixed-model
    setup could do.

    WHY THE CHAIN DOESN'T END IN StrOutputParser():
    A typical LCEL chain adds `| StrOutputParser()` to unwrap the final
    AIMessage down to a plain string. That would lose access to
    response.usage_metadata (the real token counts used for cost tracking
    below) - StrOutputParser discards everything on the message except
    its text. Since exact cost tracking is a hard requirement here, the
    chain intentionally stops at the raw AIMessage and .content is read
    off manually afterward, right where usage_metadata is also read.
    """
    model_name = model_router.select_model(question, context)
    llm = get_llm(model_name)

    chain = PROMPT_TEMPLATE | llm  # <-- the LCEL composition

    logger.debug(f"Invoking chain for model {model_name}")  # debug-level: hidden by default
    logger.info(f"Calling {model_name}")
    response = chain.invoke({"context": context, "question": question})

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