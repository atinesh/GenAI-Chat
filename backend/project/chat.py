import tiktoken
import json
from flask import current_app

from project.vector_db.redis_db import RedisDB
from project.llm.azure_openai import AzureOpenAIChat
from project.retrieval.pipeline import retrieve
from project.retrieval.query_rewriter import rewrite_query
from project.observability.request_logger import RequestTrace
import config as cfg


# Cache the tokenizer encoder at module load (avoid re-init per call).
try:
    _ENCODER = tiktoken.get_encoding(cfg.TOKENIZER_ENCODING)
except Exception:
    # Fallback if encoding name is unknown in installed tiktoken version
    _ENCODER = tiktoken.get_encoding("cl100k_base")

_INSTRUCTIONS = """
You answer the user's question using the document context provided in the previous system message. Each document there starts with its source filename in square brackets, e.g. `[policy_manual.pdf]`, followed by the document text.

**Citation rule (must be followed):**
- If any part of your answer uses information from the provided documents — facts, definitions, names, procedures, paraphrased content, specific terms — you MUST end your answer with `(SOURCE: filename)` listing every filename you drew from.
- If the answer draws from multiple documents, list all filenames separated by ` | ` inside a single `(SOURCE: ...)` tag.
- Use the filename EXACTLY as it appears in the brackets — no path prefixes, no extra punctuation.
- The `(SOURCE: ...)` tag must be the last thing in your reply, on the same line as the final sentence.

**Do NOT include `(SOURCE: ...)`** only when the reply is:
- A greeting ("Hello, how can I help you?")
- A polite closure ("Glad I could help")
- An "I don't know" / insufficient-information response
- Pure small-talk with no factual content drawn from the documents

When in doubt — if the documents contain anything relevant to what you said — include the SOURCE tag.

---

**Example 1 (single source):**
Q: What is the refund policy for product X?
A: Refunds are available within 30 days of purchase. (SOURCE: policy_manual.pdf)

**Example 2 (multiple sources):**
Q: What is a transformer model?
A: A Transformer model is a deep learning model designed primarily for handling sequential data and has become the foundation for state-of-the-art models in natural language processing. (SOURCE: Practical Natural Language Processing.pdf | llm_wiki.txt)

**Example 3 (no source — pure greeting):**
Q: Hello!
A: Hello! How can I assist you today?
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_tokens(text):
    """Counts number of tokens from text using the cached tokenizer."""
    tokens = _ENCODER.encode(json.dumps(text))
    ntokens = len(tokens)
    current_app.logger.info(f"🧮 Number of tokens: {ntokens}")
    return ntokens


def filter_messages(messages):
    """Removes past conversation turns when total tokens exceed the limit."""
    while count_tokens(messages) > cfg.TOKEN_LIMIT and len(messages) > 3:
        messages.pop(1)     # Remove old question
        messages.pop(1)     # Remove old answer
        current_app.logger.info(f"✂️ Dropped a conversation pair (token limit exceeded).")
    return messages


def _build_document_context(docs):
    """Format retrieved docs into a single system-message context string."""
    if not docs:
        return ""
    parts = []
    for d in docs:
        file_name = d.get("file_name", "unknown")
        heading = d.get("heading", "")
        content = (d.get("content", "") or "").strip()
        header = f"[{file_name}]"
        if heading:
            header += f" — {heading}"
        parts.append(f"{header}\n\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def chat_completion(question, objective, index_name, session_id):
    """
    Orchestrates the full RAG turn:
      1. Load history
      2. Rewrite query (standalone) using history
      3. Hybrid retrieve (vector + BM25) → RRF → cross-encoder rerank
      4. Compose messages and call the LLM
      5. Persist history (trimmed)
    """
    # Start a per-request trace bound to flask.g
    trace = RequestTrace.start(session_id=session_id, index_name=index_name, question=question)
    current_app.logger.info(f"💬 [{trace.request_id}] chat_completion() called")

    response = None
    try:
        # 1) Load history
        redis_obj = RedisDB()
        messages = redis_obj.get_conversation_history(session_id)
        if messages is None:
            current_app.logger.info(f"🆕 [{trace.request_id}] New session — initializing conversation.")
            messages = [
                {"role": "system", "content": objective},
            ]

        # 2) Rewrite the query + classify intent (for retrieval gating)
        with trace.timer("rewrite"):
            rewrite_result = rewrite_query(question, messages)
        search_query = rewrite_result.query
        trace.set("rewritten_query", search_query)
        trace.set("intent", rewrite_result.intent)
        trace.set("needs_retrieval", rewrite_result.needs_retrieval)
        if search_query != question:
            current_app.logger.info(f"✍️ [{trace.request_id}] rewrote → {search_query!r}")

        # 3) Hybrid retrieve → RRF → rerank (skipped for greetings / smalltalk / meta)
        if rewrite_result.needs_retrieval:
            docs = retrieve(search_query, index_name)
        else:
            current_app.logger.info(
                f"⏭️ [{trace.request_id}] skipping retrieval (intent={rewrite_result.intent})"
            )
            docs = []
        document_context = _build_document_context(docs)

        # 4) Compose messages and call the LLM..
        insert_pos = len(messages)
        if document_context:
            messages.insert(insert_pos, {"role": "system", "content": document_context})
            messages.insert(insert_pos + 1, {"role": "system", "content": _INSTRUCTIONS})
            n_transient = 2
        else:
            messages.insert(insert_pos, {"role": "system", "content": _INSTRUCTIONS})
            n_transient = 1
        messages.append({"role": "user", "content": question})

        with trace.timer("llm"):
            openai_obj = AzureOpenAIChat()
            response, token_usage = openai_obj.chat_completion(messages)
        trace.set_token_usage(cfg.MODEL, token_usage)
        current_app.logger.info(
            f"🗣️ [{trace.request_id}] LLM reply ({len(response or '')} chars): {response!r}"
        )

        # 5) Drop the transient system messages, append assistant turn, persist.
        del messages[insert_pos:insert_pos + n_transient]
        messages.append({"role": "assistant", "content": response})
        messages = filter_messages(messages)
        redis_obj.save_conversation_history(session_id, messages)

        trace.finish(status="ok")
        return response
    except Exception as e:
        current_app.logger.exception(f"❌ [{trace.request_id}] chat_completion failed")
        trace.finish(status="error", error=str(e))
        raise
