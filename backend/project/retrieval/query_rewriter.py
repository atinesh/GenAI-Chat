"""
Standalone-query rewriter + intent classifier.

In one LLM call, this:
  1. Rewrites the user's latest question into a self-contained search query.
  2. Classifies the intent (chit-chat vs. real knowledge question) so the
     caller can skip vector / BM25 retrieval for pure social messages
     ("hi", "thanks", "got it") that don't need document context.

A cheap regex prefilter short-circuits the most obvious chit-chat cases
without hitting the LLM at all.
"""

from __future__ import annotations

import os
import re
import json
from typing import List, NamedTuple
from flask import current_app
from openai import OpenAI
import config as cfg


class RewriteResult(NamedTuple):
    query: str             # rewritten (or original) query for retrieval
    needs_retrieval: bool  # False for greetings / thanks / acks / meta
    intent: str            # greeting | smalltalk | meta | knowledge_query | followup


_REWRITE_SYSTEM_PROMPT = """
You serve two roles for a RAG chatbot:
  (a) classify the user's latest message
  (b) rewrite it into a self-contained search query

Intent labels:
- greeting        — "hi", "hello there", "good morning"
- smalltalk       — "thanks", "got it", "ok cool", "bye"
- meta            — questions about the bot itself ("who are you", "what can you do")
- knowledge_query — a real question that should be answered from documents
- followup        — a follow-up to a prior turn that needs pronoun/ellipsis resolution

Set needs_retrieval=true ONLY for knowledge_query and followup.
Set needs_retrieval=false for greeting / smalltalk / meta.

Rewriting rules (only matters when needs_retrieval=true):
- Resolve pronouns ("it", "they", "this", "that") and elliptical follow-ups using the prior turns.
- Make implicit subjects explicit (e.g. "what about the deductible?" → "What is the deductible amount in the health insurance policy?").
- Keep the rewrite SHORT — one question, no preamble, no quotes.
- If the latest question is already self-contained, return it unchanged.
- When needs_retrieval=false, set rewritten_query to the original message verbatim.

Return STRICT JSON only, no prose, no code fences:
{"intent": "...", "needs_retrieval": true|false, "rewritten_query": "..."}
"""


# Compile once. Matches messages that are clearly pure chit-chat — short,
# made up of common greetings/thanks/acks with optional punctuation.
_CHIT_CHAT_RE = re.compile(
    r"^\s*("
    r"hi|hii+|hey+|hello+|yo|sup|"
    r"good\s*(morning|afternoon|evening|day)|"
    r"thanks?|thank\s*you|thx|ty|"
    r"ok|okay|okey|kk|cool|nice|great|awesome|perfect|got\s*it|understood|"
    r"bye|goodbye|see\s*you|cya|"
    r"yes|yeah|yep|no|nope|sure"
    r")"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)


# Cache the OpenAI client at module level — same key the rest of the app uses.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _extract_recent_turns(messages: List[dict], max_turns: int = 4) -> List[dict]:
    """
    Pull the last `max_turns` user/assistant turns from the stored message list,
    skipping system messages. Used as context for the rewriter.
    """
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    return convo[-max_turns:]


def _prefilter_chit_chat(question: str) -> bool:
    """Cheap regex check for obvious greetings/thanks/acks."""
    if len(question) > 40:
        return False
    return bool(_CHIT_CHAT_RE.match(question))


def rewrite_query(question: str, history_messages: list | None) -> RewriteResult:
    """
    Classify intent and (if needed) rewrite the query into a standalone form.

    Safe defaults: on any error or when the feature is disabled, returns the
    original question with needs_retrieval=True so the pipeline behaves
    exactly like it did before this change.
    """
    classify = cfg.INTENT_CLASSIFICATION_ENABLED

    # Fast path: obvious chit-chat — skip the LLM entirely.
    if classify and _prefilter_chit_chat(question):
        return RewriteResult(query=question, needs_retrieval=False, intent="smalltalk")

    if not cfg.QUERY_REWRITE_ENABLED and not classify:
        return RewriteResult(query=question, needs_retrieval=True, intent="knowledge_query")

    # First turn with no history: nothing to rewrite from, but we still want
    # classification so a lone "hi" doesn't trigger retrieval.
    recent = _extract_recent_turns(history_messages) if history_messages else []

    try:
        msgs = [{"role": "system", "content": _REWRITE_SYSTEM_PROMPT}]
        msgs.extend(recent)
        msgs.append({
            "role": "user",
            "content": (
                "Classify and rewrite this message. Return JSON only.\n\n"
                f"Message: {question}"
            ),
        })

        client = _get_client()
        resp = client.chat.completions.create(
            messages=msgs,
            model=cfg.QUERY_REWRITE_MODEL,
            max_completion_tokens=500,
            reasoning_effort=cfg.REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)

        intent = str(data.get("intent", "knowledge_query"))
        needs_retrieval = bool(data.get("needs_retrieval", True))
        rewritten = str(data.get("rewritten_query", "")).strip() or question

        # Strip stray quotes the model sometimes wraps around the query.
        if rewritten.startswith(("\"", "'")) and rewritten.endswith(("\"", "'")):
            rewritten = rewritten[1:-1].strip() or question

        # If classification is disabled, force retrieval on regardless of label.
        if not classify:
            needs_retrieval = True

        return RewriteResult(query=rewritten, needs_retrieval=needs_retrieval, intent=intent)
    except Exception as e:
        current_app.logger.warning(
            f"⚠️ rewrite_query failed: {e}; falling back to original question with retrieval=on"
        )
        return RewriteResult(query=question, needs_retrieval=True, intent="knowledge_query")
