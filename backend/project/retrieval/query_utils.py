"""
Utilities for building RediSearch full-text queries safely.

RediSearch reserves a number of punctuation characters; if they appear in user
text they must be escaped with a backslash or the parser fails. We also strip
single-character tokens (RediSearch ignores them by default) and de-duplicate.
"""

import re


# Characters that have special meaning in RediSearch queries
_RESERVED = r',.<>{}[]"\':;!@#$%^&*()-+=~|/\\?'


def escape_redisearch(text: str) -> str:
    """Backslash-escape reserved RediSearch characters in `text`."""
    if not text:
        return ""
    return "".join("\\" + c if c in _RESERVED else c for c in text)


def build_bm25_query(question: str) -> str:
    """
    Build a RediSearch full-text query targeting the `content` field.

    Strategy: tokenize on whitespace, drop very short tokens, OR them together
    inside a `@content:(...)` clause. We use OR (default) so partial keyword
    matches still surface; BM25 ranks them.
    """
    if not question:
        return ""

    # Lowercase + simple word tokenization (alphanumeric + underscore)
    tokens = re.findall(r"[A-Za-z0-9_]+", question.lower())
    # Drop noise: 1-char tokens and obvious stopwords (small list — Redis has its own too)
    tokens = [t for t in tokens if len(t) > 1]

    if not tokens:
        return ""

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    # Escape each token (defensive — they're already alnum, but keep symmetric)
    escaped = [escape_redisearch(t) for t in uniq]

    # OR query — RediSearch uses '|' for union inside parens
    return "@content:(" + " | ".join(escaped) + ")"
