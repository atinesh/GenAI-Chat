"""
Per-request structured trace + summary log.

Usage:
    trace = RequestTrace.start(session_id=..., index_name=..., question=...)
    trace.set("rewritten_query", "...")
    with trace.timer("vector"):
        ...
    trace.set("vector_hits", [...])
    ...
    trace.finish(status="ok")   # emits 1 JSON line + 1 summary

Stored on flask.g.trace
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional
from flask import current_app, g, has_app_context
import config as cfg


# A dedicated logger for machine-parseable JSON lines (so it can be routed
# to a different handler/file later without touching call sites).
_json_logger = logging.getLogger("genai_chat.trace")
if not _json_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _json_logger.addHandler(handler)
    _json_logger.setLevel(logging.INFO)
    _json_logger.propagate = False


def _estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    p = cfg.COST_PER_1K.get(model)
    if not p:
        return 0.0
    return round((in_tokens / 1000.0) * p["in"] + (out_tokens / 1000.0) * p["out"], 6)


class RequestTrace:
    """Lightweight structured trace for a single /chat request."""

    def __init__(self, session_id: str, index_name: str, question: str):
        self.request_id = "req_" + uuid.uuid4().hex[:8]
        self.t0 = time.perf_counter()
        self.data: Dict[str, Any] = {
            "request_id": self.request_id,
            "session_id": session_id,
            "index_name": index_name,
            "query": question,
            "rewritten_query": None,
            "vector_hits": [],
            "bm25_hits": [],
            "rrf_top": [],
            "rerank_top": [],
            "final_doc_ids": [],
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": 0.0,
            "latency_ms": {},
            "status": "ok",
            "error": None,
        }

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @classmethod
    def start(cls, session_id: str, index_name: str, question: str) -> "RequestTrace":
        t = cls(session_id, index_name, question)
        if has_app_context():
            g.trace = t
        return t

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    @contextmanager
    def timer(self, stage: str):
        """Context manager that records elapsed ms under latency_ms[stage]."""
        s = time.perf_counter()
        try:
            yield
        finally:
            self.data["latency_ms"][stage] = round((time.perf_counter() - s) * 1000, 1)

    def set_token_usage(self, model: str, usage_obj) -> None:
        """`usage_obj` is the OpenAI usage object (has prompt_tokens / completion_tokens / total_tokens)."""
        if usage_obj is None:
            return
        try:
            pt = getattr(usage_obj, "prompt_tokens", None)
            ct = getattr(usage_obj, "completion_tokens", None)
            tt = getattr(usage_obj, "total_tokens", None)
        except Exception:
            return
        self.data["prompt_tokens"] = pt
        self.data["completion_tokens"] = ct
        self.data["total_tokens"] = tt
        if pt is not None and ct is not None:
            self.data["estimated_cost_usd"] = _estimate_cost(model, pt, ct)

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.data["status"] = status
        self.data["error"] = error
        self.data["latency_ms"]["total"] = round((time.perf_counter() - self.t0) * 1000, 1)

        # 1) machine-parseable JSON line
        try:
            _json_logger.info(json.dumps(self.data, default=str))
        except Exception:
            pass

        # 2) summary on the Flask app logger
        try:
            d = self.data
            status_emoji = "✅" if status == "ok" else "❌"
            short_q = (d["query"] or "")[:80]
            short_rq = d.get("rewritten_query") or ""
            short_rq = short_rq[:80] if short_rq else ""
            n_final = len(d.get("final_doc_ids") or [])
            total_ms = d["latency_ms"].get("total", 0)
            pt = d.get("prompt_tokens") or 0
            ct = d.get("completion_tokens") or 0
            cost = d.get("estimated_cost_usd") or 0.0

            parts = [
                f"{status_emoji} [{d['request_id']}]",
                f"🙋 \"{short_q}\"",
            ]
            if short_rq and short_rq != short_q:
                parts.append(f"✍️ \"{short_rq}\"")
            parts += [
                f"📄 {n_final} docs",
                f"⏱️ {total_ms}ms",
                f"🧮 in={pt} out={ct}",
            ]
            if cost:
                parts.append(f"💰 ${cost}")
            if error:
                parts.append(f"⚠️ {error}")

            current_app.logger.info(" ".join(parts))
        except Exception:
            # Never let logging crash a request
            pass


def current_trace() -> Optional[RequestTrace]:
    """Return the current request's trace, or None if not in a request context."""
    if has_app_context():
        return getattr(g, "trace", None)
    return None
