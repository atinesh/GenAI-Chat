from redis.commands.search.query import Query
import redis
from flask import current_app
from openai import OpenAI
import numpy as np
import json
import os
import config as cfg

from project.retrieval.query_utils import build_bm25_query

# from dotenv import load_dotenv
# load_dotenv()


# Fields returned from RediSearch (kept in one place so both retrievers agree).
_RETURN_FIELDS = (
    "file_name",
    "chunk",
    "source",
    "tag",
    "content",
    "heading",
    "prev_id",
    "next_id",
    "vector_score",
)


class RedisDB:

    def __init__(self):
        """
        Initializes Redis and OpenAI client
        """
        self.redis_conn = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
        self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(
            input=[text], model=cfg.EMBEDDING_MODEL
        ).data[0].embedding

    # ------------------------------------------------------------------
    # Vector KNN search
    # ------------------------------------------------------------------
    def vector_search(self, question, index_name, top_k=None):
        """
        Vector KNN search. Returns list of RediSearch Document objects.
        Each doc has `.id` (the Redis key) plus the fields in _RETURN_FIELDS.
        """
        top_k = top_k or cfg.VECTOR_TOP_K
        current_app.logger.info(f"🧭 [RedisDB] vector_search(top_k={top_k}) on '{index_name}'")

        result = []
        try:
            query_vector = np.array(self.get_embedding(question), dtype=np.float32).tobytes()

            if cfg.INDEX_TYPE == "FLAT":
                query = (
                    Query(f"(*)=>[KNN {top_k} @vector $vec AS vector_score]")
                    .sort_by("vector_score")
                    .return_fields(*_RETURN_FIELDS)
                    .dialect(2)
                )
                params = {"vec": query_vector}
            elif cfg.INDEX_TYPE == "HNSW":
                query = (
                    Query(f"(*)=>[KNN {top_k} @vector $vec EF_RUNTIME $ef AS vector_score]")
                    .sort_by("vector_score")
                    .return_fields(*_RETURN_FIELDS)
                    .dialect(2)
                )
                params = {"vec": query_vector, "ef": cfg.EF_RUNTIME}
            else:
                raise ValueError("Invalid Index Type Passed")

            result = self.redis_conn.ft(index_name).search(query, params).docs
        except Exception as e:
            current_app.logger.error(f"❌ [RedisDB] vector_search: {e}")

        return result

    # ------------------------------------------------------------------
    # BM25 full-text search
    # ------------------------------------------------------------------
    def bm25_search(self, question, index_name, top_k=None):
        """
        BM25 keyword search over the `content` text field. Returns list of
        RediSearch Document objects. Each doc has `.id` + _RETURN_FIELDS.
        """
        top_k = top_k or cfg.BM25_TOP_K
        bm25_q = build_bm25_query(question)
        current_app.logger.info(f"🔤 [RedisDB] bm25_search(top_k={top_k}) q={bm25_q!r}")

        if not bm25_q:
            return []

        try:
            # WITHSCORES toggles BM25 scoring; LIMIT controls top_k.
            query = (
                Query(bm25_q)
                .with_scores()
                .paging(0, top_k)
                .return_fields(*_RETURN_FIELDS)
                .dialect(2)
            )
            return self.redis_conn.ft(index_name).search(query).docs
        except Exception as e:
            current_app.logger.error(f"❌ [RedisDB] bm25_search: {e}")
            return []

    # ------------------------------------------------------------------
    # Lookup by Redis key (used to materialise RRF-fused doc IDs)
    # ------------------------------------------------------------------
    def get_docs_by_ids(self, doc_ids):
        """
        Fetch full hash payloads for a list of Redis keys, preserving the input order.
        Returns list of dicts with the same keys as _RETURN_FIELDS (minus vector_score).
        """
        out = []
        if not doc_ids:
            return out
        try:
            pipe = self.redis_conn.pipeline()
            for did in doc_ids:
                pipe.hgetall(did)
            results = pipe.execute()
            for did, raw in zip(doc_ids, results):
                if not raw:
                    continue
                doc = {"id": did}
                for k, v in raw.items():
                    key = k.decode() if isinstance(k, bytes) else k
                    # The `vector` field is a raw float32 blob — not UTF-8 text.
                    # Decoding it would raise UnicodeDecodeError and abort the whole
                    # document, silently producing 0 retrieved docs downstream.
                    if key == "vector":
                        continue
                    if isinstance(v, bytes):
                        try:
                            v = v.decode("utf-8")
                        except UnicodeDecodeError:
                            continue
                    doc[key] = v
                out.append(doc)
        except Exception as e:
            current_app.logger.error(f"❌ [RedisDB] get_docs_by_ids: {e}")
        return out

    # ------------------------------------------------------------------
    # Backward-compatible single-shot vector-only search (used by older callers)
    # ------------------------------------------------------------------
    def search_index(self, question, index_name):
        """Legacy entry point. Prefer vector_search + bm25_search via the retrieval pipeline."""
        return self.vector_search(question, index_name, top_k=cfg.RERANK_TOP_N)

    # ------------------------------------------------------------------
    # Session conversation history
    # ------------------------------------------------------------------
    def save_conversation_history(self, session_id, messages):
        serialized_data = json.dumps(messages)
        result = self.redis_conn.setex(session_id, cfg.SESSION_EXPIRATION, serialized_data)
        if result:
            current_app.logger.info(f"💾 Session {session_id} data saved successfully.")
        else:
            current_app.logger.info(f"⚠️ Unable to save Session {session_id} data.")

    def get_conversation_history(self, session_id):
        serialized_data = self.redis_conn.get(session_id)
        if serialized_data is None:
            return None
        current_app.logger.info(f"📥 Session {session_id} data retrieved.")
        return json.loads(serialized_data)
