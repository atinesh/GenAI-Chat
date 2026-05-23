import os

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_HOST = "redis-stack"
# REDIS_HOST = "localhost"          # LOCAL TESTING
REDIS_PORT = 6379

INDEX_TYPE = os.environ.get("INDEX_TYPE", "HNSW")        # FLAT or HNSW
EF_RUNTIME = int(os.environ.get("EF_RUNTIME", "64"))     # HNSW: query-time candidate pool. Must be >= VECTOR_TOP_K.

# ---------------------------------------------------------------------------
# Session / history
# ---------------------------------------------------------------------------
SESSION_EXPIRATION = 900            # Session data to be removed from Redis after 900 secs / 15 mins
TOKEN_LIMIT = 5000                  # Maximum token limit for stored conversation history (input side)
MAX_OUTPUT_TOKENS = 1500            # Max tokens for LLM completion response

# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------
# Stage 1: candidate pool from each retriever
VECTOR_TOP_K = 20                   # Top-K from vector KNN
BM25_TOP_K = 20                     # Top-K from BM25 keyword search
# Stage 2: after Reciprocal Rank Fusion
RRF_K = 60                          # RRF constant (60 is the canonical default)
RRF_TOP_K = 20                      # Top-K kept after fusion (fed to reranker)
# Stage 3: after cross-encoder reranker
RERANK_TOP_N = 5                    # Final docs passed to the LLM
# Optional: include neighbor chunks (prev/next) of each top doc
INCLUDE_NEIGHBORS = False

# Feature flags
HYBRID_SEARCH_ENABLED = True
RERANKER_ENABLED = True
QUERY_REWRITE_ENABLED = True
INTENT_CLASSIFICATION_ENABLED = True   # When True, rewriter also classifies chit-chat and retrieval is skipped for those turns

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL = "gpt-5.5"                               # Model for Query response
REASONING_EFFORT = "low"                        # LLM reasoning effort (none, minimal, low, medium, high, and xhigh) — affects latency and cost
QUERY_REWRITE_MODEL = "gpt-5.4-mini"            # Cheap model for standalone-query rewriting
EMBEDDING_MODEL = "text-embedding-3-large"      # text-embedding-3-small (1536); text-embedding-3-large (3072)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"      # HF cross-encoder reranker
RERANKER_PROVIDER = "hf-inference"              # HuggingFace InferenceClient provider

# Tokenizer encoding for the answering MODEL.
# New gpt series models uses o200k_base, older models use cl100k_base.
TOKENIZER_ENCODING = "o200k_base"

# Rough cost table ($/1K tokens) — used only for estimated_cost in trace.
# Adjust as prices change. Set to {} to disable.
COST_PER_1K = {
    "gpt-5.5":       {"in": 0.005,   "out": 0.030},
    "gpt-5.4-mini":  {"in": 0.00075, "out": 0.0045},
}
