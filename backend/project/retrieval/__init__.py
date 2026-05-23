# Intentionally empty.
#
# Submodules (pipeline, query_rewriter, reranker, rrf, query_utils) must be
# imported explicitly. Eagerly importing them here creates a circular import
# because `pipeline` depends on `project.vector_db.redis_db`, which itself
# imports `project.retrieval.query_utils` — and that triggers this __init__.py.
