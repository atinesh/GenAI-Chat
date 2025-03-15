REDIS_HOST = "redis-stack"            
# REDIS_HOST = "localhost"          # LOCAL TESTING
REDIS_PORT = 6379
SESSION_EXPIRATION = 900            # Session data to be removed from Redis after 900 secs / 15 mins
TOKEN_LIMIT = 5000                  # Maximum token limit
TOP_K = 2                           # Top K documents to be fetched from Redis
                                    # Size of document text passed in Prompt can be adjusted by modifying TOP_K and CHUNK_SIZE (data_indexing/data_indexing.py)
INTENT_THRESH = 0.7                 # Threshold for deciding whether text belongs to any of the intent (Greet, Thanks or Bad response)
MODEL = "gpt-4o"                                # gpt-4o-mini; gpt-4o
EMBEDDING_MODEL = "text-embedding-3-large"      # text-embedding-3-small (1536); text-embedding-3-large (3072)