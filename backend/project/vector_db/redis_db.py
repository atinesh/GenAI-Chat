from redis.commands.search.query import Query
import redis
from flask import current_app
from openai import OpenAI
import numpy as np
import json
import os
import config as cfg

# from dotenv import load_dotenv
# load_dotenv()

class RedisDB:

    def __init__(self):
        """
        Initializes Redis and OpenAI client
        """

        self.redis_conn = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
        self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def get_embedding(self, text):
        """
        Gets embedding vector from OpenAI model
        """
        
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=[text], model=cfg.EMBEDDING_MODEL).data[0].embedding
        
    def search_index(self, question, index_name):
        """
        Fetch relevant documents from Redis based on user question

        Args:
            question (str): User Query.
            index_name (str): Index name from where Redis will fetch the document chunks.

        Returns:
            result (list): List of relevant documents.
        """

        current_app.logger.info(f"[RedisDB] search_index() called")

        result = []

        try:

            # create query embedding
            query_vector = np.array(self.get_embedding(question), dtype=np.float32).tobytes()
            
            # Create redis query
            query = (
                Query(f"(*)=>[KNN {cfg.TOP_K} @vector $vec AS vector_score]")
                .sort_by("vector_score")
                .return_fields("file_name", "chunk", "source", "tag", "content", "vector_score")
                .dialect(2)
            )
            # Search index
            query_params = {"vec": query_vector}
            result = self.redis_conn.ft(index_name).search(query, query_params).docs

        except Exception as e:
            current_app.logger.error(str(e))

        return result

    def save_conversation_history(self, session_id, messages):
        """
        Store conversation history in Redis with a 2-hour expiration time.
        
        Args:
            session_id (str): Unique session ID.
            messages (list): The data to store in the session.
        """
        serialized_data = json.dumps(messages)  # Convert list to JSON string
        result = self.redis_conn.setex(session_id, cfg.SESSION_EXPIRATION, serialized_data)

        if result:
            current_app.logger.info(f"Session {session_id} data saved successfully.")
        else:
            current_app.logger.info(f"Unable to save Session {session_id} data.")

    def get_conversation_history(self, session_id):
        """
        Retrieve conversation history from Redis.

        Args:
            session_id (str): Unique session ID.

        Returns:
            messages (list): Returns message object containing conversation history and objective.
        """
        serialized_data = self.redis_conn.get(session_id)
        if serialized_data is None:
            return None

        current_app.logger.info(f"Session {session_id} data retrieved.")

        return json.loads(serialized_data)  # Convert JSON string back to list
