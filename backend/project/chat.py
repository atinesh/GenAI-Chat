import tiktoken
import json
from flask import current_app
from project.vector_db.redis_db import RedisDB
from project.llm.azure_openai import AzureOpenAIChat
import config as cfg


def count_tokens(text):
    """
    Counts number of token from text.

    Args:
        text (str): Text.

    Returns:
        ntokens (int): Number of token.
    """

    # Load the encoding
    encoding = tiktoken.get_encoding("cl100k_base")
    # encoding = tiktoken.encoding_for_model("gpt-4o")

    # Encode the text and count the tokens
    tokens = encoding.encode(json.dumps(text))

    ntokens = len(tokens)
    current_app.logger.info(f"Number of tokens: {ntokens}")

    return ntokens

def filter_messages(messages):
    """
    Removes past conversations from the message object based on the token limit.

    Args:
        messages (list): List of messages.

    Returns:
        messages (list): List of messages.
    """

    while count_tokens(messages) > cfg.TOKEN_LIMIT and len(messages) > 3:
        messages.pop(1)     # Remove old question
        messages.pop(1)     # Remove old answer
        current_app.logger.info(f"Dropped a conversation as messages tokens exceeds token Limit.")

    return messages

def get_document_context(question, index_name):
    """
    Fetch relevant documents from Redis based on user question, then combine document texts to form Document context.

    Args:
        question (str): User Query.
        index_name (str): Index name from where Redis will fetch the document chunks.

    Returns:
        document_context (str): Document context prepared from most relevant docs.
    """

    # Get data from Redis

    redis_obj = RedisDB()
    docs = redis_obj.search_index(question, index_name)
    current_app.logger.info(f"Redis docs found: {len(docs)}")

    # Prepare document context
    document_context = ""
    for doc in docs:
        document_context += f"\n\n[{doc.file_name}]\n\n{doc.content.strip()}"

    return document_context

def chat_completion(question, objective, index_name, session_id):
    """
    Create message object with prompts, context and user quyestion calls LLMS to generate response.

    Args:
        question (str): User Query.
        objective (str): Instructions for model.
        index_name (str): Index name from where Redis will fetch the document chunks.
        session_id (str): Session ID.

    Returns:
        response (str): Generated Response.
    """

    current_app.logger.info("chat_completion() called")

    instructions = """
    
    Your goal is to generate accurate and clear responses using the provided context. You will be provided with the document context where each document context is accompanied by a source filename example [policy_manual.pdf] followed by document text.

    Only include the source filename(s) if the answer is **substantively derived** from the provided document context. Specifically show source filename(s) if:
    - The answer includes factual details, definitions, statistics, procedures, or any information that is clearly taken or paraphrased from the document.
    - The answer uses specific terms, quotes, or concepts from the document.
    - If the answer is derived from multiple documents then show all the source filenames separated by | symbol.

    Do **not** show source filename(s) if the answer is:
    - A general greeting (e.g., “Hello, how can I help you?”)
    - A polite closure or thank-you message (e.g., “Glad I could help”)
    - A clarification or error message due to insufficient information (e.g., “I'm sorry, I couldn't find the answer to that question”)
    - A personal opinion or subjective statement (e.g., “In my opinion, this is the best option”)

    ---

    **Example 1: With Source**

    Question: What is the refund policy for product X?
    Answer: Refunds are available within 30 days of purchase. (SOURCE: policy_manual.pdf)

    ---

    **Example 1: With Source**

    Question: What is transformer model ?
    Answer: A Transformer model is a deep learning model, it is designed primarily for handling sequential data and has become the foundation for state-of-the-art models in natural language processing. (SOURCE: Practical Natural Language Processing.pdf | llm_wiki.txt)

    ---

    **Example 2: Without Source**

    Question: Hello!
    Answer: Hello! How can I assist you today?

    """

    # Get conversation history
    redis_obj = RedisDB()
    messages = redis_obj.get_conversation_history(session_id)

    if messages is None:
        current_app.logger.info(f"Conversation history not found, Initializing the conversation.")
        messages = [{"role": "system", "content": objective}, {"role": "system", "content": instructions}]

    # Get document context
    current_app.logger.info(f"Retrieving Document Context from Redis ...")
    document_context = get_document_context(question, index_name)

    # Update message object
    messages.insert(2, {"role": "system", "content": document_context})
    messages.append({"role": "user", "content": question})
    current_app.logger.info(f"messages: {messages}")

    # Call OpenAI
    openai_obj = AzureOpenAIChat()
    response, token_usage = openai_obj.chat_completion(messages)
    current_app.logger.info(f"response: {response}")

    # Save conversation history
    messages.pop(2) # remove document context
    messages.append({"role": "assistant", "content": response})
    messages = filter_messages(messages)
    redis_obj.save_conversation_history(session_id, messages)

    return response