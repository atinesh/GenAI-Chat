from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import tiktoken
import json
from flask import current_app
from project.vector_db.redis_db import RedisDB
from project.llm.azure_openai import AzureOpenAIChat
import config as cfg

# Load the saved model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("model")
tokenizer = DistilBertTokenizer.from_pretrained("model")
# current_app.logger.info(f"Intent detection model is loaded successfully.")

# Set the model to evaluation mode
model.eval()

# Set device to GPU or CPU depending on availability
device = "cpu"
model.to(device)

# Intent Prediction
def predict_intent(text):
    """
    Predicts intent of LLM response, intent can be either Greet Intent, Thanks Intent or Bad Response Intent.

    Args:
        text (str): LLM response.

    Returns:
        intent_detected (bool): Whether any of the intent is detected or not.
    """

    intent_map = {0: "Greet Intent", 1: "Thanks Intent", 2: "Bad Response Intent"}
    intent_detected = True

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    max_prob, predicted_label = torch.max(probs, dim=-1)
    current_app.logger.info(f"Predicted Intent: {intent_map[predicted_label.item()]}, Probability {max_prob.item()}")
    
    if max_prob.item() < cfg.INTENT_THRESH:
        intent_detected = False 
    
    return intent_detected

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
        current_app.logger.info(f"Dropped 1 conversation turn as messages tokens exceeds token Limit.")

    return messages

def get_document_context(question, index_name):
    """
    Fetch relevant documents from Redis based on user question, then combine document texts to form Document context.

    Args:
        question (str): User Query.
        index_name (str): Index name from where Redis will fetch the document chunks.

    Returns:
        document_text (str): Document context prepared from most relevant docs.
        document_names (str): Document names from which Document context is prepared.
    """

    # Get data from Redis
    document_context = ""

    redis_obj = RedisDB()
    docs = redis_obj.search_index(question, index_name)
    current_app.logger.info(f"Redis docs found: {len(docs)}")

    doc_data = []
    for doc in docs:
        doc_data.append([doc.file_name, doc.source, doc.content.strip()])

    document_text = "\n\n".join([item[2] for item in doc_data])
    # token_count = count_tokens(document_text)

    document_names = " | ".join(list(set([item[0] for item in doc_data])))

    return document_text, document_names

def chat_completion(question, objective, index_name, session_id, smart_src_filter):
    """
    Create message object with prompts, context and user quyestion calls LLMS to generate response.

    Args:
        question (str): User Query.
        objective (str): Instructions for model.
        index_name (str): Index name from where Redis will fetch the document chunks.
        session_id (str): Session ID.
        smart_src_filter (bool): Whether to use smart source filtering feature.

    Returns:
        response (str): Generated Response.
    """

    current_app.logger.info("chat_completion() called")

    redis_obj = RedisDB()

    messages = redis_obj.get_conversation_history(session_id)
    if messages is None:
        current_app.logger.info(f"Conversation history not found, Initializing the conversation.")
        messages = []
        messages.append({"role": "system", "content": objective})

    # Get document context
    current_app.logger.info(f"Retrieving Document Context from Redis ...")
    document_context, document_names = get_document_context(question, index_name)

    # Create message object
    messages.insert(1, {"role": "system", "content": document_context})
    messages.append({"role": "user", "content": question})
    current_app.logger.info(f"messages: {messages}")

    # Call OpenAI
    openai_obj = AzureOpenAIChat()
    response, token_usage = openai_obj.chat_completion(messages)

    # Save conversation history
    messages.pop(1) # remove document context
    messages.append({"role": "assistant", "content": response})
    messages = filter_messages(messages)
    redis_obj.save_conversation_history(session_id, messages)

    # Add source filenames in response
    if smart_src_filter:
        if not predict_intent(response) and len(document_names) > 0:
            response = response + f" [SOURCE: {document_names}]"
        else:
            response = response + f" [SOURCE: None]"
    else:
        response = response + f" [SOURCE: {document_names}]"

    current_app.logger.info(f"response: {response}")
    
    return response