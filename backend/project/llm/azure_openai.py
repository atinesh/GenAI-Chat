from flask import current_app
from openai import OpenAI
import os
import config as cfg

# from dotenv import load_dotenv
# load_dotenv()

class AzureOpenAIChat:

    def __init__(self):
        """
        Initializes the LLM model 
        """

        self.openai_client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

    def chat_completion(self, messages):
        """
        Takes message and generates response by calling OpenAI model.

        Args:
            messages (list): Contains conversation history, document context and user query.

        Returns:
            response_text (str): Generated Response.
            token_usage (int): Number of tokens consumed while generating the response.
        """

        current_app.logger.info(f"[OpenAI] chat_completion() called")

        response_text = None
        token_usage = 0

        try:
            response = self.openai_client.chat.completions.create(
                messages = messages,
                model = cfg.MODEL,
                temperature = 0,
                max_tokens = cfg.TOKEN_LIMIT,
            )
            response_text = response.choices[0].message.content
            token_usage = response.usage
        except Exception as e:
            current_app.logger.error(str(e))

        return response_text, token_usage
