from flask import request
from flask import current_app
from flask_restx import Resource
import project.chat as chat

class GenerativeAIChat(Resource):
    def post(self):
        data = request.get_json()
        current_app.logger.info(data)

        question: str   = data['question']
        objective: str  = data['objective']
        index_name: str = data['index']
        session_id: str = data['session_id']
        smart_src_filter: bool = data['smart_src_filter']

        if len(question) > 2000 or len(objective) > 2000 or len(index_name) > 100:
            current_app.logger.info(f"Character limit exceeded.")
            return None, 500

        result = None
        response_code = 200

        try: 
            result = chat.chat_completion(question, objective, index_name, session_id, smart_src_filter)
        except Exception as e:
            response_code = 500
            current_app.logger.error(str(e))

        return result, response_code