from flask import Flask
from flask_restx import Api
import logging
from project.controller import GenerativeAIChat

app = Flask(__name__)

app.logger.setLevel(logging.INFO)
app.logger.info("Welcome to Generative AI Chat!")

# Create an instance of the Flask-RESTX API
api = Api(app, version='1.1', title='Generative AI Chat', description='Generative AI Chat')

# Create a Flask-RESTX Namespace
genai_namespace = api.namespace('api/v1', description='')

# Add resources to the API
genai_namespace.add_resource(GenerativeAIChat, '/chat')

if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True)
