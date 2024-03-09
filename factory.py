from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register your blueprints or import your routes here
    from routes.prompt_embedding import prompt_embedding
    app.register_blueprint(prompt_embedding)
    return app