from flask import Flask

from config import BaseConfig
from src.controllers import register_blueprints


def create_app(config_class=BaseConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    register_blueprints(app)
    return app
