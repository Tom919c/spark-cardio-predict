from pathlib import Path

from flask import Flask

from config import BaseConfig
from src.controllers import register_blueprints


def create_app(config_class=BaseConfig):
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_object(config_class)

    register_blueprints(app)
    return app
