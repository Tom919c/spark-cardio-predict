from pathlib import Path

from flask import Flask, request

from config import BaseConfig
from src.controllers import register_blueprints
from src.utils.response import error_response


def create_app(config_class=BaseConfig):
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_object(config_class)

    register_blueprints(app)

    @app.errorhandler(404)
    def handle_not_found(error):
        if request.path.startswith("/api/"):
            return error_response(message="API 接口不存在，请确认后端服务已重启并使用正确地址。", code=404)
        return error

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        if request.path.startswith("/api/"):
            return error_response(message="当前 API 不支持该请求方法。", code=405)
        return error

    return app
