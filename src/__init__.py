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

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "font-src 'self' data:; connect-src 'self'; frame-ancestors 'self'",
        )
        return response

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

    @app.errorhandler(413)
    def handle_payload_too_large(error):
        if request.path.startswith("/api/"):
            return error_response(message="单次请求超过分片大小限制，请减小上传分片。", code=413)
        return error

    return app
