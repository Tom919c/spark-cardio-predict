from flask import Blueprint

from src.utils.response import success_response

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/status")
def index():
    return success_response(
        message="心脑血管疾病风险预测与分析平台后端骨架已启动。",
        data={
            "system": "cardio-cerebrovascular-risk-platform",
            "stage": "phase-2",
        },
    )


@health_bp.route("/health")
def health_check():
    return success_response(message="服务运行正常。")
