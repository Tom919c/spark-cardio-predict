from flask import Blueprint, current_app

from src.services.platform_capability_service import PlatformCapabilityService
from src.utils.response import success_response

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/status")
def index():
    return success_response(
        message="心脑血管疾病风险预测与分析平台后端骨架已启动。",
        data={
            "system": "cardio-cerebrovascular-risk-platform",
            "stage": "part-1-foundation",
        },
    )


@health_bp.route("/health")
def health_check():
    return success_response(message="服务运行正常。")


@health_bp.route("/api/capabilities")
def capabilities():
    return success_response(
        message="平台运行能力获取成功。",
        data=PlatformCapabilityService(current_app.config).inspect(),
    )
