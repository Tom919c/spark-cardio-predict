from flask import Blueprint

from src.services.risk_service import RiskService
from src.utils.response import success_response

risk_bp = Blueprint("risk", __name__)


@risk_bp.route("/summary")
def risk_summary():
    service = RiskService()
    summary = service.get_platform_summary()
    return success_response(message="风险平台摘要获取成功。", data=summary)
