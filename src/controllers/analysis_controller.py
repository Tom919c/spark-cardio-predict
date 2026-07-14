"""群体分析控制器：大屏聚合统计和高危随访名单接口。"""

from flask import Blueprint, current_app, request

from src.services.institution_data_service import InstitutionDataService
from src.utils.response import error_response, success_response

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/dashboard")
def dashboard_analysis():
    data = InstitutionDataService(current_app.config).dashboard()
    return success_response(message="群体健康分析获取成功。", data=data)


@analysis_bp.route("/follow-ups")
def follow_up_analysis():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
    data = InstitutionDataService(current_app.config).follow_ups(limit=limit)
    return success_response(message="重点随访名单获取成功。", data=data)
