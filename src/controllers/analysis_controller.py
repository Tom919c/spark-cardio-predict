"""Institutional dashboard APIs backed by the Chengdu simulated population."""

from flask import Blueprint, current_app, request

from src.services.population_service import PopulationService
from src.utils.response import error_response, success_response


analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/dashboard")
def dashboard_analysis():
    try:
        data = PopulationService(current_app.config).dashboard()
    except FileNotFoundError as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="群体健康分析获取成功。", data=data)


@analysis_bp.route("/follow-ups")
def follow_up_analysis():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
    try:
        data = PopulationService(current_app.config).follow_up_list(limit=limit)
    except FileNotFoundError as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="重点随访名单获取成功。", data=data)
