"""个人风险评估历史 API。"""

from flask import Blueprint, current_app, request

from src.services.assessment_record_service import AssessmentRecordService
from src.utils.response import error_response, success_response


assessment_bp = Blueprint("assessment", __name__)


@assessment_bp.route("")
def list_assessments():
    try:
        limit = int(request.args.get("limit", 10))
        records = AssessmentRecordService(current_app.config).list_for_client(
            request.headers.get("X-Client-ID"), limit=limit
        )
    except ValueError as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="个人评估历史获取成功。", data={"items": records})


@assessment_bp.route("/<assessment_id>")
def get_assessment(assessment_id):
    try:
        record = AssessmentRecordService(current_app.config).get_for_client(
            assessment_id, request.headers.get("X-Client-ID")
        )
    except ValueError as exc:
        return error_response(message=str(exc), code=400)
    except FileNotFoundError as exc:
        return error_response(message=str(exc), code=404)
    return success_response(message="个人评估记录获取成功。", data=record)
