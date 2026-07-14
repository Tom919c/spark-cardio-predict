"""阶段二个人情景模拟和批量评估接口。"""

from __future__ import annotations

from flask import Blueprint, current_app, request

from src.services.batch_risk_service import BatchRiskService
from src.services.risk_service import RiskService
from src.utils.response import error_response, success_response


risk_phase2_bp = Blueprint("risk_phase2", __name__)
WHAT_IF_FIELDS = {
    "bmi",
    "cholesterol",
    "diabetes",
    "hypertension",
    "smoker",
    "alcohol",
    "exercise",
}


@risk_phase2_bp.route("/what-if", methods=["POST"])
def risk_what_if():
    payload = request.get_json(silent=True) or {}
    baseline = payload.get("baseline") or {}
    scenario = payload.get("scenario") or {}
    if not baseline or not scenario:
        return error_response(message="baseline 和 scenario 均为必填对象。", code=400)
    required_fields = set(current_app.config.get("CARDIO_FEATURE_COLUMNS", []))
    baseline_fields = set(baseline)
    scenario_fields = set(scenario)
    if baseline_fields != scenario_fields or baseline_fields != required_fields:
        return error_response(
            message="baseline 和 scenario 必须包含完全一致的全部模型特征字段。",
            code=400,
        )
    changed_fields = {
        field for field in WHAT_IF_FIELDS
        if str(baseline.get(field)) != str(scenario.get(field))
    }
    if not changed_fields:
        return error_response(message="scenario 至少需要修改一个允许的健康因素。", code=400)
    try:
        result = RiskService(current_app.config).predict_what_if(baseline, scenario)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return error_response(message=str(exc), code=400)
    result["changed_fields"] = sorted(changed_fields)
    return success_response(message="模型情景模拟完成。", data=result)


@risk_phase2_bp.route("/batch", methods=["POST"])
def batch_risk():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return error_response(message="rows 必须为非空数组。", code=400)
    if len(rows) > int(current_app.config.get("BATCH_API_MAX_ROWS", 5000)):
        return error_response(message="单次批量评估记录数超过限制，请使用上传任务。", code=413)
    try:
        import pandas as pd

        result = BatchRiskService(current_app.config).score(
            pd.DataFrame(rows),
            include_records=bool(payload.get("include_records", True)),
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="批量双模型评估完成。", data=result)

