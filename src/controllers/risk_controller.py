from flask import Blueprint, current_app, request

from src.services.risk_service import RiskService
from src.utils.response import error_response, success_response

risk_bp = Blueprint("risk", __name__)


@risk_bp.route("/summary")
def risk_summary():
    service = RiskService(current_app.config)
    summary = service.get_platform_summary()
    return success_response(message="风险平台摘要获取成功。", data=summary)


@risk_bp.route("/train")
def train_risk_model():
    service = RiskService(current_app.config)
    run_label = request.args.get("run_label", "").strip()

    try:
        result = service.train_models(run_label=run_label or "phase1")
    except (ValueError, FileNotFoundError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="心脏事件与卒中双模型训练完成。", data=result)


@risk_bp.route("/predict", methods=["POST"])
def predict_single_risk():
    service = RiskService(current_app.config)
    payload = request.get_json(silent=True) or {}

    try:
        result = service.predict_risk(sample=payload)
    except (ValueError, FileNotFoundError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="双模型单人风险评估完成。", data=result)
