from flask import Blueprint, current_app, request

from src.services.risk_service import RiskService
from src.utils.response import error_response, success_response

risk_bp = Blueprint("risk", __name__)


@risk_bp.route("/summary")
def risk_summary():
    service = RiskService(current_app.config)
    summary = service.get_platform_summary()
    return success_response(message="Risk platform summary loaded successfully.", data=summary)


@risk_bp.route("/train")
def train_risk_model():
    service = RiskService(current_app.config)
    run_label = request.args.get("run_label", "").strip()
    rounds = request.args.get("rounds", "").strip()

    try:
        training_rounds = int(rounds) if rounds else None
    except ValueError:
        return error_response(message="rounds must be a positive integer.", code=400)

    try:
        result = service.train_model(
            run_label=run_label,
            rounds=training_rounds,
        )
    except (ValueError, FileNotFoundError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="Heart and stroke models trained successfully.", data=result)


@risk_bp.route("/predict", methods=["POST"])
def predict_single_risk():
    service = RiskService(current_app.config)
    payload = request.get_json(silent=True) or {}
    heart_model_path = payload.pop("heart_model_path", "").strip()
    stroke_model_path = payload.pop("stroke_model_path", "").strip()

    try:
        result = service.predict_risk(
            sample=payload,
            heart_model_path=heart_model_path or None,
            stroke_model_path=stroke_model_path or None,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        return error_response(message=str(exc), code=400)
    return success_response(message="Dual-risk prediction completed successfully.", data=result)
