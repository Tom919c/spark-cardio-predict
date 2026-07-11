from pathlib import Path

from flask import Blueprint, current_app, render_template


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/dashboard")
def dashboard_page():
    return render_template(
        "dashboard.html",
        model_output_dir=current_app.config.get("MODEL_OUTPUT_DIR", "data/feature/models"),
    )


@pages_bp.route("/risk-report")
def risk_report_page():
    model_dir = Path(current_app.config.get("MODEL_OUTPUT_DIR", "data/feature/models"))
    heart_model_file = current_app.config.get("DEFAULT_HEART_MODEL_FILE", "")
    stroke_model_file = current_app.config.get("DEFAULT_STROKE_MODEL_FILE", "")

    return render_template(
        "risk_report.html",
        default_heart_model_path=str((model_dir / heart_model_file).as_posix()),
        default_stroke_model_path=str((model_dir / stroke_model_file).as_posix()),
    )


@pages_bp.route("/shap-analysis")
def shap_analysis_page():
    return render_template("shap_analysis.html")


@pages_bp.route("/result-report")
def result_report_page():
    return render_template("result_report.html")
