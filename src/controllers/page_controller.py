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
    return render_template(
        "risk_report.html",
        default_heart_model_path="C:/Users/1/PycharmProjects/FlaskProject1/data/feature/models/random_forest_heart_round2_20260711_091204.joblib",
        default_stroke_model_path="C:/Users/1/PycharmProjects/FlaskProject1/data/feature/models/random_forest_stroke_round2_20260711_091246.joblib",
    )
