from flask import Blueprint, render_template


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def home_page():
    return render_template("dashboard.html")


@pages_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@pages_bp.route("/risk-report")
def risk_report_page():
    return render_template("risk_report.html")
