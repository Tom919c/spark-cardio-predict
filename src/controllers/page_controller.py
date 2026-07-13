"""页面路由控制器：注册系统所有前端页面入口。"""

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


@pages_bp.route("/shap-analysis")
def shap_analysis_page():
    return render_template("shap_analysis.html")


@pages_bp.route("/result-report")
def result_report_page():
    return render_template("result_report.html")


@pages_bp.route("/favicon.ico")
def favicon():
    return "", 204
