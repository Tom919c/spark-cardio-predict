from src.controllers.analysis_controller import analysis_bp
from src.controllers.data_controller import data_bp
from src.controllers.health_controller import health_bp
from src.controllers.page_controller import pages_bp
from src.controllers.risk_controller import risk_bp


def register_blueprints(app):
    app.register_blueprint(health_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(data_bp, url_prefix="/api/data")
    app.register_blueprint(risk_bp, url_prefix="/api/risk")
    app.register_blueprint(analysis_bp, url_prefix="/api/analysis")
